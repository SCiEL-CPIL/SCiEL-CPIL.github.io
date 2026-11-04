"""Kissing number in dimension 11: construct nonzero integer vectors in R^11
whose minimum pairwise distance is at least the maximum vector norm.
Interface: construct() -> integer array of shape (k, d)."""

import itertools
import time
import numpy as np

D = 11
DEADLINE_S = 118.0  # hard wall-clock safety margin before the 120 s limit


def construct():
    t0 = time.perf_counter()
    best = None
    best_k = 0

    def now():
        return time.perf_counter()

    def elapsed():
        return now() - t0

    def is_deadline():
        return elapsed() > DEADLINE_S

    def update_best(arr):
        nonlocal best, best_k
        if arr is None:
            return
        k = arr.shape[0]
        if k > best_k:
            best = arr
            best_k = k

    # ------------------------------------------------------------------
    # 1) Fast deterministic baseline: the full "2-weight" shell of norm^2=2
    #    There are exactly C(11,2)*2^2 = 220 such vectors, and they satisfy
    #    min pairwise distance (sqrt(2) for shared coordinate) == max norm (sqrt(2)).
    # ------------------------------------------------------------------
    base = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                base.append(v)
    base_arr = np.array(base, dtype=np.int64)
    update_best(base_arr)

    # ------------------------------------------------------------------
    # 2) Generate all vectors with norm^2 = m
    # ------------------------------------------------------------------
    def gen_all_vectors_m(m):
        """Generate all integer vectors in R^D with L2^2 = m."""
        if m < 1:
            return []

        results = []

        if m == 2:
            # Two ±1 coords
            for i, j in itertools.combinations(range(D), 2):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i], v[j] = si, sj
                        results.append(v)
            # One ±2 coord
            for i in range(D):
                for s in (2, -2):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = s
                    results.append(v)

        elif m == 3:
            # Three ±1 coords
            for combo in itertools.combinations(range(D), 3):
                for signs in itertools.product([1, -1], repeat=3):
                    v = np.zeros(D, dtype=np.int64)
                    for idx, s in zip(combo, signs):
                        v[idx] = s
                    results.append(v)

        elif m == 4:
            # Four ±1 coords
            for combo in itertools.combinations(range(D), 4):
                for signs in itertools.product([1, -1], repeat=4):
                    v = np.zeros(D, dtype=np.int64)
                    for idx, s in zip(combo, signs):
                        v[idx] = s
                    results.append(v)
            # One ±2 coord
            for i in range(D):
                for s in (2, -2):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = s
                    results.append(v)

        elif m == 5:
            # Five ±1 coords
            for combo in itertools.combinations(range(D), 5):
                for signs in itertools.product([1, -1], repeat=5):
                    v = np.zeros(D, dtype=np.int64)
                    for idx, s in zip(combo, signs):
                        v[idx] = s
                    results.append(v)
            # One ±1 and one ±2 coord
            for i, j in itertools.combinations(range(D), 2):
                for si in (1, -1):
                    for sj in (2, -2):
                        v = np.zeros(D, dtype=np.int64)
                        v[i], v[j] = si, sj
                        results.append(v)

        elif m == 6:
            # Six ±1 coords
            for combo in itertools.combinations(range(D), 6):
                for signs in itertools.product([1, -1], repeat=6):
                    v = np.zeros(D, dtype=np.int64)
                    for idx, s in zip(combo, signs):
                        v[idx] = s
                    results.append(v)
            # One ±2 and two ±1 coords
            for combo in itertools.combinations(range(D), 3):
                for signs2 in (2, -2):
                    for signs1 in itertools.product([1, -1], repeat=2):
                        v = np.zeros(D, dtype=np.int64)
                        v[combo[0]] = signs2
                        v[combo[1]] = signs1[0]
                        v[combo[2]] = signs1[1]
                        results.append(v)

        return results

    # ------------------------------------------------------------------
    # 3) Maximum Independent Set solver via branch-and-bound with coloring
    #    This is the structural change: instead of stochastic local search,
    #    we use a deterministic B&B with greedy coloring upper bounds.
    # ------------------------------------------------------------------
    def greedy_color_order(adj_matrix, n, deadline_fn):
        """
        Greedy coloring of the graph to get an upper bound on MIS size.
        Returns: (color_bound, color_order) where color_order is a permutation
        that groups vertices by color class (for branch-and-bound ordering).
        
        adj_matrix: (n, n) boolean, True means edge (conflict)
        """
        # Greedy coloring: process vertices in order, assign smallest available color
        colors = np.zeros(n, dtype=np.int32)
        color_of = [-1] * n
        available = [True] * 200  # max colors needed
        
        # Process in order of decreasing degree for better bounds
        degrees = adj_matrix.sum(axis=1)
        order = np.argsort(-degrees)
        
        for v in order:
            # Find smallest color not used by neighbors
            used = set()
            for u in range(n):
                if adj_matrix[v, u] and color_of[u] >= 0:
                    used.add(color_of[u])
            for c in range(200):
                if c not in used:
                    color_of[v] = c
                    break
        
        max_color = max(color_of) + 1 if n > 0 else 0
        
        # Build color order: sort by color, then within color by degree
        color_order = sorted(range(n), key=lambda i: (color_of[i], -degrees[i]))
        
        return max_color, color_order, color_of

    def branch_and_bound_mis(adj_matrix, n, deadline_fn, time_limit=30.0):
        """
        Branch and bound for Maximum Independent Set.
        adj_matrix: (n, n) boolean, True = edge in conflict graph
        Returns: list of indices in the MIS
        """
        if n == 0:
            return []
        
        t_start = time.perf_counter()
        best_set = []
        best_size = 0
        
        # Greedy initialization: min-degree greedy
        remaining = set(range(n))
        greedy_set = []
        while remaining:
            # Pick vertex with min degree in remaining subgraph
            min_deg = float('inf')
            min_v = None
            for v in remaining:
                deg = sum(1 for u in remaining if u != v and adj_matrix[v, u])
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
            greedy_set.append(min_v)
            remaining.discard(min_v)
            # Remove all neighbors of min_v
            remaining = {u for u in remaining if not adj_matrix[min_v, u]}
        
        best_set = list(greedy_set)
        best_size = len(best_set)
        
        # If greedy already gives a large set, use it
        if best_size >= 0.7 * n:
            return best_set
        
        # Branch and bound
        # We'll use a recursive approach with pruning
        # State: (candidates, current_set, current_size)
        # candidates: set of vertices that can still be added (no conflicts with current_set)
        
        # Precompute adjacency as lists for speed
        adj_lists = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i, j]:
                    adj_lists[i].add(j)
                    adj_lists[j].add(i)
        
        # Get initial coloring for upper bound
        color_bound, color_order, color_of = greedy_color_order(adj_matrix, n, deadline_fn)
        
        # Use iterative deepening or simple B&B
        # For simplicity, use a stack-based B&B
        
        # State: (candidates_frozenset, current_set_list)
        # But frozensets are expensive for large n. Use a different approach.
        
        # Actually, for n=330, let's use a more efficient approach:
        # Process vertices in color_order, at each step decide include/exclude
        
        # Better: use the standard B&B where we maintain a candidate set
        # and use coloring to bound
        
        # Let's use a simpler but effective approach:
        # Recursive B&B with memoization-like pruning
        
        def bb(candidates, current_set, depth):
            nonlocal best_set, best_size
            
            if deadline_fn() or (time.perf_counter() - t_start) > time_limit:
                return
            
            if not candidates:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = list(current_set)
                return
            
            # Upper bound: current_size + max possible from candidates
            # Use coloring of the candidate subgraph
            cand_list = list(candidates)
            cand_n = len(cand_list)
            
            if len(current_set) + cand_n <= best_size:
                return
            
            # Greedy color the candidate subgraph for tighter bound
            # Build local adjacency
            cand_set = set(cand_list)
            local_adj = {}
            for v in cand_list:
                local_adj[v] = {u for u in cand_list if u != v and adj_matrix[v, u]}
            
            # Greedy coloring
            local_colors = {}
            for v in cand_list:
                used = set()
                for u in local_adj[v]:
                    if u in local_colors:
                        used.add(local_colors[u])
                c = 0
                while c in used:
                    c += 1
                local_colors[v] = c
            
            max_local_color = max(local_colors.values()) + 1 if local_colors else 0
            
            if len(current_set) + max_local_color <= best_size:
                return
            
            # Branch: pick a vertex to branch on
            # Use max-degree in candidate subgraph
            max_deg = -1
            branch_v = None
            for v in cand_list:
                deg = len(local_adj[v])
                if deg > max_deg:
                    max_deg = deg
                    branch_v = v
            
            # Branch 1: include branch_v
            new_candidates_1 = cand_set - {branch_v} - local_adj[branch_v]
            current_set.append(branch_v)
            bb(new_candidates_1, current_set, depth + 1)
            current_set.pop()
            
            # Branch 2: exclude branch_v
            new_candidates_2 = cand_set - {branch_v}
            bb(new_candidates_2, current_set, depth + 1)
        
        # Start B&B
        all_candidates = set(range(n))
        bb(all_candidates, [], 0)
        
        return best_set

    # ------------------------------------------------------------------
    # 4) For the 4-subset problem (m=4 shell), build conflict graph and solve MIS
    # ------------------------------------------------------------------
    def solve_4subset_mis(deadline_fn, time_limit=60.0):
        """
        Solve the maximum independent set problem for the 4-subset conflict graph.
        Two 4-subsets conflict if they share >= 3 elements.
        Returns the set of selected subset indices.
        """
        subsets = list(itertools.combinations(range(D), 4))
        n = len(subsets)  # 330
        
        # Build conflict matrix
        # Two 4-sets conflict if |S ∩ T| >= 3
        masks = np.zeros(n, dtype=np.int64)
        for i, s in enumerate(subsets):
            m = 0
            for pos in s:
                m |= (1 << pos)
            masks[i] = m
        
        # Compute pairwise intersection sizes using popcount
        popcount_table = np.zeros(2048, dtype=np.int32)
        for i in range(2048):
            popcount_table[i] = bin(i).count('1')
        
        m1 = masks[:, np.newaxis]
        m2 = masks[np.newaxis, :]
        and_matrix = m1 & m2
        popcount_matrix = popcount_table[and_matrix]
        
        # Conflict if intersection >= 3
        conflict = (popcount_matrix >= 3)
        np.fill_diagonal(conflict, False)  # No self-conflict
        
        # Solve MIS on this conflict graph
        t_start = time.perf_counter()
        selected_indices = branch_and_bound_mis(conflict, n, deadline_fn, time_limit)
        
        return subsets, selected_indices

    # ------------------------------------------------------------------
    # 5) Build vectors from selected 4-subsets
    # ------------------------------------------------------------------
    def build_vectors_from_subsets(subsets, selected_indices):
        """Build the integer vectors for the selected 4-subsets."""
        vecs = []
        for idx in selected_indices:
            combo = subsets[idx]
            # For each 4-subset, we need to choose signs.
            # The standard construction uses all +1 signs for the base,
            # but we can also try other sign patterns.
            # For simplicity, use all +1 first.
            v = np.zeros(D, dtype=np.int64)
            for pos in combo:
                v[pos] = 1
            vecs.append(v)
        
        if not vecs:
            return np.zeros((0, D), dtype=np.int64)
        return np.array(vecs, dtype=np.int64)

    # ------------------------------------------------------------------
    # 6) Try the 4-subset MIS approach
    # ------------------------------------------------------------------
    if not is_deadline():
        try:
            subsets, selected_indices = solve_4subset_mis(is_deadline, time_limit=60.0)
            if selected_indices:
                V = build_vectors_from_subsets(subsets, selected_indices)
                if V.shape[0] > 0:
                    update_best(V)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 7) Also try greedy + local search on the 4-subset problem as backup
    # ------------------------------------------------------------------
    if not is_deadline():
        try:
            subsets = list(itertools.combinations(range(D), 4))
            n = len(subsets)
            
            masks = np.zeros(n, dtype=np.int64)
            for i, s in enumerate(subsets):
                m = 0
                for pos in s:
                    m |= (1 << pos)
                masks[i] = m
            
            popcount_table = np.zeros(2048, dtype=np.int32)
            for i in range(2048):
                popcount_table[i] = bin(i).count('1')
            
            m1 = masks[:, np.newaxis]
            m2 = masks[np.newaxis, :]
            and_matrix = m1 & m2
            popcount_matrix = popcount_table[and_matrix]
            
            compatible = (popcount_matrix <= 2)
            np.fill_diagonal(compatible, True)
            
            # Greedy: min-degree
            remaining = set(range(n))
            selected = set()
            
            while remaining:
                min_deg = float('inf')
                min_v = None
                for v in remaining:
                    deg = sum(1 for u in remaining if u != v and not compatible[v, u])
                    if deg < min_deg:
                        min_deg = deg
                        min_v = v
                selected.add(min_v)
                remaining.discard(min_v)
                remaining = {u for u in remaining if compatible[min_v, u]}
            
            # Local search: try swaps
            rng = np.random.default_rng(42)
            current_selected = set(selected)
            current_size = len(current_selected)
            best_selected_ls = set(current_selected)
            best_size_ls = current_size
            
            t_start_ls = time.perf_counter()
            T = 5.0
            cooling = 0.9995
            
            while (time.perf_counter() - t_start_ls) < 30.0 and not is_deadline():
                move_type = rng.choice(['swap', 'flip_in', 'flip_out'])
                
                if move_type == 'swap':
                    if not current_selected or not (set(range(n)) - current_selected):
                        continue
                    out_idx = rng.choice(list(current_selected))
                    in_candidates = [i for i in (set(range(n)) - current_selected) 
                                     if all(compatible[in_idx, j] for j in current_selected if j != out_idx)]
                    if not in_candidates:
                        continue
                    in_idx = rng.choice(in_candidates)
                    
                    new_selected = (current_selected - {out_idx}) | {in_idx}
                    # Check validity
                    valid = True
                    for i in new_selected:
                        for j in new_selected:
                            if i != j and not compatible[i, j]:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        new_size = len(new_selected)
                        delta = new_size - current_size
                        if delta >= 0 or (T > 0.01 and rng.random() < np.exp(delta / T)):
                            current_selected = new_selected
                            current_size = new_size
                            if current_size > best_size_ls:
                                best_size_ls = current_size
                                best_selected_ls = set(current_selected)
                
                elif move_type == 'flip_in':
                    unselected = list(set(range(n)) - current_selected)
                    if not unselected:
                        continue
                    idx = rng.choice(unselected)
                    if all(compatible[idx, j] for j in current_selected):
                        current_selected.add(idx)
                        current_size += 1
                        if current_size > best_size_ls:
                            best_size_ls = current_size
                            best_selected_ls = set(current_selected)
                
                elif move_type == 'flip_out':
                    if not current_selected:
                        continue
                    idx = rng.choice(list(current_selected))
                    current_selected.discard(idx)
                    current_size -= 1
                
                T *= cooling
            
            if best_selected_ls:
                V = build_vectors_from_subsets(subsets, list(best_selected_ls))
                if V.shape[0] > 0:
                    update_best(V)
                    
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 8) Multi-order greedy on each shell (from current program)
    # ------------------------------------------------------------------
    for m in [2, 3, 4, 5, 6]:
        if is_deadline():
            break
        
        t_start = elapsed()
        try:
            vecs = gen_all_vectors_m(m)
            if len(vecs) == 0:
                continue
            
            V = np.array(vecs, dtype=np.int64)
            n = V.shape[0]
            max_ip_allowed = m // 2
            
            G = V.astype(np.int64) @ V.astype(np.int64).T
            G = np.ascontiguousarray(G)
            
            best_chosen = []
            best_len = 0
            
            rng = np.random.RandomState(42 + m)
            n_trials = min(30, max(5, 60 // max(1, n // 1000)))
            
            for trial in range(n_trials):
                if is_deadline() or elapsed() > t_start + 20:
                    break
                
                if trial == 0:
                    order = list(range(n))
                elif trial == 1:
                    order = list(range(n - 1, -1, -1))
                else:
                    order = rng.permutation(n).tolist()
                
                chosen_mask = np.zeros(n, dtype=bool)
                max_ip_with_chosen = np.full(n, -999999, dtype=np.int64)
                chosen = []
                
                for idx in order:
                    if is_deadline() or elapsed() > t_start + 20:
                        break
                    if chosen_mask[idx]:
                        continue
                    if max_ip_with_chosen[idx] <= max_ip_allowed:
                        chosen.append(idx)
                        chosen_mask[idx] = True
                        max_ip_with_chosen = np.maximum(max_ip_with_chosen, G[idx])
                        max_ip_with_chosen[chosen_mask] = -999999
                
                if len(chosen) > best_len:
                    best_len = len(chosen)
                    best_chosen = chosen
            
            if best_chosen:
                update_best(V[best_chosen])
                
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 9) Safety: return best found
    # ------------------------------------------------------------------
    if best is None or best.shape[0] == 0:
        return base_arr
    return best
