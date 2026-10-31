"""Kissing number in dimension 11 - vectorized MIS over equal-norm vectors with clique-cutting-plane refinement."""
import numpy as np
import itertools
import time

D = 11


def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # ---- Candidate pool: all vectors with norm^2 == 4 ----
    # Type A: weight-4 vectors in {-1,0,1}^11  (4 positions, +/-1)
    v4_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product((1, -1), repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_list.append(v)
    # Type B: weight-1 vectors in {-2,0,2}^11  (1 position, +/-2)
    for pos in range(D):
        for sign in (1, -1):
            v = [0] * D
            v[pos] = 2 * sign
            v4_list.append(v)

    V = np.array(v4_list, dtype=np.int64)
    N = len(V)
    if N == 0:
        return np.zeros((0, D), dtype=np.int64)

    # ---- Conflict graph: conflict iff dot(v,w) > 2 (norm^2 = 4 -> min dist >= max norm) ----
    dots = V @ V.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)

    # ---- Precompute clique cover for cutting-plane guidance ----
    # Find maximal cliques in the conflict graph to create "at most one per clique" constraints
    # This tightens the LP relaxation and guides better vertex selection
    def find_maximal_cliques(conflict_matrix, max_cliques=500):
        """Find a set of maximal (or near-maximal) cliques in the conflict graph.
        Returns list of arrays, each containing indices of vertices in a clique."""
        cliques = []
        n = len(conflict_matrix)
        # Use a simple greedy clique finder: repeatedly find a vertex with high degree
        # to its unprocessed neighbors, build a clique around it
        unprocessed = set(range(n))
        attempts = 0
        max_attempts = 2000
        while unprocessed and len(cliques) < max_cliques and attempts < max_attempts:
            attempts += 1
            # Pick vertex with highest degree within unprocessed
            best_v = -1
            best_deg = -1
            for v in unprocessed:
                deg = conflict_matrix[v].sum()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            if best_v < 0:
                break
            # Build maximal clique containing best_v using greedy expansion
            clique = {best_v}
            candidates = set(unprocessed)
            candidates.add(best_v)
            # Iteratively add vertices that connect to all current clique members
            changed = True
            while changed:
                changed = False
                for v in list(candidates):
                    if v in clique:
                        continue
                    # Check if v connects to all in clique
                    if all(conflict_matrix[v, u] for u in clique):
                        clique.add(v)
                        changed = True
            if len(clique) >= 2:
                cliques.append(np.array(sorted(clique)))
                unprocessed -= clique
        return cliques

    # Find cliques for cutting-plane constraints
    cliques = find_maximal_cliques(conflicts, max_cliques=300)
    
    # Build clique membership: for each vertex, which cliques contain it
    # And for each clique, the list of vertices
    vertex_cliques = [[] for _ in range(N)]
    for ci, clique in enumerate(cliques):
        for v in clique:
            vertex_cliques[v].append(ci)
    
    # Precompute clique sizes
    clique_sizes = np.array([len(c) for c in cliques]) if cliques else np.array([])

    def greedy_clique_aware(rng):
        """Greedy MIS that prefers vertices in larger cliques (more constrained)
        and breaks ties by considering clique pressure."""
        alive = np.ones(N, dtype=bool)
        selected = np.zeros(N, dtype=bool)
        degrees = conflicts.sum(axis=1).astype(np.int64)
        idx_all = np.arange(N)
        
        # Compute initial "clique pressure" for each vertex: sum of sizes of cliques containing it
        clique_pressure = np.zeros(N, dtype=np.float64)
        for ci, clique in enumerate(cliques):
            for v in clique:
                clique_pressure[v] += len(clique)
        
        while True:
            if time.time() > deadline:
                break
            alive_idx = np.where(alive)[0]
            if len(alive_idx) == 0:
                break
            
            # Score = degree + small weight * clique_pressure (prefer lower degree, but
            # among similar degrees, prefer vertices in larger cliques as they're more "valuable" to resolve)
            alive_deg = degrees[alive_idx].astype(np.float64)
            alive_pressure = clique_pressure[alive_idx]
            # Normalize pressure to comparable scale
            if alive_pressure.max() > 0:
                score = alive_deg + 0.1 * alive_pressure / alive_pressure.max()
            else:
                score = alive_deg
            
            min_score = score.min()
            cands = alive_idx[score <= min_score + 1e-9]
            chosen = cands[rng.integers(len(cands))]
            selected[chosen] = True
            to_remove = np.where(conflicts[chosen] | (idx_all == chosen))[0]
            alive[to_remove] = False
            remaining = np.where(alive)[0]
            if len(remaining) > 0:
                degrees[remaining] -= conflicts[remaining][:, to_remove].sum(axis=1)
        return selected

    def greedy_min_degree(rng):
        """Vectorized greedy: repeatedly pick min-degree alive vertex, remove it + neighbors."""
        alive = np.ones(N, dtype=bool)
        selected = np.zeros(N, dtype=bool)
        degrees = conflicts.sum(axis=1).astype(np.int64)
        idx_all = np.arange(N)
        while True:
            if time.time() > deadline:
                break
            alive_idx = np.where(alive)[0]
            if len(alive_idx) == 0:
                break
            alive_deg = degrees[alive_idx]
            min_deg = alive_deg.min()
            cands = alive_idx[alive_deg == min_deg]
            chosen = cands[rng.integers(len(cands))]
            selected[chosen] = True
            to_remove = np.where(conflicts[chosen] | (idx_all == chosen))[0]
            alive[to_remove] = False
            remaining = np.where(alive)[0]
            if len(remaining) > 0:
                degrees[remaining] -= conflicts[remaining][:, to_remove].sum(axis=1)
        return selected

    def local_search_2swap(selected, rng, max_rounds=300):
        """Try to add one unselected vertex; if it conflicts with selected ones,
        drop those and re-add, then attempt to recover by adding a compatible
        unselected vertex. Accept if net size increases."""
        sel = selected.copy()
        sel_set = set(np.where(sel)[0].tolist())
        cur_size = len(sel_set)
        best_size = cur_size
        best_sel = sel.copy()
        unselected = np.where(~sel)[0]
        if len(unselected) == 0:
            return best_sel, best_size
        for _ in range(max_rounds):
            if time.time() > deadline - 5:
                break
            # Pick a random unselected vertex not already selected
            pool = [u for u in unselected if u not in sel_set]
            if not pool:
                break
            u = pool[rng.integers(len(pool))]
            # Neighbors of u that are currently selected
            nbrs = np.where(conflicts[u])[0]
            drop = [n for n in nbrs if n in sel_set]
            if len(drop) >= 2:
                continue  # net non-positive gain for a 1-swap
            # Remove dropped, add u
            for n in drop:
                sel_set.discard(n)
            sel_set.add(u)
            # Try to recover: add any unselected vertex compatible with all in sel_set
            improved = False
            for w in unselected:
                if w in sel_set:
                    continue
                # check w conflicts with no member of sel_set
                if not np.any(conflicts[w][list(sel_set)]):
                    sel_set.add(w)
                    improved = True
                    break
            new_size = len(sel_set)
            if new_size > best_size:
                best_size = new_size
                best_sel = np.zeros(N, dtype=bool)
                best_sel[list(sel_set)] = True
            else:
                # revert
                sel_set.discard(u)
                for n in drop:
                    sel_set.add(n)
        return best_sel, best_size

    def clique_constrained_local_search(selected, rng, max_rounds=400):
        """Enhanced local search that uses clique constraints to guide swaps.
        When considering adding a vertex, prioritize dropping selected vertices
        that are in the same cliques as the candidate (tighter constraints)."""
        sel = selected.copy()
        sel_set = set(np.where(sel)[0].tolist())
        best_size = len(sel_set)
        best_sel = sel.copy()
        unselected = np.where(~sel)[0]
        if len(unselected) == 0:
            return best_sel, best_size
        
        # Precompute for each vertex: list of cliques it belongs to
        v_cliques = {v: vertex_cliques[v] for v in range(N)}
        
        for _ in range(max_rounds):
            if time.time() > deadline - 5:
                break
            pool = [u for u in unselected if u not in sel_set]
            if not pool:
                break
            u = pool[rng.integers(len(pool))]
            nbrs = np.where(conflicts[u])[0]
            drop_candidates = [n for n in nbrs if n in sel_set]
            if len(drop_candidates) >= 2:
                continue
            
            # Score drop candidates: prefer dropping those in fewer cliques (less constraint loss)
            # or those in cliques not shared with u (so we don't violate clique constraints unnecessarily)
            if drop_candidates:
                u_cliques_set = set(v_cliques.get(u, []))
                def drop_score(n):
                    n_cliques = set(v_cliques.get(n, []))
                    shared = len(n_cliques & u_cliques_set)
                    total = len(n_cliques)
                    # Lower score = better to drop (less constraint impact)
                    return total - shared * 0.5
                drop_candidates.sort(key=drop_score)
                drop = drop_candidates[:len(drop_candidates)]  # Take all if < 2
            else:
                drop = []
            
            # Remove dropped, add u
            for n in drop:
                sel_set.discard(n)
            sel_set.add(u)
            
            # Try to recover: add any unselected vertex compatible with all in sel_set
            improved = False
            for w in unselected:
                if w in sel_set:
                    continue
                if not np.any(conflicts[w][list(sel_set)]):
                    sel_set.add(w)
                    improved = True
                    break
            
            new_size = len(sel_set)
            if new_size > best_size:
                best_size = new_size
                best_sel = np.zeros(N, dtype=bool)
                best_sel[list(sel_set)] = True
            else:
                sel_set.discard(u)
                for n in drop:
                    sel_set.add(n)
        return best_sel, best_size

    # ---- Multiple random restarts, mixing greedy strategies ----
    best_selected = None
    best_count = 0
    n_restarts = 50
    for r in range(n_restarts):
        if time.time() > deadline - 15:
            break
        rng = np.random.default_rng(12345 + r * 97)
        # Alternate between standard greedy and clique-aware greedy
        if r % 2 == 0:
            sel = greedy_min_degree(rng)
        else:
            sel = greedy_clique_aware(rng)
        cnt = int(sel.sum())
        if cnt > best_count:
            best_count = cnt
            best_selected = sel.copy()

    if best_selected is None:
        best_selected = greedy_min_degree(np.random.default_rng(0))
        best_count = int(best_selected.sum())

    # ---- Local search refinement on the best solution ----
    if time.time() < deadline - 10:
        rng = np.random.default_rng(999)
        best_selected, best_count = local_search_2swap(best_selected, rng, max_rounds=500)
    
    # ---- Second pass with clique-constrained local search ----
    if time.time() < deadline - 5:
        rng = np.random.default_rng(1234)
        best_selected, best_count = clique_constrained_local_search(best_selected, rng, max_rounds=400)

    result = V[best_selected]
    return result
