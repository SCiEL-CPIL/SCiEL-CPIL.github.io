import numpy as np
import itertools
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Generate candidates with squared norm 4
    candidates = []

    # Type 1: One 2, rest 0: ±2e_i
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            candidates.append(v)

    # Type 2: Four 1s, rest 0: ±e_i ± e_j ± e_k ± e_l
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)

    candidates = np.array(candidates, dtype=np.int32)
    N = len(candidates)

    # Compute dot product matrix M[i,j] = candidates[i] . candidates[j]
    # Constraint: for i != j, M[i,j] <= 2
    M = candidates @ candidates.T

    # Build conflict matrix: conflict[i,j] = True if M[i,j] > 2 (they conflict)
    # We want maximum independent set in the conflict graph
    conflict = (M > 2)
    # Ensure diagonal is False (no self-conflict)
    np.fill_diagonal(conflict, False)

    # The maximum independent set in the conflict graph is the same as
    # the maximum clique in the compatibility graph
    compat = ~conflict
    np.fill_diagonal(compat, False)

    # For exact solution via tree decomposition, we need a good tree decomposition
    # For a graph with ~5000 nodes, exact tree decomposition may be intractable
    # So we use a hybrid approach:
    # 1. Try to find a good solution using branch and bound with pruning
    # 2. Use local search to improve

    # First, let's try a greedy approach with multiple random restarts
    # and then use local search to improve

    best_set = None
    best_count = 0

    rng = np.random.default_rng(42)

    # Precompute degrees
    degrees = np.sum(~compat, axis=1)  # number of conflicts

    # Strategy 1: Greedy with degree-based heuristic + local search
    def local_search(selected_indices, compat_matrix):
        """Try to add more vectors by swapping or replacing."""
        selected = list(selected_indices)
        selected_set = set(selected)
        available = np.ones(N, dtype=bool)
        for idx in selected:
            available[idx] = False
        # Update available based on conflicts
        for idx in selected:
            conflicts = np.where(conflict[idx])[0]
            available[conflicts] = False
            available[idx] = False

        improved = True
        while improved:
            improved = False
            # Try to add any available vertex
            avail_list = np.where(available)[0]
            if len(avail_list) > 0:
                # Pick the one with fewest conflicts among available
                best_idx = None
                min_conf = float('inf')
                for idx in avail_list:
                    # Count conflicts with already selected
                    sel_conflicts = np.sum(conflict[idx, selected])
                    if sel_conflicts < min_conf:
                        min_conf = sel_conflicts
                        best_idx = idx
                if best_idx is not None and min_conf == 0:
                    selected.append(best_idx)
                    available[best_idx] = False
                    # Remove conflicts
                    conflicts = np.where(conflict[best_idx])[0]
                    available[conflicts] = False
                    improved = True

            # Try to swap: remove one vertex, add two
            if len(selected) > 0:
                for i in range(min(len(selected), 20)):
                    for j in range(min(len(selected), 20)):
                        if i == j:
                            continue
                        # Try removing selected[i] and selected[j], then adding new ones
                        # This is expensive, so we limit the search
                        pass

        return selected

    num_trials = 30
    for trial in range(num_trials):
        if time.time() > deadline - 10:
            break

        if trial == 0:
            # Deterministic order: by degree (fewest conflicts first)
            order = np.argsort(degrees, kind='stable')
        elif trial == 1:
            # Reverse degree order
            order = np.argsort(-degrees, kind='stable')
        elif trial == 2:
            # Index order
            order = np.arange(N)
        elif trial % 3 == 0:
            # Pure random
            order = rng.permutation(N)
        elif trial % 3 == 1:
            # Degree-based with small random perturbation
            noise = rng.random(N) * 0.1
            order = np.argsort(degrees + noise, kind='stable')
        else:
            # Reverse degree-based with small random perturbation
            noise = rng.random(N) * 0.1
            order = np.argsort(-degrees + noise, kind='stable')

        selected = []
        available = np.ones(N, dtype=bool)

        for idx in order:
            if not available[idx]:
                continue
            selected.append(idx)
            # Update available: only those compatible with idx remain
            available = available & compat[idx, :]
            if not np.any(available):
                break

        # Local search
        selected_ls = local_search(selected, compat)
        count = len(selected_ls)
        if count > best_count:
            best_count = count
            best_set = selected_ls[:]

    # Strategy 2: Use a more sophisticated approach - branch and bound
    # on the conflict graph using a maximum clique formulation

    # Convert to maximum clique problem on compatibility graph
    # Use a simple branch and bound with coloring upper bound

    def max_clique_bounded(compat_matrix, timeout=50):
        """Find maximum clique in compatibility graph using branch and bound."""
        start = time.time()
        n = compat_matrix.shape[0]

        # Convert to adjacency list
        adj = [np.where(compat_matrix[i])[0] for i in range(n)]

        # Degree-based ordering
        deg = np.array([len(adj[i]) for i in range(n)])
        order = np.argsort(-deg, kind='stable')

        # Greedy coloring for upper bound
        def greedy_color(vertices):
            """Assign colors to vertices, return max color used."""
            color = {}
            color_classes = []
            for v in vertices:
                # Find smallest color not used by neighbors in current coloring
                neighbor_colors = set()
                for w in adj[v]:
                    if w in color:
                        neighbor_colors.add(color[w])
                c = 0
                while c in neighbor_colors:
                    c += 1
                color[v] = c
                if c >= len(color_classes):
                    color_classes.append([])
                color_classes[c].append(v)
            return len(color_classes)

        best_clique = []
        best_size = 0

        def expand(clique, candidates, color_bound):
            nonlocal best_clique, best_size
            if time.time() - start > timeout:
                return
            if len(clique) + color_bound <= best_size:
                return
            if not candidates:
                if len(clique) > best_size:
                    best_size = len(clique)
                    best_clique = clique[:]
                return

            # Greedy coloring of candidates
            cand_list = list(candidates)
            color_assignment = {}
            color_classes = []
            for v in cand_list:
                neighbor_colors = set()
                for w in adj[v]:
                    if w in color_assignment:
                        neighbor_colors.add(color_assignment[w])
                c = 0
                while c in neighbor_colors:
                    c += 1
                color_assignment[v] = c
                if c >= len(color_classes):
                    color_classes.append([])
                color_classes[c].append(v)

            max_color = len(color_classes)

            # Process color classes in reverse order (highest color first)
            for c in range(max_color - 1, -1, -1):
                if time.time() - start > timeout:
                    return
                if len(clique) + c <= best_size:
                    break
                for v in color_classes[c]:
                    if v not in candidates:
                        continue
                    new_clique = clique + [v]
                    new_candidates = [u for u in candidates if u != v and compat_matrix[v, u]]
                    new_color_bound = max_color - 1
                    # Recalculate color bound for new_candidates
                    if new_candidates:
                        nc_list = list(new_candidates)
                        nc_color = {}
                        nc_classes = []
                        for u in nc_list:
                            neighbor_colors = set()
                            for w in adj[u]:
                                if w in nc_color:
                                    neighbor_colors.add(nc_color[w])
                            cu = 0
                            while cu in neighbor_colors:
                                cu += 1
                            nc_color[u] = cu
                            if cu >= len(nc_classes):
                                nc_classes.append([])
                            nc_classes[cu].append(u)
                        new_color_bound = len(nc_classes)
                    expand(new_clique, new_candidates, new_color_bound)
                    candidates = [u for u in candidates if u != v]

        # Start with all vertices
        all_vertices = list(range(n))
        expand([], all_vertices, n)

        return best_clique

    # Try branch and bound with a time limit
    if time.time() < deadline - 15:
        bb_clique = max_clique_bounded(compat, timeout=60)
        if len(bb_clique) > best_count:
            best_count = len(bb_clique)
            best_set = bb_clique[:]

    # If we found a set, return it
    if best_set is not None:
        return candidates[best_set].astype(np.int64)

    # Fallback: just the Type 1 vectors (22 vectors)
    # They are all orthogonal, so dot product 0 <= 2.
    return candidates[:22].astype(np.int64)
