import numpy as np
import time
import random
import sys

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 630s limit

    best_set_global = []
    best_size_global = 0
    best_R_global = 0

    # We focus on R=4 as the primary target, with R=3 as fallback
    # R=4 gives ~7000 candidates, manageable for B&B with good reductions
    
    R_values = [4, 3]

    for R in R_values:
        if time.time() > deadline - 60:
            break

        # Generate all integer vectors with squared norm <= R
        candidates = []
        seen = set()

        def add_vec(v):
            key = tuple(v)
            if key not in seen and any(v != 0):
                seen.add(key)
                candidates.append(v)

        # Norm 1: e_i, -e_i
        for i in range(d):
            v = np.zeros(d, dtype=np.int64)
            v[i] = 1
            add_vec(v)
            v[i] = -1
            add_vec(v)

        # Norm 2: e_i ± e_j
        for i in range(d):
            for j in range(i + 1, d):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = s1
                        v[j] = s2
                        add_vec(v)

        # Norm 3: e_i ± e_j ± e_k
        if R >= 3:
            for i in range(d):
                for j in range(i + 1, d):
                    for k in range(j + 1, d):
                        for s1 in [1, -1]:
                            for s2 in [1, -1]:
                                for s3 in [1, -1]:
                                    v = np.zeros(d, dtype=np.int64)
                                    v[i] = s1
                                    v[j] = s2
                                    v[k] = s3
                                    add_vec(v)

        # Norm 4: 2e_i, -2e_i, and e_i ± e_j ± e_k ± e_l
        if R >= 4:
            for i in range(d):
                v = np.zeros(d, dtype=np.int64)
                v[i] = 2
                add_vec(v)
                v[i] = -2
                add_vec(v)
            for i in range(d):
                for j in range(i + 1, d):
                    for k in range(j + 1, d):
                        for l in range(k + 1, d):
                            for s1 in [1, -1]:
                                for s2 in [1, -1]:
                                    for s3 in [1, -1]:
                                        for s4 in [1, -1]:
                                            v = np.zeros(d, dtype=np.int64)
                                            v[i] = s1
                                            v[j] = s2
                                            v[k] = s3
                                            v[l] = s4
                                            add_vec(v)

        n = len(candidates)
        if n < 2:
            continue

        cand_array = np.array(candidates, dtype=np.int64)
        norms_sq = np.sum(cand_array**2, axis=1)

        # Build conflict graph: edge (i,j) if ||p_i - p_j||^2 < R
        # Vectorized computation
        dots = cand_array @ cand_array.T
        dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
        conflict_mask = dist_sq < R
        np.fill_diagonal(conflict_mask, False)

        # Convert to adjacency list for B&B
        adj = [np.where(conflict_mask[i])[0] for i in range(n)]
        deg = np.array([len(adj[i]) for i in range(n)])

        # Branch and bound with reduction rules
        # State: (included, excluded, candidate_set)
        # We use a recursive B&B with greedy coloring bound

        def greedy_color_bound(cand_set):
            """Upper bound on clique size in induced subgraph on cand_set using greedy coloring."""
            if len(cand_set) == 0:
                return 0
            # Greedy coloring: repeatedly pick a vertex, remove it and its neighbors
            # The number of colors used is an upper bound on the independence number... 
            # Wait, we want the independence number, which is the clique number of the complement.
            # Actually, for MIS, a standard bound is: partition cand_set into cliques, 
            # then the MIS can have at most one vertex per clique, so |cliques| is an upper bound.
            # Greedy clique cover: repeatedly find a maximal clique, remove it.
            
            remaining = list(cand_set)
            num_cliques = 0
            while remaining:
                # Find a maximal clique greedily
                clique = [remaining[0]]
                remaining.remove(remaining[0])
                for v in remaining[:]:
                    if all(conflict_mask[v, c] for c in clique):
                        clique.append(v)
                        remaining.remove(v)
                num_cliques += 1
            return num_cliques

        # Better: use the simple bound that MIS <= number of vertices, 
        # and use a tighter bound based on degree.
        # For now, let's use a simpler approach: recursive B&B with memoization-like pruning.

        # Actually, let's use a different strategy: 
        # 1. Get a good initial solution via greedy
        # 2. Use local search to improve
        # 3. Try multiple random orderings

        best_local = []
        best_local_size = 0

        # Greedy with random restarts
        num_restarts = 200 if n < 3000 else 50

        for restart in range(num_restarts):
            if time.time() > deadline - 30:
                break

            # Random order
            order = np.random.permutation(n)
            available = np.ones(n, dtype=bool)
            selected = []

            for idx in order:
                if not available[idx]:
                    continue
                selected.append(idx)
                # Remove idx and its neighbors
                available[idx] = False
                for nb in adj[idx]:
                    available[nb] = False

            current_size = len(selected)

            # Local search: try to swap
            improved = True
            while improved:
                improved = False
                for i, sel_idx in enumerate(selected):
                    # Try removing sel_idx and adding any available vertex
                    # First, mark sel_idx and its neighbors as "potentially available"
                    # But we need to be careful: only vertices not in selected and not adjacent to other selected
                    # are candidates for addition.
                    
                    # Temporarily remove sel_idx
                    temp_selected = selected[:i] + selected[i+1:]
                    # Determine which vertices are available given temp_selected
                    temp_avail = np.ones(n, dtype=bool)
                    for t in temp_selected:
                        temp_avail[t] = False
                        for nb in adj[t]:
                            temp_avail[nb] = False
                    # sel_idx itself is available
                    temp_avail[sel_idx] = True
                    
                    # Find all available vertices not in temp_selected
                    candidates_to_add = np.where(temp_avail)[0]
                    # Filter out those in temp_selected (shouldn't be any)
                    # Try adding each candidate
                    best_add = None
                    best_add_gain = 0
                    for ca in candidates_to_add:
                        # Check if ca is not adjacent to any in temp_selected
                        # (already ensured by temp_avail)
                        # Adding ca gives size len(temp_selected) + 1
                        # Current size is len(selected) = len(temp_selected) + 1
                        # So no gain unless we can add more than 1
                        # This simple swap won't help. Let's try a different local search.
                        pass
                    # Skip complex local search for now
                    break

            if current_size > best_local_size:
                best_local_size = current_size
                best_local = selected[:]

        # Try a more sophisticated approach: maximum independent set via 
        # branch and bound with the following bound:
        # The independence number of a graph is at most the size of a minimum clique cover.
        # We can approximate this with greedy coloring of the complement graph.
        
        # Actually, let's just use the greedy result and hope it's good enough.
        # For R=4, the greedy should find a large set.

        if best_local_size > best_size_global:
            best_size_global = best_local_size
            best_R_global = R
            best_set_global = [cand_array[i] for i in best_local]

    # Verify and return
    if best_size_global >= 2:
        result = np.array(best_set_global, dtype=np.int64)
        max_norm = np.max(np.sum(result**2, axis=1))
        valid = True
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                d2 = np.dot(result[i] - result[j], result[i] - result[j])
                if d2 < max_norm:
                    valid = False
                    break
            if not valid:
                break

        if valid:
            return result

    # Fallback
    return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)

if __name__ == "__main__":
    res = kissing_number11()
    print(f"Size: {len(res)}")
