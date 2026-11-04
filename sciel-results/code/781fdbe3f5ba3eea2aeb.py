import numpy as np
import itertools
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

D = 11

def construct():
    # Generate candidate vectors of different norm types
    candidates = []
    weights = []  # weight = 1 for each vector (maximize count)
    
    # Type 1: Norm squared 2 (two coordinates +/- 1) - 220 vectors
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
                weights.append(1)
    
    # Type 2: Norm squared 4 (one coordinate +/- 2) - 22 vectors
    for i in range(D):
        for s in (2, -2):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
            weights.append(1)
    
    # Type 3: Norm squared 8 vectors - need to identify 96 such vectors
    # Norm squared 8: combinations of coordinates that sum of squares = 8
    # Options: (2,2,0,...) -> 4+4=8, or (2,1,1,0,...) -> 4+1+1=6 no
    # (2,2): C(11,2)*4 = 55*4 = 220... too many
    # Let's try: one coordinate ±2, two coordinates ±1: 4+1+1=6, no
    # Two coordinates ±2: 4+4=8. C(11,2)*4 = 220
    # One coordinate ±2, four coordinates... no
    # Four coordinates ±1: 1+1+1+1=4, no
    # (2,2,1,1) would be 4+4+1+1=10, no
    # Actually maybe: (2,2) gives norm sq 8. But that's 220 vectors.
    # The directive says 96 mixed-norm vectors. Let me think differently.
    # Maybe norm squared 8 with specific structure.
    # Or perhaps the "mixed-norm" refers to vectors that mix types in the configuration.
    # For now, let's include norm-sq-4 vectors (the 22 axis-like ones)
    # and see if the MWIS finds a better combination.
    
    n = len(candidates)
    if n == 0:
        return np.array([], dtype=np.int64).reshape(0, D)
    
    candidates_arr = np.array(candidates)
    
    # Build conflict graph: two vectors conflict if they cannot both be in the set
    # Two vectors u, v conflict if |u-v| < max(|u|, |v|)
    # But max norm depends on the set, so this is tricky.
    # Alternative: for each possible max norm squared M, build graph where
    # u,v conflict if |u-v|^2 < M.
    # We want to maximize |S| such that for all u,v in S, |u-v|^2 >= max_{w in S} |w|^2
    
    # Strategy: try different max norm squared values
    # Possible max norm squared: 2, 4, 8
    best_solution = None
    best_count = 0
    
    for max_norm_sq in [2, 4, 8]:
        # Filter candidates to those with norm_sq <= max_norm_sq
        norms_sq = np.sum(candidates_arr ** 2, axis=1)
        mask = norms_sq <= max_norm_sq
        if not np.any(mask):
            continue
        filtered_idx = np.where(mask)[0]
        filtered = candidates_arr[filtered_idx]
        filtered_weights = [weights[i] for i in filtered_idx]
        n_f = len(filtered)
        
        # Build conflict matrix: conflict if |u-v|^2 < max_norm_sq
        # |u-v|^2 = |u|^2 + |v|^2 - 2*u.v
        dots = filtered @ filtered.T
        norms_f_sq = np.sum(filtered ** 2, axis=1)
        dist_sq = norms_f_sq[:, None] + norms_f_sq[None, :] - 2 * dots
        
        # Conflict if dist_sq < max_norm_sq (and u != v)
        conflict = dist_sq < max_norm_sq
        np.fill_diagonal(conflict, False)
        
        # Solve MWIS on this graph using branch and bound
        # For small graphs, we can use a simple approach
        # For larger, use greedy coloring bound
        
        # Convert to adjacency matrix for branch and bound
        adj = conflict.astype(np.int8)
        
        # Use a simple branch and bound for MWIS
        # Greedy coloring for upper bound
        def greedy_color_bound(adj, remaining):
            """Return upper bound on independent set size using greedy coloring"""
            n_rem = len(remaining)
            if n_rem == 0:
                return 0
            # Simple greedy coloring
            colors = {}
            color_count = 0
            for i in remaining:
                used_colors = set()
                for j in remaining:
                    if j != i and adj[i, j] and j in colors:
                        used_colors.add(colors[j])
                # Assign smallest available color
                c = 0
                while c in used_colors:
                    c += 1
                colors[i] = c
                color_count = max(color_count, c + 1)
            return color_count
        
        # Branch and bound
        best = []
        best_weight = 0
        
        def branch_and_bound(remaining, current, current_weight):
            nonlocal best, best_weight
            if not remaining:
                if current_weight > best_weight:
                    best_weight = current_weight
                    best = current[:]
                return
            
            # Upper bound
            ub = greedy_color_bound(adj, list(remaining)) + current_weight
            if ub <= best_weight:
                return
            
            # Choose vertex with highest degree in remaining subgraph
            rem_list = list(remaining)
            degs = [sum(adj[rem_list[i], rem_list[j]] for j in range(len(rem_list)) if i != j) for i in range(len(rem_list))]
            if degs:
                max_deg_idx = np.argmax(degs)
            else:
                max_deg_idx = 0
            v = rem_list[max_deg_idx]
            
            # Branch 1: include v
            new_remaining = [u for u in rem_list if u != v and not adj[v, u]]
            branch_and_bound(new_remaining, current + [v], current_weight + 1)
            
            # Branch 2: exclude v
            new_remaining = [u for u in rem_list if u != v]
            branch_and_bound(new_remaining, current, current_weight)
        
        # Only run B&B if graph is small enough
        if n_f <= 50:
            branch_and_bound(list(range(n_f)), [], 0)
            if best_weight > best_count:
                best_count = best_weight
                best_solution = [candidates_arr[i] for i in best]
        else:
            # For larger graphs, use greedy approach
            # Greedy: repeatedly pick vertex with minimum degree
            selected = []
            remaining = set(range(n_f))
            while remaining:
                # Find vertex with min degree in remaining subgraph
                min_deg = float('inf')
                min_v = None
                for v in remaining:
                    deg = sum(1 for u in remaining if u != v and adj[v, u])
                    if deg < min_deg:
                        min_deg = deg
                        min_v = v
                if min_v is None:
                    break
                selected.append(min_v)
                remaining.discard(min_v)
                # Remove all neighbors of min_v
                to_remove = [u for u in remaining if adj[min_v, u]]
                for u in to_remove:
                    remaining.discard(u)
            
            if len(selected) > best_count:
                best_count = len(selected)
                best_solution = [candidates_arr[i] for i in selected]
    
    if best_solution is None:
        # Fallback: return all norm-sq-2 vectors
        vectors = []
        for i in range(D):
            for j in range(i+1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        vectors.append(v)
        return np.array(vectors)
    
    return np.array(best_solution)
