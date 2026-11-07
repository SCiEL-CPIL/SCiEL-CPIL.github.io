"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict
import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import milp, LinearConstraint, Bounds

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    n = P['n']
    edges = P['edges']

    # Build adjacency sets
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
    
    total_edges = len(edge_set)
    if total_edges == 0:
        return []

    # Map edges to indices for ILP constraints
    edge_list = list(edge_set)
    edge_to_idx = {e: i for i, e in enumerate(edge_list)}

    # --- Phase 1: Generate Candidate Bicliques ---
    
    def expand_biclique(u, v, adj, n):
        """Expands seed edge (u, v) to a maximal biclique."""
        L = {u}
        R = {v}
        
        # Greedy expansion
        # We try to add vertices to L or R if they are connected to all vertices in the opposite set.
        # To make it faster, we precompute common neighbors.
        
        changed = True
        while changed:
            changed = False
            # Try adding to L: w must be connected to all in R
            # Candidate L vertices: neighbors of all in R
            common_N_R = set(adj[x] for x in R)
            # Actually, intersection of neighborhoods
            common_N_R = set.intersection(*[set(adj[x]) for x in R]) if R else set(range(n))
            
            # Exclude vertices already in L or R
            cand_L = common_N_R - L - R
            if cand_L:
                # Pick a vertex with high degree to R? Or just any?
                # Let's pick the one with max degree in the remaining graph to encourage growth
                # Or just iterate deterministically for consistency?
                # Randomness helps in generation diversity.
                w = random.choice(list(cand_L))
                L.add(w)
                changed = True
                continue

            # Try adding to R: w must be connected to all in L
            common_N_L = set.intersection(*[set(adj[x]) for x in L]) if L else set(range(n))
            cand_R = common_N_L - L - R
            if cand_R:
                w = random.choice(list(cand_R))
                R.add(w)
                changed = True
                continue
                
        return L, R

    candidates = []
    max_candidates = 3000
    gen_time_limit = start_time + 20.0
    
    random.seed(42)
    
    # Ensure single edges are always candidates (feasibility guarantee)
    for e in edge_list:
        u, v = e
        candidates.append(({u}, {v}))
        
    # Generate larger bicliques
    attempts = 0
    while time.time() < gen_time_limit and len(candidates) < max_candidates:
        u, v = random.choice(edge_list)
        L, R = expand_biclique(u, v, adj, n)
        
        # Filter: Must be a valid biclique (it is by construction)
        # Check if it covers at least 2 edges to be interesting? 
        # No, keep all.
        
        # Normalize for deduplication: 
        # A biclique is defined by its edge set. 
        # Edge set is { (l, r) for l in L, r in R } normalized to min, max.
        # Two bicliques (L1, R1) and (L2, R2) are identical if their edge sets are identical.
        # This happens if {L1, R1} == {L2, R2} (as sets of sets) or if L1=L2, R1=R2 or L1=R2, R1=L2?
        # If L1=L2 and R1=R2, they are the same.
        # If L1=R2 and R1=L2, the edge set is the same (symmetric).
        # So we should canonicalize (L, R) such that sum(L) <= sum(R) or similar.
        
        key_L = frozenset(L)
        key_R = frozenset(R)
        if sum(key_L) > sum(key_R):
             key_L, key_R = key_R, key_L
             
        if (key_L, key_R) not in [c[2] for c in candidates]: # Simple check, slow
             # Better to use a set
             pass

    # Redo deduplication properly
    unique_candidates = []
    seen_keys = set()
    for L, R in candidates:
        # Canonicalize
        if sum(L) > sum(R):
            L, R = R, L
        key = (frozenset(L), frozenset(R))
        if key not in seen_keys:
            seen_keys.add(key)
            unique_candidates.append((L, R))
            
    candidates = unique_candidates
    
    # --- Phase 2: Formulate and Solve ILP ---
    
    if not candidates:
        return []

    num_candidates = len(candidates)
    
    # Objective: Minimize sum(x_i)
    c_obj = np.ones(num_candidates)
    
    # Constraints: For each edge e_j, sum(x_i for i where e_j in biclique_i) == 1
    # A is a sparse matrix of size (total_edges, num_candidates)
    A = lil_matrix((total_edges, num_candidates), dtype=np.int8)
    
    for i, (L, R) in enumerate(candidates):
        for u in L:
            for v in R:
                if u < v:
                    e = (u, v)
                else:
                    e = (v, u)
                if e in edge_to_idx:
                    j = edge_to_idx[e]
                    A[j, i] = 1
                else:
                    # This shouldn't happen if candidates are generated from graph edges
                    pass
    
    # Convert to CSR for efficiency
    A_csr = A.tocsr()
    
    constraints = LinearConstraint(A_csr, 1, 1)
    bounds = Bounds(0, 1)
    
    # Solve
    sol = milp(
        c=c_obj,
        constraints=constraints,
        integrality=np.ones(num_candidates),
        bounds=bounds,
        options={"time_limit": 40.0, "mip_rel_gap": 0.0}
    )
    
    best_bicliques = []
    
    if sol.success or (sol.status == 4 and sol.x is not None): # 4 is "best solution found"
        # sol.x contains the decision variables
        # Thresholding: if x_i > 0.5, it's selected
        for i, x_i in enumerate(sol.x):
            if x_i > 0.5:
                L, R = candidates[i]
                best_bicliques.append((list(L), list(R)))
    
    # --- Phase 3: Validation and Fallback ---
    
    def is_valid_partition(bicliques):
        covered = set()
        for left, right in bicliques:
            if not left or not right:
                return False
            if len(set(left)) != len(left) or len(set(right)) != len(right):
                return False
            if set(left) & set(right):
                return False
            for u in left:
                for v in right:
                    if u < v:
                        pair = (u, v)
                    else:
                        pair = (v, u)
                    if pair not in edge_set:
                        return False
                    if pair in covered:
                        return False
                    covered.add(pair)
        return covered == edge_set

    if not best_bicliques or not is_valid_partition(best_bicliques):
        # Fallback: Greedy
        def greedy_partition():
            remaining = set(edge_set)
            bicliques = []
            while remaining:
                # Pick random edge
                u, v = random.choice(list(remaining))
                L, R = expand_biclique(u, v, adj, n)
                # Ensure L and R are within the remaining edges?
                # The expansion uses the global graph adj. 
                # We must filter L and R such that ALL cross edges are in 'remaining'.
                new_L = set()
                for u in L:
                    if all((min(u, v), max(u, v)) in remaining for v in R):
                        new_L.add(u)
                new_R = set()
                for v in R:
                    if all((min(u, v), max(u, v)) in remaining for u in new_L):
                        new_R.add(v)
                        
                if not new_L or not new_R:
                    # Fallback to single edge
                    new_L = {u}
                    new_R = {v}
                    if (min(u,v), max(u,v)) not in remaining:
                         # Should not happen
                         continue
                 
                bicliques.append((list(new_L), list(new_R)))
                
                # Remove covered edges
                for u in new_L:
                    for v in new_R:
                        e = (min(u, v), max(u, v))
                        remaining.discard(e)
                        
            return bicliques

        best_bicliques = greedy_partition()
        
        if not is_valid_partition(best_bicliques):
             # Last resort: single edges
             best_bicliques = [([u], [v]) for u, v in edge_set]

    # Final cleanup and formatting
    result = []
    for left, right in best_bicliques:
        left_clean = [int(x) for x in left]
        right_clean = [int(x) for x in right]
        result.append([left_clean, right_clean])
        
    return result

# EVOLVE-BLOCK-END
