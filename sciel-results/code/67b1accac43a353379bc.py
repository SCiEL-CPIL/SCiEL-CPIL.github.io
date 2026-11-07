import time
import random
import sys
import numpy as np

# Check for scipy.optimize.milp
try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    HAS_MILP = True
except ImportError:
    HAS_MILP = False

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj = [0] * n
    edge_set = set()
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        if u < v:
            edge_set.add((u, v))
        else:
            edge_set.add((v, u))
            
    # List of edges for ILP rows
    edge_list = list(edge_set)
    edge_index = {e: i for i, e in enumerate(edge_list)}
    m_edges = len(edge_list)

    # Function to expand a biclique to maximal
    def expand_biclique(L_mask, R_mask):
        changed = True
        while changed:
            changed = False
            # Expand L
            for u in range(n):
                if (L_mask >> u) & 1:
                    continue
                if (adj[u] & R_mask) == R_mask and R_mask != 0:
                    L_mask |= (1 << u)
                    changed = True
            # Expand R
            for v in range(n):
                if (R_mask >> v) & 1:
                    continue
                if (adj[v] & L_mask) == L_mask and L_mask != 0:
                    R_mask |= (1 << v)
                    changed = True
        return L_mask, R_mask

    # Enumerate all maximal bicliques
    # A biclique is identified by (frozenset(L), frozenset(R))
    # We store the edge set covered by each biclique for the ILP
    maximal_bicliques = []
    seen_bicliques = set()
    
    for u, v in edges:
        L_mask = 1 << u
        R_mask = 1 << v
        L_mask, R_mask = expand_biclique(L_mask, R_mask)
        
        # Canonical representation to avoid duplicates
        # Sort vertices in L and R
        L_list = [i for i in range(n) if (L_mask >> i) & 1]
        R_list = [i for i in range(n) if (R_mask >> i) & 1]
        
        # To handle symmetry (L,R) vs (R,L), we can just use the edge set as the key
        # Or ensure L_list < R_list lexicographically?
        # Actually, the edge set uniquely identifies the biclique in terms of coverage.
        # Two different pairs (L1,R1) and (L2,R2) cover the same edges iff L1=L2 and R1=R2 (or swapped).
        # Let's use the frozenset of edges as the key.
        biclique_edges = set()
        for u in L_list:
            for v in R_list:
                if u < v:
                    biclique_edges.add((u, v))
                else:
                    biclique_edges.add((v, u))
        
        key = frozenset(biclique_edges)
        if key not in seen_bicliques:
            seen_bicliques.add(key)
            # Store (L_list, R_list, edge_set)
            maximal_bicliques.append((L_list, R_list, biclique_edges))

    # If no bicliques found (should not happen if edges exist)
    if not maximal_bicliques:
        return []

    # Fallback Seed Solution: One biclique per vertex
    seed_bicliques = []
    for u in range(n):
        right_side = [v for u2, v in edges if u2 == u]
        if right_side:
            seed_bicliques.append(( [u], right_side ))
    
    best_bicliques = seed_bicliques
    best_count = len(seed_bicliques)

    # Try ILP
    if HAS_MILP and time.time() < deadline - 10.0:
        try:
            m_bics = len(maximal_bicliques)
            # A matrix: m_edges x m_bics
            A = np.zeros((m_edges, m_bics))
            for j, (L_list, R_list, b_edges) in enumerate(maximal_bicliques):
                for e in b_edges:
                    i = edge_index[e]
                    A[i, j] = 1
            
            # Objective: minimize sum(z)
            c = np.ones(m_bics)
            
            # Constraints: A @ z >= 1
            # milp expects constraints in form: lb <= A @ x <= ub
            # So we need to invert or use appropriate bounds.
            # scipy.optimize.milp handles LinearConstraint(A, lb, ub)
            # We want A @ z >= 1.
            # This is equivalent to -A @ z <= -1.
            # Or simply use LinearConstraint(A, 1, np.inf)
            
            constraints = [LinearConstraint(A, 1, np.inf)]
            integrality = np.ones(m_bics)
            bounds = Bounds(0, 1)
            
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
            
            if res.status == 0: # Optimal
                z = res.x
                selected_indices = [i for i in range(m_bics) if z[i] > 0.5]
                
                if selected_indices:
                    ilp_bicliques = []
                    for idx in selected_indices:
                        L_list, R_list, _ = maximal_bicliques[idx]
                        ilp_bicliques.append((L_list, R_list))
                    
                    # Verify coverage just in case
                    # (Not strictly necessary if ILP is correct, but good for safety)
                    # If valid and better, update best
                    if len(ilp_bicliques) < best_count:
                        best_count = len(ilp_bicliques)
                        best_bicliques = ilp_bicliques
                        if best_count == 1:
                            break # Can't do better than 1
                            
        except Exception:
            pass # Fall through to greedy

    # Greedy Set Cover on maximal bicliques
    # This is a strong fallback
    if time.time() < deadline - 5.0:
        # We want to find a subset of maximal_bicliques that covers all edges
        # Greedy: repeatedly pick the biclique covering the most uncovered edges
        
        # For efficiency, we can precompute the edge lists as sets
        # And maintain a set of uncovered edges
        
        uncovered_edges = set(edge_list)
        greedy_bicliques = []
        
        # Sort bicliques by size descending? Or just pick best each time.
        # Picking best each time is O(M * E) per step, total O(M^2 * E).
        # M is small (~100-500). E=145. This is fast.
        
        available_bics = list(range(len(maximal_bicliques)))
        
        while uncovered_edges:
            if time.time() > deadline - 2.0:
                break
                
            best_idx = -1
            best_count_covered = 0
            
            for idx in available_bics:
                b_edges = maximal_bicliques[idx][2]
                # Count how many edges in b_edges are in uncovered_edges
                covered = 0
                # Optimization: if b_edges is a subset of uncovered, covered = len(b_edges)
                # But we need intersection size.
                # len(b_edges & uncovered_edges)
                # Sets are small, so this is fine.
                inter = b_edges & uncovered_edges
                if len(inter) > best_count_covered:
                    best_count_covered = len(inter)
                    best_idx = idx
                    
            if best_idx == -1:
                break # No biclique covers any remaining edge? Should not happen if maximal bicliques cover all edges.
            
            # Add biclique
            L_list, R_list, b_edges = maximal_bicliques[best_idx]
            greedy_bicliques.append((L_list, R_list))
            
            # Remove covered edges
            uncovered_edges -= b_edges
            
            # Remove biclique from available (it can be used again? No, set cover implies distinct sets. 
            # But in biclique partition, can we use the same biclique twice? 
            # "The cross-edge sets must partition all pinned graph edges exactly once."
            # This implies we select a subset of bicliques. We don't select the same biclique twice.
            # So remove it.
            available_bics.remove(best_idx)
            
        if greedy_bicliques and len(greedy_bicliques) < best_count:
            best_count = len(greedy_bicliques)
            best_bicliques = greedy_bicliques

    # Return best found
    # Ensure types are correct: lists of ints
    result = []
    for L, R in best_bicliques:
        left = [int(x) for x in L]
        right = [int(x) for x in R]
        result.append([left, right])

    return result

