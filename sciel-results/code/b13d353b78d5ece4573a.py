import time
import random
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
from collections import defaultdict

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = P['n']
    edges = P['edges']
    
    adj = [[False] * n for _ in range(n)]
    edge_set = set()
    edge_indices = {}
    for idx, (u, v) in enumerate(edges):
        adj[u][v] = True
        adj[v][u] = True
        key = (min(u, v), max(u, v))
        edge_set.add(key)
        edge_indices[key] = idx
        
    # 1. Generate a diverse pool of candidate bicliques
    # We use a randomized expansion from each edge to find maximal bicliques.
    candidates = []
    # To ensure we have a good pool, we run multiple random seeds for expansion
    for u, v in edges:
        for k in range(5):
            L = {u}
            R = {v}
            rng = random.Random(k)
            changed = True
            while changed:
                changed = False
                # Add to L
                cand_L = [x for x in range(n) if x not in L and x not in R and all(adj[x][y] for y in R)]
                if cand_L:
                    x = rng.choice(cand_L)
                    L.add(x)
                    changed = True
                # Add to R
                cand_R = [y for y in range(n) if y not in R and y not in L and all(adj[x][y] for x in L)]
                if cand_R:
                    y = rng.choice(cand_R)
                    R.add(y)
                    changed = True
            
            # Compute edge set for this biclique
            b_edges = set()
            valid = True
            for x in L:
                for y in R:
                    key = (min(x, y), max(x, y))
                    if key not in edge_set:
                        valid = False
                        break
                    b_edges.add(key)
            if valid:
                candidates.append((tuple(sorted(L)), tuple(sorted(R)), b_edges))
                
    # Deduplicate candidates based on (L, R)
    unique_candidates = []
    seen = set()
    for L, R, e_set in candidates:
        key = (L, R)
        if key not in seen:
            seen.add(key)
            unique_candidates.append((L, R, e_set))
            
    # Also add all single edges as candidates (size 1 bicliques) to ensure completeness
    for u, v in edges:
        L = (min(u, v),)
        R = (max(u, v),)
        key = (L, R)
        if key not in seen:
            seen.add(key)
            e_set = {(min(u, v), max(u, v))}
            unique_candidates.append((L, R, e_set))
            
    # Sort candidates by size descending to prioritize large ones in greedy if needed
    unique_candidates.sort(key=lambda x: len(x[2]), reverse=True)
    
    num_candidates = len(unique_candidates)
    num_edges = len(edges)
    
    if num_candidates == 0:
        return []
        
    # 2. Formulate and solve Linear Programming Relaxation
    # Minimize sum(x_i) subject to sum_{j: e in C_j} x_j = 1 for all e, 0 <= x_i <= 1
    # c = [1, 1, ..., 1]
    c = [1] * num_candidates
    
    # A_eq: num_edges x num_candidates
    # A_eq[e][c] = 1 if candidate c covers edge e
    A_eq = lil_matrix((num_edges, num_candidates), dtype=int)
    for c_idx, (L, R, e_set) in enumerate(unique_candidates):
        for e in e_set:
            e_idx = edge_indices[e]
            A_eq[e_idx, c_idx] = 1
    A_eq = A_eq.tocsr()
    b_eq = [1] * num_edges
    
    bounds = [(0, 1)] * num_candidates
    
    try:
        if time.time() > deadline - 10:
            raise TimeoutError
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if not res.success:
            raise Exception("LP failed")
        x_opt = res.x
    except Exception:
        x_opt = None
    
    # 3. Greedy Selection guided by LP solution
    # We will select candidates one by one.
    # Priority: high x_opt value, high coverage of remaining edges.
    
    def greedy_select(x_hint):
        uncovered = set(edge_set)
        selected = []
        used_candidates = set()
        
        while uncovered:
            if time.time() > deadline - 5:
                return None
            
            best_c_idx = -1
            best_score = -1
            best_covered = 0
            best_edges = None
            
            for c_idx, (L, R, e_set) in enumerate(unique_candidates):
                if c_idx in used_candidates:
                    continue
                
                # Count how many edges in e_set are in uncovered
                covered = e_set & uncovered
                count = len(covered)
                if count == 0:
                    continue
                
                # Score: prioritize LP value, then coverage
                lp_val = x_hint[c_idx] if x_hint else 0
                # We want to pick high LP value. 
                # If LP values are close, pick larger coverage.
                # Let's use a combined score.
                # To avoid floating point issues, we can sort by (lp_val, count)
                # But we need a single score for comparison.
                # Let's just iterate and keep the one with max (lp_val, count) lexicographically?
                # Or max lp_val, tie-break by count.
                
                if best_c_idx == -1:
                    best_c_idx = c_idx
                    best_score = (lp_val, count)
                    best_covered = count
                    best_edges = covered
                else:
                    current_score = (lp_val, count)
                    # Compare tuples
                    if current_score > best_score:
                        best_c_idx = c_idx
                        best_score = current_score
                        best_covered = count
                        best_edges = covered
            
            if best_c_idx == -1:
                return None # Should not happen if single edges are in pool
                
            L, R, _ = unique_candidates[best_c_idx]
            selected.append((L, R))
            used_candidates.add(best_c_idx)
            uncovered -= best_edges
            
        return selected

    best_partition = None
    
    # Try multiple random restarts of the greedy selection, perturbing the LP solution
    # or just using it directly.
    
    if x_opt is not None:
        # Use LP solution directly
        partition = greedy_select(x_opt)
        if partition:
            best_partition = partition
    
    # If LP failed or partition is bad, try random greedy
    # Also, we can try a "randomized LP" approach where we add noise to x_opt
    
    # Let's also try a pure greedy (max coverage) as a baseline
    # This is equivalent to x_hint = 0 for all, so it picks max coverage.
    partition_greedy = greedy_select(None)
    if partition_greedy:
        if best_partition is None or len(partition_greedy) < len(best_partition):
            best_partition = partition_greedy
            
    # Local Search: Merge
    # Try to merge bicliques in the best_partition found so far.
    # We can do this on the best_partition.
    
    if best_partition:
        current_bicliques = [[list(L), list(R)] for L, R in best_partition]
        
        def can_merge(L1, R1, L2, R2):
            for u in L1:
                for v in R2:
                    if not adj[u][v]:
                        return False
            for u in L2:
                for v in R1:
                    if not adj[u][v]:
                        return False
            return True

        def merge(L1, R1, L2, R2):
            new_L = list(set(L1) | set(L2))
            new_R = list(set(R1) | set(R2))
            return new_L, new_R

        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            indices = list(range(len(current_bicliques)))
            random.shuffle(indices)
            
            for i_idx in range(len(indices)):
                i = indices[i_idx]
                if i >= len(current_bicliques): continue
                for j_idx in range(i_idx + 1, len(indices)):
                    j = indices[j_idx]
                    if j >= len(current_bicliques): continue
                    
                    L1, R1 = current_bicliques[i]
                    L2, R2 = current_bicliques[j]
                    
                    if can_merge(L1, R1, L2, R2):
                        new_L, new_R = merge(L1, R1, L2, R2)
                        current_bicliques.pop(j)
                        current_bicliques[i] = [new_L, new_R]
                        improved = True
                        break
                if improved:
                    break
                    
        best_partition = [(tuple(L), tuple(R)) for L, R in current_bicliques]
        
    # Final validation and type conversion
    if not best_partition:
        # Fallback: simple greedy partition
        rem = set(edge_set)
        best_partition = []
        while rem:
            u, v = next(iter(rem))
            L = {u}
            R = {v}
            changed = True
            while changed:
                changed = False
                for x in range(n):
                    if x in L: continue
                    if x in R: continue
                    if all((min(x, y), max(x, y)) in rem for y in R):
                        L.add(x)
                        changed = True
                for y in range(n):
                    if y in R: continue
                    if y in L: continue
                    if all((min(x, y), max(x, y)) in rem for x in L):
                        R.add(y)
                        changed = True
            best_partition.append((tuple(L), tuple(R)))
            rem -= set((min(x, y), max(x, y)) for x in L for y in R)
            
    covered_edges = set()
    for L, R in best_partition:
        for u in L:
            for v in R:
                e = (min(u, v), max(u, v))
                if e in covered_edges:
                    return [] # Overlap found
                covered_edges.add(e)
                if not adj[u][v]:
                    return [] # Invalid edge
    
    if covered_edges != edge_set:
        return [] # Missing edges
        
    final_result = []
    for L, R in best_partition:
        final_L = [int(x) for x in L]
        final_R = [int(x) for x in R]
        final_result.append([final_L, final_R])
        
    return final_result

if __name__ == "__main__":
    res = construct()
    print(f"Found {len(res)} bicliques")

