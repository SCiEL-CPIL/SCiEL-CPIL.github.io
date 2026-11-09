import random
import time
import sys
import numpy as np
from numpy.linalg import eigvalsh

def construct():
    n = 20
    start = time.time()
    deadline = start + 115.0  # safety margin under 120s limit
    rng = random.Random(42)

    def verify(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            if u == v or not (0 <= u < n and 0 <= v < n):
                return None
            if v in adj[u]:
                return None  # duplicate
            adj[u].add(v)
            adj[v].add(u)
        for u in range(n):
            for v in range(u + 1, n):
                if len(adj[u] & adj[v]) > 2:
                    return None
        return edges

    # Fallback: 5 disjoint K4s (10 edges)
    def fallback():
        e = []
        for k in range(5):
            base = 4 * k
            for a in range(4):
                for b in range(a + 1, 4):
                    e.append([base + a, base + b])
        return e

    best = None
    best_e = 0

    def spectral_init(seed_rng):
        """Spectral optimization with smooth constraint handling.
        Maximize sum(A) subject to (A@A)[u,v] <= 2 for all u != v.
        Uses log-sum-exp style smooth penalty for constraint violations."""
        A = np.zeros((n, n), dtype=np.float64)
        for u in range(n):
            for v in range(u + 1, n):
                A[u, v] = seed_rng.random() * 0.5
                A[v, u] = A[u, v]
        
        lr = 0.02
        penalty_weight = 10.0
        
        for step in range(300):
            if time.time() > deadline - 80.0:
                break
            
            # Objective: maximize sum of off-diagonal A entries
            # Constraint: for each (u,v), C[u,v] = sum_w A[u,w]*A[v,w] <= 2
            # Use smooth penalty: sum over violated pairs of max(0, C[u,v]-2)^2
            
            C = A @ A
            np.fill_diagonal(C, 0)
            
            # Gradient of objective
            grad = np.ones_like(A) * 0.5
            np.fill_diagonal(grad, 0)
            
            # Gradient of smooth penalty
            penalty_grad = np.zeros_like(A)
            for u in range(n):
                for v in range(u + 1, n):
                    violation = C[u, v] - 2.0
                    if violation > 0:
                        # d/dA[x,y] of violation^2 = 2*violation * d(violation)/dA[x,y]
                        # violation = sum_w A[u,w]*A[v,w]
                        # d(violation)/dA[u,w] = A[v,w] for w != u,v
                        # d(violation)/dA[v,w] = A[u,w] for w != u,v
                        factor = 2.0 * violation * penalty_weight
                        for w in range(n):
                            if w != u and w != v:
                                penalty_grad[u, w] -= factor * A[v, w]
                                penalty_grad[v, w] -= factor * A[u, w]
            
            # Update with adaptive learning rate
            A += lr * (grad + penalty_grad)
            
            # Keep A symmetric and in [0, 1]
            A = (A + A.T) / 2.0
            A = np.clip(A, 0, 1)
            np.fill_diagonal(A, 0)
            
            # Decay learning rate
            lr *= 0.995
        
        # Round to binary with threshold tuning
        best_adj = None
        best_count = 0
        for threshold in [0.4, 0.45, 0.5, 0.55, 0.6]:
            adj = (A > threshold).astype(np.uint8)
            np.fill_diagonal(adj, 0)
            
            # Remove K2,3 violations
            common = np.zeros((n, n), dtype=np.int16)
            for u in range(n):
                for v in range(n):
                    if u != v:
                        common[u, v] = np.sum(adj[u] & adj[v])
            
            changed = True
            while changed:
                changed = False
                for u in range(n):
                    for v in range(u + 1, n):
                        if common[u, v] > 2:
                            common_neighbors = np.where((adj[u] == 1) & (adj[v] == 1))[0]
                            if len(common_neighbors) > 0:
                                # Pick the edge whose removal fixes the most violations
                                best_w = common_neighbors[0]
                                best_fixes = 0
                                for w in common_neighbors:
                                    fixes = 0
                                    for x in range(n):
                                        if x != u and x != w and adj[x, u] and adj[x, w]:
                                            fixes += 1
                                        if x != v and x != w and adj[x, v] and adj[x, w]:
                                            fixes += 1
                                    if fixes > best_fixes:
                                        best_fixes = fixes
                                        best_w = w
                                w = best_w
                                adj[u, w] = 0
                                adj[w, u] = 0
                                for x in range(n):
                                    if x != u and x != w:
                                        if adj[x, u] and adj[x, w]:
                                            common[x, u] -= 1
                                            common[u, x] -= 1
                                        if adj[x, v] and adj[x, w]:
                                            common[x, v] -= 1
                                            common[v, x] -= 1
                                changed = True
            
            count = int(np.sum(adj)) // 2
            if count > best_count:
                best_count = count
                best_adj = adj.copy()
        
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if best_adj[u, v]:
                    edges.append([u, v])
        return edges

    def numpy_greedy_build(seed_rng):
        """Build a K2,3-free graph using vectorized NumPy common-neighbor checks."""
        adj = np.zeros((n, n), dtype=np.uint8)
        common = np.zeros((n, n), dtype=np.int16)
        
        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        seed_rng.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if adj[u, v]:
                continue
            
            neighbors_u = np.where(adj[u] == 1)[0]
            neighbors_v = np.where(adj[v] == 1)[0]
            
            if len(neighbors_u) > 0:
                if np.max(common[neighbors_u, v]) > 1:
                    continue
            
            if len(neighbors_v) > 0:
                if np.max(common[u, neighbors_v]) > 1:
                    continue
            
            adj[u, v] = 1
            adj[v, u] = 1
            
            for w in neighbors_u:
                if w != v:
                    common[w, v] += 1
                    common[v, w] += 1
            
            for w in neighbors_v:
                if w != u:
                    common[u, w] += 1
                    common[w, u] += 1
        
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    edges.append([u, v])
        return edges

    def numpy_local_search_with_swaps(edges, max_iters=300):
        """Local search with edge additions, 1-for-2 swaps, and 1-for-1 swaps."""
        adj = np.zeros((n, n), dtype=np.uint8)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1
        
        common = np.zeros((n, n), dtype=np.int16)
        for u in range(n):
            for v in range(n):
                if u != v:
                    common[u, v] = np.sum(adj[u] & adj[v])
        
        current_edge_count = len(edges)
        improved = True
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline - 10.0:
            improved = False
            iters += 1
            
            # Phase 1: Try adding any valid non-edge
            non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if not adj[u, v]]
            rng.shuffle(non_edges)
            
            for u, v in non_edges:
                neighbors_u = np.where(adj[u] == 1)[0]
                neighbors_v = np.where(adj[v] == 1)[0]
                
                valid = True
                if len(neighbors_u) > 0:
                    if np.max(common[neighbors_u, v]) > 1:
                        valid = False
                if valid and len(neighbors_v) > 0:
                    if np.max(common[u, neighbors_v]) > 1:
                        valid = False
                
                if valid:
                    adj[u, v] = 1
                    adj[v, u] = 1
                    for w in neighbors_u:
                        if w != v:
                            common[w, v] += 1
                            common[v, w] += 1
                    for w in neighbors_v:
                        if w != u:
                            common[u, w] += 1
                            common[w, u] += 1
                    current_edge_count += 1
                    improved = True
                    break
            
            # Phase 2: Try 1-for-2 swaps (remove one edge, add two) - must be net +1
            if not improved and current_edge_count > 15:
                edge_list = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u, v]]
                rng.shuffle(edge_list)
                
                for rem_u, rem_v in edge_list[:50]:
                    adj_save = adj.copy()
                    common_save = common.copy()
                    count_save = current_edge_count
                    
                    neighbors_ru = np.where(adj[rem_u] == 1)[0]
                    neighbors_rv = np.where(adj[rem_v] == 1)[0]
                    
                    adj[rem_u, rem_v] = 0
                    adj[rem_v, rem_u] = 0
                    
                    for w in neighbors_ru:
                        if w != rem_v:
                            common[w, rem_v] -= 1
                            common[rem_v, w] -= 1
                    for w in neighbors_rv:
                        if w != rem_u:
                            common[rem_u, w] -= 1
                            common[w, rem_u] -= 1
                    
                    # Try to add two edges
                    added = 0
                    for u, v in non_edges:
                        if adj[u, v]:
                            continue
                        if u == rem_u and v == rem_v:
                            continue
                        
                        neighbors_u = np.where(adj[u] == 1)[0]
                        neighbors_v = np.where(adj[v] == 1)[0]
                        
                        valid = True
                        if len(neighbors_u) > 0:
                            if np.max(common[neighbors_u, v]) > 1:
                                valid = False
                        if valid and len(neighbors_v) > 0:
                            if np.max(common[u, neighbors_v]) > 1:
                                valid = False
                        
                        if valid:
                            adj[u, v] = 1
                            adj[v, u] = 1
                            for w in neighbors_u:
                                if w != v:
                                    common[w, v] += 1
                                    common[v, w] += 1
                            for w in neighbors_v:
                                if w != u:
                                    common[u, w] += 1
                                    common[w, u] += 1
                            added += 1
                            if added == 2:
                                break
                    
                    # Only accept if net gain (added 2, removed 1 => net +1)
                    if added == 2:
                        current_edge_count = count_save + 1
                        improved = True
                        break
                    else:
                        adj[:] = adj_save
                        common[:] = common_save
                        current_edge_count = count_save
            
            # Phase 3: Try 1-for-1 swaps that enable future additions (neutral moves)
            if not improved and current_edge_count > 10:
                edge_list = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u, v]]
                rng.shuffle(edge_list)
                
                for rem_u, rem_v in edge_list[:30]:
                    adj_save = adj.copy()
                    common_save = common.copy()
                    
                    neighbors_ru = np.where(adj[rem_u] == 1)[0]
                    neighbors_rv = np.where(adj[rem_v] == 1)[0]
                    
                    adj[rem_u, rem_v] = 0
                    adj[rem_v, rem_u] = 0
                    
                    for w in neighbors_ru:
                        if w != rem_v:
                            common[w, rem_v] -= 1
                            common[rem_v, w] -= 1
                    for w in neighbors_rv:
                        if w != rem_u:
                            common[rem_u, w] -= 1
                            common[w, rem_u] -= 1
                    
                    # Try to add one edge (net 0, but might unlock future additions)
                    added = 0
                    for u, v in non_edges:
                        if adj[u, v]:
                            continue
                        if u == rem_u and v == rem_v:
                            continue
                        
                        neighbors_u = np.where(adj[u] == 1)[0]
                        neighbors_v = np.where(adj[v] == 1)[0]
                        
                        valid = True
                        if len(neighbors_u) > 0:
                            if np.max(common[neighbors_u, v]) > 1:
                                valid = False
                        if valid and len(neighbors_v) > 0:
                            if np.max(common[u, neighbors_v]) > 1:
                                valid = False
                        
                        if valid:
                            adj[u, v] = 1
                            adj[v, u] = 1
                            for w in neighbors_u:
                                if w != v:
                                    common[w, v] += 1
                                    common[v, w] += 1
                            for w in neighbors_v:
                                if w != u:
                                    common[u, w] += 1
                                    common[w, u] += 1
                            added += 1
                            break
                    
                    if added == 1:
                        # Accept neutral move - hope it helps in next iteration
                        improved = True
                        break
                    else:
                        adj[:] = adj_save
                        common[:] = common_save
        
        result = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    result.append([u, v])
        return result

    def final_polish(edges):
        """High-precision polish with exhaustive search for small n."""
        adj = np.zeros((n, n), dtype=np.uint8)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1
        
        common = np.zeros((n, n), dtype=np.int16)
        for u in range(n):
            for v in range(n):
                if u != v:
                    common[u, v] = np.sum(adj[u] & adj[v])
        
        current_edge_count = len(edges)
        improved = True
        
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # Try all non-edges
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u, v]:
                        continue
                    
                    neighbors_u = np.where(adj[u] == 1)[0]
                    neighbors_v = np.where(adj[v] == 1)[0]
                    
                    valid = True
                    if len(neighbors_u) > 0:
                        if np.max(common[neighbors_u, v]) > 1:
                            valid = False
                    if valid and len(neighbors_v) > 0:
                        if np.max(common[u, neighbors_v]) > 1:
                            valid = False
                    
                    if valid:
                        adj[u, v] = 1
                        adj[v, u] = 1
                        for w in neighbors_u:
                            if w != v:
                                common[w, v] += 1
                                common[v, w] += 1
                        for w in neighbors_v:
                            if w != u:
                                common[u, w] += 1
                                common[w, u] += 1
                        current_edge_count += 1
                        improved = True
                        break
                if improved:
                    break
            
            # Try all 1-for-2 swaps (exhaustive)
            if not improved:
                for rem_u in range(n):
                    for rem_v in range(rem_u + 1, n):
                        if not adj[rem_u, rem_v]:
                            continue
                        
                        adj_save = adj.copy()
                        common_save = common.copy()
                        
                        neighbors_ru = np.where(adj[rem_u] == 1)[0]
                        neighbors_rv = np.where(adj[rem_v] == 1)[0]
                        
                        adj[rem_u, rem_v] = 0
                        adj[rem_v, rem_u] = 0
                        
                        for w in neighbors_ru:
                            if w != rem_v:
                                common[w, rem_v] -= 1
                                common[rem_v, w] -= 1
                        for w in neighbors_rv:
                            if w != rem_u:
                                common[rem_u, w] -= 1
                                common[w, rem_u] -= 1
                        
                        added = 0
                        for u in range(n):
                            for v in range(u + 1, n):
                                if adj[u, v]:
                                    continue
                                if u == rem_u and v == rem_v:
                                    continue
                                
                                neighbors_u = np.where(adj[u] == 1)[0]
                                neighbors_v = np.where(adj[v] == 1)[0]
                                
                                valid = True
                                if len(neighbors_u) > 0:
                                    if np.max(common[neighbors_u, v]) > 1:
                                        valid = False
                                if valid and len(neighbors_v) > 0:
                                    if np.max(common[u, neighbors_v]) > 1:
                                        valid = False
                                
                                if valid:
                                    adj[u, v] = 1
                                    adj[v, u] = 1
                                    for w in neighbors_u:
                                        if w != v:
                                            common[w, v] += 1
                                            common[v, w] += 1
                                    for w in neighbors_v:
                                        if w != u:
                                            common[u, w] += 1
                                            common[w, u] += 1
                                    added += 1
                                    if added == 2:
                                        break
                            if added == 2:
                                break
                        
                        if added == 2:
                            current_edge_count += 1
                            improved = True
                            break
                        else:
                            adj[:] = adj_save
                            common[:] = common_save
                    if improved:
                        break
            
            # Try 2-for-3 swaps: remove 2 edges, add 3 edges
            if not improved and current_edge_count > 20:
                edge_list = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u, v]]
                for i in range(min(len(edge_list), 40)):
                    for j in range(i + 1, min(len(edge_list), 40)):
                        if time.time() > deadline - 5.0:
                            break
                        
                        rem1_u, rem1_v = edge_list[i]
                        rem2_u, rem2_v = edge_list[j]
                        
                        adj_save = adj.copy()
                        common_save = common.copy()
                        
                        neighbors_r1u = np.where(adj[rem1_u] == 1)[0]
                        neighbors_r1v = np.where(adj[rem1_v] == 1)[0]
                        
                        adj[rem1_u, rem1_v] = 0
                        adj[rem1_v, rem1_u] = 0
                        
                        for w in neighbors_r1u:
                            if w != rem1_v:
                                common[w, rem1_v] -= 1
                                common[rem1_v, w] -= 1
                        for w in neighbors_r1v:
                            if w != rem1_u:
                                common[rem1_u, w] -= 1
                                common[w, rem1_u] -= 1
                        
                        neighbors_r2u = np.where(adj[rem2_u] == 1)[0]
                        neighbors_r2v = np.where(adj[rem2_v] == 1)[0]
                        
                        adj[rem2_u, rem2_v] = 0
                        adj[rem2_v, rem2_u] = 0
                        
                        for w in neighbors_r2u:
                            if w != rem2_v:
                                common[w, rem2_v] -= 1
                                common[rem2_v, w] -= 1
                        for w in neighbors_r2v:
                            if w != rem2_u:
                                common[rem2_u, w] -= 1
                                common[w, rem2_u] -= 1
                        
                        added = 0
                        for u in range(n):
                            for v in range(u + 1, n):
                                if adj[u, v]:
                                    continue
                                if (u == rem1_u and v == rem1_v) or (u == rem2_u and v == rem2_v):
                                    continue
                                
                                neighbors_u = np.where(adj[u] == 1)[0]
                                neighbors_v = np.where(adj[v] == 1)[0]
                                
                                valid = True
                                if len(neighbors_u) > 0:
                                    if np.max(common[neighbors_u, v]) > 1:
                                        valid = False
                                if valid and len(neighbors_v) > 0:
                                    if np.max(common[u, neighbors_v]) > 1:
                                        valid = False
                                
                                if valid:
                                    adj[u, v] = 1
                                    adj[v, u] = 1
                                    for w in neighbors_u:
                                        if w != v:
                                            common[w, v] += 1
                                            common[v, w] += 1
                                    for w in neighbors_v:
                                        if w != u:
                                            common[u, w] += 1
                                            common[w, u] += 1
                                    added += 1
                                    if added == 3:
                                        break
                            if added == 3:
                                break
                        
                        if added == 3:
                            current_edge_count += 1
                            improved = True
                            break
                        else:
                            adj[:] = adj_save
                            common[:] = common_save
                    if improved:
                        break
        
        result = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    result.append([u, v])
        return result

    def maxsat_vectorized_search(seed_rng, max_iters=500):
        """MaxSAT-style: start from complete graph, remove violating edges,
        then use vectorized local search to maximize edge count."""
        # Start with complete graph
        adj = np.ones((n, n), dtype=np.uint8)
        np.fill_diagonal(adj, 0)
        
        # Precompute common neighbor counts
        common = np.zeros((n, n), dtype=np.int16)
        for u in range(n):
            for v in range(n):
                if u != v:
                    common[u, v] = np.sum(adj[u] & adj[v])
        
        # Phase 1: Remove edges to make K2,3-free (greedy violation fixing)
        # Priority: remove edge (u,v) that appears in most K2,3 violations
        for step in range(5000):
            if time.time() > deadline - 30.0:
                break
            
            # Find all violating pairs
            violations = []
            for u in range(n):
                for v in range(u + 1, n):
                    if common[u, v] > 2:
                        # Count how many times each common neighbor contributes
                        # to violations involving u and v
                        common_neighbors = np.where((adj[u] == 1) & (adj[v] == 1))[0]
                        for w in common_neighbors:
                            # Edge (u,w) is in a violation if common[u,w] > 2 or common[v,w] > 2
                            # Actually, removing (u,w) helps if it reduces common[x,w] for some x
                            violations.append((u, w, common[u, v]))
                            violations.append((v, w, common[u, v]))
            
            if not violations:
                break
            
            # Count frequency of each edge in violations
            edge_freq = {}
            for u, v, _ in violations:
                key = (min(u, v), max(u, v))
                edge_freq[key] = edge_freq.get(key, 0) + 1
            
            # Remove the most frequently violating edge
            worst_edge = max(edge_freq.items(), key=lambda x: x[1])
            rem_u, rem_v = worst_edge[0]
            
            # Remove edge
            neighbors_u = np.where(adj[rem_u] == 1)[0]
            neighbors_v = np.where(adj[rem_v] == 1)[0]
            
            adj[rem_u, rem_v] = 0
            adj[rem_v, rem_u] = 0
            
            for w in neighbors_u:
                if w != rem_v:
                    common[w, rem_v] -= 1
                    common[rem_v, w] -= 1
            for w in neighbors_v:
                if w != rem_u:
                    common[rem_u, w] -= 1
                    common[w, rem_u] -= 1
        
        # Phase 2: Vectorized local search - try to add back edges
        # Use NumPy to vectorize the validity check
        current_edge_count = int(np.sum(adj)) // 2
        
        for iters in range(max_iters):
            if time.time() > deadline - 15.0:
                break
            
            improved = False
            
            # Generate all non-edges
            non_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if not adj[u, v]:
                        non_edges.append((u, v))
            
            if not non_edges:
                break
            
            # Shuffle for diversity
            seed_rng.shuffle(non_edges)
            
            # Vectorized validity check: for each non-edge (u,v),
            # it's valid if max(common[neighbors_u, v]) <= 1 AND max(common[u, neighbors_v]) <= 1
            for u, v in non_edges:
                neighbors_u = np.where(adj[u] == 1)[0]
                neighbors_v = np.where(adj[v] == 1)[0]
                
                valid = True
                if len(neighbors_u) > 0:
                    if np.max(common[neighbors_u, v]) > 1:
                        valid = False
                if valid and len(neighbors_v) > 0:
                    if np.max(common[u, neighbors_v]) > 1:
                        valid = False
                
                if valid:
                    adj[u, v] = 1
                    adj[v, u] = 1
                    for w in neighbors_u:
                        if w != v:
                            common[w, v] += 1
                            common[v, w] += 1
                    for w in neighbors_v:
                        if w != u:
                            common[u, w] += 1
                            common[w, u] += 1
                    current_edge_count += 1
                    improved = True
                    break
            
            if not improved:
                # Try 1-for-2 swaps
                edge_list = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u, v]]
                if len(edge_list) > 10:
                    seed_rng.shuffle(edge_list)
                    for rem_u, rem_v in edge_list[:30]:
                        adj_save = adj.copy()
                        common_save = common.copy()
                        
                        neighbors_ru = np.where(adj[rem_u] == 1)[0]
                        neighbors_rv = np.where(adj[rem_v] == 1)[0]
                        
                        adj[rem_u, rem_v] = 0
                        adj[rem_v, rem_u] = 0
                        
                        for w in neighbors_ru:
                            if w != rem_v:
                                common[w, rem_v] -= 1
                                common[rem_v, w] -= 1
                        for w in neighbors_rv:
                            if w != rem_u:
                                common[rem_u, w] -= 1
                                common[w, rem_u] -= 1
                        
                        # Try to add 2 edges
                        added = 0
                        for u, v in non_edges:
                            if adj[u, v]:
                                continue
                            if u == rem_u and v == rem_v:
                                continue
                            
                            neighbors_u = np.where(adj[u] == 1)[0]
                            neighbors_v = np.where(adj[v] == 1)[0]
                            
                            valid = True
                            if len(neighbors_u) > 0:
                                if np.max(common[neighbors_u, v]) > 1:
                                    valid = False
                            if valid and len(neighbors_v) > 0:
                                if np.max(common[u, neighbors_v]) > 1:
                                    valid = False
                            
                            if valid:
                                adj[u, v] = 1
                                adj[v, u] = 1
                                for w in neighbors_u:
                                    if w != v:
                                        common[w, v] += 1
                                        common[v, w] += 1
                                for w in neighbors_v:
                                    if w != u:
                                        common[u, w] += 1
                                        common[w, u] += 1
                                added += 1
                                if added == 2:
                                    break
                        
                        if added >= 1:
                            current_edge_count += added - 1
                            improved = True
                            break
                        else:
                            adj[:] = adj_save
                            common[:] = common_save
                
                if not improved:
                    break
        
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    edges.append([u, v])
        return edges

    # Run multiple approaches with different seeds
    for trial in range(80):
        if time.time() > deadline - 15.0:
            break
        
        trial_rng = random.Random(trial * 12345 + 6789)
        
        # Alternate between different strategies
        if trial % 5 == 0:
            # MaxSAT-style from complete graph
            edges = maxsat_vectorized_search(trial_rng, max_iters=250)
        elif trial % 5 == 1:
            # Spectral init (improved)
            edges = spectral_init(trial_rng)
        elif trial % 5 == 2:
            # Greedy build
            edges = numpy_greedy_build(trial_rng)
        elif trial % 5 == 3:
            # MaxSAT with more iterations
            edges = maxsat_vectorized_search(trial_rng, max_iters=400)
        else:
            # Spectral init with different seed
            edges = spectral_init(trial_rng)
        
        edges = verify(edges)
        
        if edges is not None:
            # Local search with swaps
            improved = numpy_local_search_with_swaps(edges, max_iters=150)
            improved = verify(improved)
            
            # Final polish
            if improved is not None:
                polished = final_polish(improved)
                polished = verify(polished)
                if polished is not None and len(polished) > best_e:
                    best_e = len(polished)
                    best = polished
            elif edges is not None and len(edges) > best_e:
                best_e = len(edges)
                best = edges
    
    # Final verification
    if best is not None:
        best = verify(best)
    
    if best is None:
        return fallback()
    return best

if __name__ == "__main__":
    result = construct()
    print(f"Edges: {len(result)}")
