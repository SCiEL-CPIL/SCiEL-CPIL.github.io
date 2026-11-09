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
        """Initialize using spectral optimization: maximize edges subject to K2,3-free constraint.
        Use continuous relaxation with gradient ascent, then round."""
        # Start with a random symmetric matrix
        A = np.zeros((n, n), dtype=np.float64)
        for u in range(n):
            for v in range(u + 1, n):
                A[u, v] = seed_rng.random() * 0.3
                A[v, u] = A[u, v]
        
        # Gradient ascent: maximize sum of A entries (proxy for edge count)
        # subject to K2,3-free constraint
        # K2,3-free means: for any pair (u,v), |N(u) ∩ N(v)| <= 2
        # In continuous relaxation: for any (u,v), sum over w of A[u,w]*A[v,w] <= 2
        
        lr = 0.05
        for step in range(200):
            if time.time() > deadline - 100.0:
                break
            
            # Compute gradient of objective (maximize sum of A)
            grad = np.ones_like(A)
            
            # Compute constraint violation: for each pair (u,v), C[u,v] = sum_w A[u,w]*A[v,w]
            # This is A @ A^T = A^2 for symmetric A
            C = A @ A
            np.fill_diagonal(C, 0)
            
            # For pairs where C[u,v] > 2, we need to reduce A[u,w]*A[v,w] for some w
            # Gradient of constraint: dC[u,v]/dA[x,y] = A[x,v]*delta_{y,u} + A[u,y]*delta_{x,v} + ...
            # Simplified: push back on edges that contribute to violations
            
            # Add penalty gradient for violated constraints
            penalty = np.zeros_like(A)
            for u in range(n):
                for v in range(u + 1, n):
                    if C[u, v] > 2.0:
                        # Reduce A[u,w] and A[v,w] for w that contribute
                        for w in range(n):
                            if w != u and w != v:
                                contrib = A[u, w] * A[v, w]
                                if contrib > 0:
                                    penalty[u, w] -= 1.0 * A[v, w]
                                    penalty[v, w] -= 1.0 * A[u, w]
            
            # Update
            A += lr * (grad + 5.0 * penalty)
            
            # Keep A symmetric and in [0, 1]
            A = (A + A.T) / 2.0
            A = np.clip(A, 0, 1)
            np.fill_diagonal(A, 0)
        
        # Round to binary: threshold at 0.5
        adj = (A > 0.5).astype(np.uint8)
        np.fill_diagonal(adj, 0)
        
        # Remove any K2,3 violations by deleting edges
        common = np.zeros((n, n), dtype=np.int16)
        for u in range(n):
            for v in range(n):
                if u != v:
                    common[u, v] = np.sum(adj[u] & adj[v])
        
        # Iteratively remove edges that cause violations
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if common[u, v] > 2:
                        # Find an edge to remove: pick (u, w) where w is common neighbor
                        common_neighbors = np.where((adj[u] == 1) & (adj[v] == 1))[0]
                        if len(common_neighbors) > 0:
                            w = common_neighbors[0]
                            # Remove edge (u, w)
                            adj[u, w] = 0
                            adj[w, u] = 0
                            for x in range(n):
                                if x != u and x != w:
                                    if adj[x, u] and adj[x, w]:
                                        common[x, u] -= 1
                                        common[u, x] -= 1
                                        common[x, w] -= 1
                                        common[w, x] -= 1
                            changed = True
        
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
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
        """Local search with edge additions, removals, and 1-for-2 swaps."""
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
            
            # Phase 2: Try 1-for-2 swaps (remove one edge, add two)
            if not improved and current_edge_count > 15:
                edge_list = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u, v]]
                rng.shuffle(edge_list)
                
                for rem_u, rem_v in edge_list[:50]:  # Limit swap attempts
                    # Save state
                    adj_save = adj.copy()
                    common_save = common.copy()
                    count_save = current_edge_count
                    
                    # Remove edge (rem_u, rem_v)
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
                    
                    # Try to add two edges from the freed capacity
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
                    
                    # Accept if we added 2 edges (net +1) or 1 edge (net 0, but might help future)
                    if added >= 1:
                        current_edge_count = count_save - 1 + added
                        improved = True
                        break
                    else:
                        # Restore
                        adj[:] = adj_save
                        common[:] = common_save
                        current_edge_count = count_save
        
        result = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    result.append([u, v])
        return result

    def final_polish(edges):
        """High-precision polish: try all possible single-edge additions, 2-edge swaps, and 2-for-3 swaps."""
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
            
            # Try all 1-for-2 swaps
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
                # Try pairs of edges to remove
                for i in range(min(len(edge_list), 30)):
                    for j in range(i + 1, min(len(edge_list), 30)):
                        if time.time() > deadline - 5.0:
                            break
                        
                        rem1_u, rem1_v = edge_list[i]
                        rem2_u, rem2_v = edge_list[j]
                        
                        adj_save = adj.copy()
                        common_save = common.copy()
                        
                        # Remove first edge
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
                        
                        # Remove second edge
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
                        
                        # Try to add 3 edges
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

    # Run multiple greedy builds with different seeds
    for trial in range(50):
        if time.time() > deadline - 20.0:
            break
        
        trial_rng = random.Random(trial * 12345 + 6789)
        
        # Alternate between spectral init and greedy build
        if trial % 3 == 0:
            edges = spectral_init(trial_rng)
        else:
            edges = numpy_greedy_build(trial_rng)
        
        edges = verify(edges)
        
        if edges is not None:
            # Local search with swaps
            improved = numpy_local_search_with_swaps(edges, max_iters=200)
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
