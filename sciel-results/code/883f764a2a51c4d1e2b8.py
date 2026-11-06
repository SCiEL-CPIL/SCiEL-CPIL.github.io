import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def time_left():
        return deadline - time.time()
    
    def popcount(x):
        return bin(x).count('1')
    
    def is_c4_free(n, adj):
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    common = adj[i] & adj[j]
                    if popcount(common) > 1:
                        return False
        return True
    
    def max_independent_set_size(n, adj, timeout=30):
        best = [0]
        start = time.time()
        
        def branch(candidates_mask, current_size):
            if time.time() - start > timeout:
                return
            if candidates_mask == 0:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            if current_size + popcount(candidates_mask) <= best[0]:
                return
            
            # Pick vertex with highest degree in candidate subgraph
            best_v = -1
            best_deg = -1
            m = candidates_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                deg = popcount(candidates_mask & adj[v])
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                m -= lsb
            
            if best_v == -1:
                return
            
            v_bit = 1 << best_v
            
            # Include best_v
            new_candidates = candidates_mask & ~adj[best_v] & ~v_bit
            branch(new_candidates, current_size + 1)
            
            # Exclude best_v
            branch(candidates_mask & ~v_bit, current_size)
        
        branch((1 << n) - 1, 0)
        return best[0]
    
    def try_add_edge(n, adj, i, j):
        """Check if adding edge (i,j) maintains C4-freeness."""
        # Check pair (i, j): |N(i) ∩ N(j)| <= 1
        common_ij = adj[i] & adj[j]
        if popcount(common_ij) > 1:
            return False
        
        # Check pairs (i, k) for k in N(j)
        mj = adj[j] & ~(1 << i)
        while mj:
            lsb = mj & -mj
            k = lsb.bit_length() - 1
            # If k is in N(i), then k is common neighbor of i and j
            # We already checked that. Now check if adding (i,j) creates C4 with k
            # N(i) gains j, N(k) unchanged
            # Common neighbors of i and k after adding: old + (j if j in N(k))
            if adj[i] & (1 << k):
                # k is in N(i), so k is common neighbor of i and j
                # But we need to check if this creates a C4
                # Actually, the condition is just |N(i) ∩ N(k)| <= 1
                # After adding (i,j), N(i) = old_N(i) ∪ {j}
                # N(k) = old_N(k)
                # Common = (old_N(i) ∪ {j}) ∩ old_N(k) = (old_N(i) ∩ old_N(k)) ∪ ({j} ∩ old_N(k))
                # So we need |old_N(i) ∩ old_N(k)| + (1 if j in old_N(k) else 0) <= 1
                common_ik = adj[i] & adj[k]
                count = popcount(common_ik)
                if adj[j] & (1 << k):
                    count += 1
                if count > 1:
                    return False
            mj -= lsb
        
        # Check pairs (j, k) for k in N(i)
        mi = adj[i] & ~(1 << j)
        while mi:
            lsb = mi & -mi
            k = lsb.bit_length() - 1
            if adj[j] & (1 << k):
                common_jk = adj[j] & adj[k]
                count = popcount(common_jk)
                if adj[i] & (1 << k):
                    count += 1
                if count > 1:
                    return False
            mi -= lsb
        
        return True
    
    def maximize_edges(n, adj):
        adj = adj[:]
        improved = True
        while improved and time_left() > 15:
            improved = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj[i] & (1 << j)):
                        if try_add_edge(n, adj, i, j):
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                            improved = True
        return adj
    
    def prune_c4s(n, adj):
        adj = adj[:]
        changed = True
        while changed and time_left() > 15:
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i] & (1 << j):
                        common = adj[i] & adj[j]
                        if popcount(common) > 1:
                            adj[i] &= ~(1 << j)
                            adj[j] &= ~(1 << i)
                            changed = True
                            break
                if changed:
                    break
        return adj
    
    def random_graph(n, p):
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj
    
    def local_search(n, adj, target_alpha=10):
        """Local search to reduce alpha while maintaining C4-freeness."""
        adj = adj[:]
        alpha = max_independent_set_size(n, adj, timeout=20)
        if alpha <= target_alpha:
            return adj
        
        # Try adding edges to reduce alpha
        improved = True
        while improved and time_left() > 15:
            improved = False
            # Try all possible edges
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj[i] & (1 << j)):
                        if try_add_edge(n, adj, i, j):
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                            new_alpha = max_independent_set_size(n, adj, timeout=10)
                            if new_alpha < alpha:
                                alpha = new_alpha
                                improved = True
                                if alpha <= target_alpha:
                                    return adj
                            else:
                                # Revert
                                adj[i] &= ~(1 << j)
                                adj[j] &= ~(1 << i)
        return adj
    
    # Strategy: Try to construct a 39-vertex graph
    best_n = 0
    best_adj = None
    
    # Try multiple random initializations
    for n_target in [39, 38, 37, 36, 35]:
        if time_left() < 30:
            break
        
        for attempt in range(5):
            if time_left() < 20:
                break
            
            random.seed(attempt * 1000 + n_target)
            
            # Start with a sparse random graph
            adj = random_graph(n_target, 0.15)
            
            # Prune C4s
            adj = prune_c4s(n_target, adj)
            
            # Maximize edges
            adj = maximize_edges(n_target, adj)
            
            # Check alpha
            alpha = max_independent_set_size(n_target, adj, timeout=30)
            
            if alpha < 11 and n_target > best_n:
                best_n = n_target
                best_adj = adj[:]
                if n_target == 39:
                    break
        
        if best_n == 39:
            break
    
    # If we haven't found a good graph, use the polarity graph as fallback
    if best_n < 31:
        # Construct polarity graph of PG(2, 5)
        points = []
        for x in range(5):
            for y in range(5):
                for z in range(5):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    if x != 0:
                        inv_x = pow(x, -1, 5)
                        points.append((1, y * inv_x % 5, z * inv_x % 5))
                    elif y != 0:
                        inv_y = pow(y, -1, 5)
                        points.append((0, 1, z * inv_y % 5))
                    else:
                        points.append((0, 0, 1))
        points = list(set(points))
        n = len(points)
        
        def bilinear(u, v):
            return (u[0]*v[2] + u[1]*v[1] + u[2]*v[0]) % 5
        
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if bilinear(points[i], points[j]) == 0:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        if is_c4_free(n, adj):
            alpha = max_independent_set_size(n, adj, timeout=30)
            if alpha < 11:
                best_n = n
                best_adj = adj
    
    if best_n == 0 or best_adj is None:
        return {'n': 1, 'edges': []}
    
    # Final verification
    if not is_c4_free(best_n, best_adj):
        return {'n': 1, 'edges': []}
    
    alpha = max_independent_set_size(best_n, best_adj, timeout=30)
    if alpha >= 11:
        return {'n': 1, 'edges': []}
    
    # Convert to edge list
    edge_list = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edge_list.append([i, j])
    
    return {'n': best_n, 'edges': edge_list}
