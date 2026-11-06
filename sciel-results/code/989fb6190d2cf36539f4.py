import time
import random

def construct():
    deadline = time.time() + 130.0
    n_max = 43
    
    # Use bitsets for adjacency
    def is_c4_free(adj, n):
        for u in range(n):
            for v in range(u+1, n):
                common = adj[u] & adj[v]
                if common & (common - 1):
                    return False
        return True
    
    def greedy_alpha_lb(adj, n):
        """Lower bound on independence number"""
        remaining = (1 << n) - 1
        count = 0
        while remaining:
            best_v = -1
            best_deg = n + 1
            for v in range(n):
                if remaining & (1 << v):
                    non_adj = remaining & ~(adj[v] | (1 << v))
                    deg = bin(non_adj).count('1')
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
            count += 1
            non_adj = remaining & ~(adj[best_v] | (1 << best_v))
            remaining &= ~(non_adj | (1 << best_v))
        return count
    
    def find_indep_11(adj, n):
        """Check if independent set of size 11 exists"""
        if n < 11:
            return False
        
        def bt(cands, size):
            if time.time() > deadline - 3:
                return False
            if size >= 11:
                return True
            if not cands:
                return False
            if bin(cands).count('1') < 11 - size:
                return False
            
            # Pick vertex with fewest non-neighbors in cands
            best_v = -1
            best_cnt = n + 1
            for v in range(n):
                if cands & (1 << v):
                    non = cands & ~(adj[v] | (1 << v))
                    cnt = bin(non).count('1')
                    if cnt < best_cnt:
                        best_cnt = cnt
                        best_v = v
            
            if best_v < 0:
                return False
            
            v = best_v
            rest = cands & ~(1 << v)
            non = rest & ~adj[v]
            
            if bt(non, size + 1):
                return True
            if bt(rest, size):
                return True
            return False
        
        return bt((1 << n) - 1, 0)
    
    def try_build(seed):
        random.seed(seed)
        adj = [0] * n_max
        n = 0
        
        for target in range(1, n_max + 1):
            if time.time() > deadline - 5:
                break
            
            added = False
            for attempt in range(100):
                if target == 1:
                    adj[0] = 0
                    n = 1
                    added = True
                    break
                
                existing = list(range(target - 1))
                random.shuffle(existing)
                
                # Greedily select neighbors for new vertex
                S = []
                for v in existing:
                    ok = True
                    for u in S:
                        if adj[u] & adj[v]:
                            ok = False
                            break
                    if ok:
                        S.append(v)
                
                # Try different subset sizes
                # Sort S by some heuristic
                for s_size in range(len(S), 0, -1):
                    # Take first s_size elements of S
                    neighbors = S[:s_size]
                    
                    # Apply
                    new_adj = adj[:]
                    for v in neighbors:
                        new_adj[target-1] |= (1 << v)
                        new_adj[v] |= (1 << (target-1))
                    
                    # Check C4-free
                    if not is_c4_free(new_adj, target):
                        continue
                    
                    # Check alpha
                    if greedy_alpha_lb(new_adj, target) >= 11:
                        continue
                    
                    if target <= 38:
                        if find_indep_11(new_adj, target):
                            continue
                    
                    # Accept
                    for v in neighbors:
                        adj[target-1] |= (1 << v)
                        adj[v] |= (1 << (target-1))
                    n = target
                    added = True
                    break
                
                if added:
                    break
            
            if not added:
                # Try with just 1 neighbor
                for v in range(target - 1):
                    new_adj = adj[:]
                    new_adj[target-1] |= (1 << v)
                    new_adj[v] |= (1 << (target-1))
                    if not is_c4_free(new_adj, target):
                        continue
                    if greedy_alpha_lb(new_adj, target) >= 11:
                        continue
                    if target <= 38:
                        if find_indep_11(new_adj, target):
                            continue
                    adj[target-1] |= (1 << v)
                    adj[v] |= (1 << (target-1))
                    n = target
                    added = True
                    break
            
            if not added:
                break
        
        return n, adj
    
    best_n = 0
    best_adj = [0] * n_max
    
    for seed in range(200):
        if time.time() > deadline - 10:
            break
        
        n, adj = try_build(seed)
        if n > best_n:
            best_n = n
            best_adj = adj[:]
            assert is_c4_free(best_adj, best_n)
            if best_n >= 39:
                break
    
    edges = []
    for u in range(best_n):
        for v in range(u+1, best_n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
    
    assert is_c4_free(best_adj, best_n)
    
    return {'n': best_n, 'edges': edges}
