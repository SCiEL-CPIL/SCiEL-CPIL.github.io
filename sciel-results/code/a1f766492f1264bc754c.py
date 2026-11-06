import time
import random

def _precompute_dangerous_tuples(adj, n):
    """
    Precompute the list of 4-tuples (b1, b2, c1, c2) such that the edges
    (a, b1), (a, b2), (a, c1), (a, c2), (b1, c1), (b1, c2), (b2, c1), (b2, c2)
    are all of color c, for some 'a'.
    """
    E0 = []
    E1 = []
    
    for a in range(n):
        for b1 in range(n):
            if b1 == a: continue
            for b2 in range(b1 + 1, n):
                if b2 == a: continue
                
                # Check color 0 (adj bit is 1)
                if ((adj[a] >> b1) & 1) == 1 and ((adj[a] >> b2) & 1) == 1:
                    for c1 in range(n):
                        if c1 == a or c1 == b1 or c1 == b2: continue
                        if ((adj[a] >> c1) & 1) != 1: continue
                        for c2 in range(c1 + 1, n):
                            if c2 == a or c2 == b1 or c2 == b2: continue
                            if ((adj[a] >> c2) & 1) != 1: continue
                            
                            if (((adj[b1] >> c1) & 1) == 1 and ((adj[b1] >> c2) & 1) == 1 and 
                                ((adj[b2] >> c1) & 1) == 1 and ((adj[b2] >> c2) & 1) == 1):
                                E0.append((b1, b2, c1, c2))
                
                # Check color 1 (adj bit is 0)
                if ((adj[a] >> b1) & 1) == 0 and ((adj[a] >> b2) & 1) == 0:
                    for c1 in range(n):
                        if c1 == a or c1 == b1 or c1 == b2: continue
                        if ((adj[a] >> c1) & 1) != 0: continue
                        for c2 in range(c1 + 1, n):
                            if c2 == a or c2 == b1 or c2 == b2: continue
                            if ((adj[a] >> c2) & 1) != 0: continue
                            
                            if (((adj[b1] >> c1) & 1) == 0 and ((adj[b1] >> c2) & 1) == 0 and 
                                ((adj[b2] >> c1) & 1) == 0 and ((adj[b2] >> c2) & 1) == 0):
                                E1.append((b1, b2, c1, c2))
                                
    return E0, E1

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # 1. Construct Paley(29)
    q = 29
    squares = set()
    for x in range(1, q):
        squares.add(x * x % q)
        
    n_base = q
    adj_base = [0] * n_base
    for i in range(n_base):
        for j in range(i + 1, n_base):
            diff = (j - i) % q
            if diff in squares:
                adj_base[i] |= (1 << j)
                adj_base[j] |= (1 << i)
                
    best_n = n_base
    best_adj = adj_base[:]
    
    # 2. Attempt to extend to 30
    n_target = 30
    
    # Precompute dangerous tuples for the base graph
    E0, E1 = _precompute_dangerous_tuples(adj_base, n_base)
    
    def count_violations(x, e0, e1):
        count = 0
        for t in e0:
            if ((x >> t[0]) & 1 == 0) and ((x >> t[1]) & 1 == 0) and ((x >> t[2]) & 1 == 0) and ((x >> t[3]) & 1 == 0):
                count += 1
        for t in e1:
            if ((x >> t[0]) & 1 == 1) and ((x >> t[1]) & 1 == 1) and ((x >> t[2]) & 1 == 1) and ((x >> t[3]) & 1 == 1):
                count += 1
        return count

    best_x = None
    best_viol = float('inf')
    
    # Try a few random starts
    for _ in range(50):
        if time.time() > deadline - 5.0:
            break
        x = random.getrandbits(n_base)
        cur_viol = count_violations(x, E0, E1)
        
        if cur_viol < best_viol:
            best_viol = cur_viol
            best_x = x
            if cur_viol == 0:
                break
        
        T = 10.0
        cooling = 0.99
        cur_x = x
        cur_v = cur_viol
        
        while T > 0.01 and time.time() < deadline - 5.0:
            idx = random.randint(0, n_base - 1)
            new_x = cur_x ^ (1 << idx)
            new_v = count_violations(new_x, E0, E1)
            
            delta = new_v - cur_v
            if delta < 0 or random.random() < pow(2.71828, -delta / T):
                cur_x = new_x
                cur_v = new_v
                if cur_v < best_viol:
                    best_viol = cur_v
                    best_x = cur_x
                    if cur_v == 0:
                        break
            T *= cooling
        
        if best_viol == 0:
            break
    
    if best_viol == 0:
        adj_30 = [0] * n_target
        for i in range(n_base):
            adj_30[i] = adj_base[i]
        for i in range(n_base):
            if (best_x >> i) & 1:
                adj_30[i] |= (1 << 29)
                adj_30[29] |= (1 << i)
        
        colors = []
        for i in range(n_target):
            for j in range(i + 1, n_target):
                if (adj_30[i] >> j) & 1:
                    colors.append(0)
                else:
                    colors.append(1)
        return {'n': n_target, 'colors': colors}
        
    # If SA on fixed core failed, try perturbing the core
    core_adj = adj_base[:]
    
    for _ in range(100):
        if time.time() > deadline - 10.0:
            break
            
        u = random.randint(0, n_base - 1)
        v = random.randint(0, n_base - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        if (core_adj[u] >> v) & 1:
            core_adj[u] &= ~(1 << v)
            core_adj[v] &= ~(1 << u)
        else:
            core_adj[u] |= (1 << v)
            core_adj[v] |= (1 << u)
            
        E0_new, E1_new = _precompute_dangerous_tuples(core_adj, n_base)
        
        best_x_local = None
        best_viol_local = float('inf')
        
        x = random.getrandbits(n_base)
        cur_v = count_violations(x, E0_new, E1_new)
        if cur_v < best_viol_local:
            best_viol_local = cur_v
            best_x_local = x
            
        T = 5.0
        cooling = 0.99
        cur_x = x
        cur_v = cur_v
        
        while T > 0.01 and time.time() < deadline - 5.0:
            idx = random.randint(0, n_base - 1)
            new_x = cur_x ^ (1 << idx)
            new_v = count_violations(new_x, E0_new, E1_new)
            
            delta = new_v - cur_v
            if delta < 0 or random.random() < pow(2.71828, -delta / T):
                cur_x = new_x
                cur_v = new_v
                if cur_v < best_viol_local:
                    best_viol_local = cur_v
                    best_x_local = cur_x
                    if cur_v == 0:
                        break
            T *= cooling
            
        if best_viol_local == 0:
            adj_30 = [0] * n_target
            for i in range(n_base):
                adj_30[i] = core_adj[i]
            for i in range(n_base):
                if (best_x_local >> i) & 1:
                    adj_30[i] |= (1 << 29)
                    adj_30[29] |= (1 << i)
                    
            colors = []
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    if (adj_30[i] >> j) & 1:
                        colors.append(0)
                    else:
                        colors.append(1)
            return {'n': n_target, 'colors': colors}
            
        if (core_adj[u] >> v) & 1:
            core_adj[u] &= ~(1 << v)
            core_adj[v] &= ~(1 << u)
        else:
            core_adj[u] |= (1 << v)
            core_adj[v] |= (1 << u)

    colors = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_adj[i] >> j) & 1:
                colors.append(0)
            else:
                colors.append(1)
    return {'n': best_n, 'colors': colors}
