import time

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin
    
    n = 256
    odds = [i for i in range(1, n + 1) if i % 2 != 0]
    
    # Identify safe evens
    safe_evens = []
    for e in range(2, n + 1, 2):
        # Find odd part
        m = e
        while m % 2 == 0:
            m //= 2
            
        # Find odd divisors of m^2
        # Since m is odd, divisors of m^2 are odd
        m_sq = m * m
        divs = []
        for i in range(1, int(m_sq**0.5) + 1):
            if m_sq % i == 0:
                divs.append(i)
                if i != m_sq // i:
                    divs.append(m_sq // i)
        
        is_safe = True
        for d in divs:
            o = d - e
            if 1 <= o <= n and o % 2 != 0 and o != e:
                is_safe = False
                break
                
        if is_safe:
            safe_evens.append(e)
            
    # Build graph of conflicts among safe_evens
    k = len(safe_evens)
    # adjacency list or bitset
    # k is likely small (< 50)
    adj = [[False] * k for _ in range(k)]
    for i in range(k):
        if time.time() > deadline:
            break
        for j in range(i + 1, k):
            e1 = safe_evens[i]
            e2 = safe_evens[j]
            if (e1 + e2) != 0 and (e1 * e2) % (e1 + e2) == 0:
                adj[i][j] = True
                adj[j][i] = True
                
    # Find MIS
    # Simple backtracking
    best_size = 0
    best_set = []
    
    def backtrack(idx, current_set, available):
        nonlocal best_size, best_set
        if time.time() > deadline:
            return
        if idx == k:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
            return
            
        # Pruning
        if len(current_set) + len(available) <= best_size:
            return
            
        # Option 1: Exclude safe_evens[idx]
        backtrack(idx + 1, current_set, available)
        
        # Option 2: Include safe_evens[idx]
        new_available = [v for v in available if not adj[idx][v]]
        backtrack(idx + 1, current_set + [safe_evens[idx]], new_available)
        
    # Initial available is all vertices
    # Better: standard Bron-Kerbosch or simple DFS
    # Given small k, simple DFS is fine.
    # Let's optimize the DFS slightly
    
    def mis_search(idx, current, avail_mask):
        nonlocal best_size, best_set
        if time.time() > deadline:
            return
        if idx == k:
            if len(current) > best_size:
                best_size = len(current)
                best_set = current[:]
            return
            
        # Count bits in avail_mask
        # Simple count
        count = bin(avail_mask).count('1')
        if len(current) + count <= best_size:
            return
            
        # Exclude
        mis_search(idx + 1, current, avail_mask)
        
        # Include
        # New mask: remove idx and neighbors
        if avail_mask & (1 << idx):
            new_mask = avail_mask
            # Clear bit idx
            new_mask &= ~(1 << idx)
            # Clear neighbors
            for v in range(k):
                if adj[idx][v]:
                    new_mask &= ~(1 << v)
            mis_search(idx + 1, current + [safe_evens[idx]], new_mask)
            
    initial_mask = (1 << k) - 1
    mis_search(0, [], initial_mask)
    
    result = odds + best_set
    return result
