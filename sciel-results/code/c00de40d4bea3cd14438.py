"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 256
    start_time = time.time()
    time_limit = 55.0
    
    # Build conflict graph using bitsets
    adj = [0] * n
    for i in range(n):
        a = i + 1
        for j in range(i + 1, n):
            b = j + 1
            if (a * b) % (a + b) == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    degrees = [bin(adj[i]).count('1') for i in range(n)]
    
    # Initial solution: all odd numbers
    odd_mask = 0
    for i in range(n):
        if (i + 1) % 2 == 1:
            odd_mask |= (1 << i)
    best_size = bin(odd_mask).count('1')
    best_mask = odd_mask
    
    # Also try greedy for a potentially better initial bound
    greedy_order = sorted(range(n), key=lambda x: degrees[x])
    greedy_mask = 0
    for v in greedy_order:
        if (adj[v] & greedy_mask) == 0:
            greedy_mask |= (1 << v)
    greedy_size = bin(greedy_mask).count('1')
    if greedy_size > best_size:
        best_size = greedy_size
        best_mask = greedy_mask
    
    def popcount(x):
        return bin(x).count('1')
    
    def search(candidates, current_mask, current_size):
        nonlocal best_mask, best_size, start_time
        
        if time.time() - start_time > time_limit:
            return
        
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return
        
        if current_size + popcount(candidates) <= best_size:
            return
        
        # Pick vertex with highest degree among candidates
        c = candidates
        best_v = -1
        best_deg = -1
        while c:
            lowbit = c & (-c)
            idx = lowbit.bit_length() - 1
            if degrees[idx] > best_deg:
                best_deg = degrees[idx]
                best_v = idx
            c ^= lowbit
        
        if best_v == -1:
            return
        
        # Branch 1: include best_v
        new_cands = candidates & ~(1 << best_v) & ~adj[best_v]
        search(new_cands, current_mask | (1 << best_v), current_size + 1)
        
        if time.time() - start_time > time_limit:
            return
        
        # Branch 2: exclude best_v
        search(candidates & ~(1 << best_v), current_mask, current_size)
    
    all_vertices = (1 << n) - 1
    search(all_vertices, 0, 0)
    
    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
            result.append(i + 1)
    
    if not result:
        result = [1]
    
    return sorted(result)

# EVOLVE-BLOCK-END
