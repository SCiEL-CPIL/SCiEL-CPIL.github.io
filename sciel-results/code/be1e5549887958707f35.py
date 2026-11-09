import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 150s limit, leave 35s for safety/overhead
    
    # Fallback: Andrasfai graph And(9) on 26 vertices
    # Triangle-free, alpha = 9
    best_result = {
        'n': 26,
        'edges': [[i, j] for i in range(26) for j in range(i + 1, 26) if (j - i) % 26 % 3 == 1]
    }
    
    # Try to find a larger graph
    # R(3, 10) <= 41, so max n is 40
    
    for n in range(40, 26, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random restarts for this n
        restarts = 0
        max_restarts = 8000
        
        while time.time() < deadline - 5 and restarts < max_restarts:
            restarts += 1
            
            # Generate random maximal triangle-free graph
            adj = [0] * n
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(pairs)
            
            # Use a faster triangle check with precomputed masks
            # Instead of bit operations on full n-bit integers, 
            # we can use a simpler approach for small n
            for u, v in pairs:
                # Check if u and v share any common neighbor
                # adj[u] & adj[v] gives common neighbors
                if (adj[u] & adj[v]) == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # Check if alpha < 10 using a more efficient branch and bound
            all_mask = (1 << n) - 1
            non_adj = [0] * n
            for i in range(n):
                non_adj[i] = (all_mask ^ (1 << i)) & ~adj[i]
            
            # Precompute degree for heuristic
            degrees = [0] * n
            for i in range(n):
                degrees[i] = adj[i].bit_count()
            
            # Fast greedy check: pick vertices in fixed order, stop at 10
            def greedy_fast():
                remaining = all_mask
                count = 0
                while remaining and count < 10:
                    lsb = remaining & -remaining
                    v = lsb.bit_length() - 1
                    count += 1
                    remaining &= non_adj[v]
                return count
            
            if greedy_fast() >= 10:
                continue
            
            # Exact B&B with min-degree heuristic
            def has_independent_set_10():
                def dfs(mask, size):
                    if size == 10:
                        return True
                    if mask == 0:
                        return False
                    
                    remaining_count = mask.bit_count()
                    if remaining_count + size < 10:
                        return False
                    
                    # Min-degree heuristic for independent set
                    min_deg = n + 1
                    best_v = -1
                    temp = mask
                    while temp:
                        lsb = temp & -temp
                        v = lsb.bit_length() - 1
                        deg = (adj[v] & mask).bit_count()
                        if deg < min_deg:
                            min_deg = deg
                            best_v = v
                        temp ^= lsb
                    
                    if best_v == -1:
                        return False
                    
                    v_bit = 1 << best_v
                    
                    # Include best_v
                    if dfs(non_adj[best_v] & mask, size + 1):
                        return True
                    
                    # Exclude best_v
                    if dfs(mask & ~v_bit, size):
                        return True
                    
                    return False
                
                return dfs(all_mask, 0)
            
            if not has_independent_set_10():
                # Valid graph found
                edges = []
                for i in range(n):
                    mask = adj[i]
                    while mask:
                        lsb = mask & -mask
                        j = lsb.bit_length() - 1
                        if i < j:
                            edges.append([i, j])
                        mask ^= lsb
                return {'n': n, 'edges': edges}
        
    return best_result
