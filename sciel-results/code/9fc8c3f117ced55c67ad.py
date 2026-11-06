import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin for 120s limit
    
    # Target n=26 first
    # We run local search for n=26
    # If found, try n=27
    
    best_result = None
    
    # Try to find n=26
    # Use a generous time limit for the first attempt
    # If it fails, try again with different seed? 
    # The local search should be robust.
    
    # We can run multiple restarts within the time limit
    # The local_search function handles restarts internally.
    
    res_26 = local_search(26, deadline - 10.0)
    
    if res_26:
        # Try to extend to 27
        # Allocate remaining time
        remaining = deadline - time.time()
        if remaining > 5.0:
            res_27 = local_search(27, remaining - 5.0)
            if res_27:
                return res_27
        
        return res_26
    
    # If n=26 failed, which is very unlikely, try again with remaining time
    remaining = deadline - time.time()
    if remaining > 5.0:
        res_26_retry = local_search(26, remaining - 5.0)
        if res_26_retry:
            return res_26_retry
            
    # Absolute fallback
    return {'n': 1, 'colors': []}

def local_search(n, time_limit):
    num_edges = n * (n - 1) // 2
    # Precompute edge indices
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
            
    # Helper for popcount
    def popcount(x):
        try:
            return x.bit_count()
        except AttributeError:
            return bin(x).count('1')

    def get_initial_state():
        colors = [random.randint(0, 4) for _ in range(num_edges)]
        adj = [ [0] * n for _ in range(5) ]
        
        for idx, (u, v) in enumerate(edges):
            c = colors[idx]
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
            
        # Calculate total violations
        total_violations = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj[c][u] & adj[c][v]
                    k = popcount(common)
                    if k >= 2:
                        total_violations += k * (k - 1) // 2
                        
        return colors, adj, total_violations

    def calculate_delta(u, v, c_old, c_new, adj):
        # Delta for removing (u, v) from c_old
        delta_remove = 0
        Nv_old = adj[c_old][v]
        Nu_old = adj[c_old][u]
        
        # Iterate bits of Nv_old
        # Exclude u
        w = Nv_old & ~(1 << u)
        while w:
            lsb = w & -w
            w ^= lsb
            w_idx = lsb.bit_length() - 1
            # d = popcount(adj[c_old][u] & adj[c_old][w_idx])
            d = popcount(adj[c_old][u] & adj[c_old][w_idx])
            delta_remove += d - 1
            
        # Iterate bits of Nu_old
        # Exclude v
        w = Nu_old & ~(1 << v)
        while w:
            lsb = w & -w
            w ^= lsb
            w_idx = lsb.bit_length() - 1
            d = popcount(adj[c_old][v] & adj[c_old][w_idx])
            delta_remove += d - 1
            
        # Delta for adding (u, v) to c_new
        delta_add = 0
        Nv_new = adj[c_new][v]
        Nu_new = adj[c_new][u]
        
        w = Nv_new & ~(1 << u)
        while w:
            lsb = w & -w
            w ^= lsb
            w_idx = lsb.bit_length() - 1
            d = popcount(adj[c_new][u] & adj[c_new][w_idx])
            delta_add += d
            
        w = Nu_new & ~(1 << v)
        while w:
            lsb = w & -w
            w ^= lsb
            w_idx = lsb.bit_length() - 1
            d = popcount(adj[c_new][v] & adj[c_new][w_idx])
            delta_add += d
            
        return delta_add - delta_remove

    start_time = time.time()
    temperature = 2.0
    cooling_rate = 0.999
    min_temperature = 0.01
    no_improve_steps = 0
    max_no_improve = 5000
    best_violations = float('inf')
    best_colors = None
    
    while time.time() < start_time + time_limit:
        colors, adj, total_violations = get_initial_state()
        
        if total_violations == 0:
            return {'n': n, 'colors': colors}
            
        best_violations = total_violations
        best_colors = colors[:]
        no_improve_steps = 0
        
        while time.time() < start_time + time_limit:
            e_idx = random.randint(0, num_edges - 1)
            u, v = edges[e_idx]
            old_color = colors[e_idx]
            
            new_color = random.randint(0, 4)
            if new_color == old_color:
                continue
                
            delta = calculate_delta(u, v, old_color, new_color, adj)
            
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                # Accept
                colors[e_idx] = new_color
                
                # Update adj
                adj[old_color][u] ^= (1 << v)
                adj[old_color][v] ^= (1 << u)
                adj[new_color][u] ^= (1 << v)
                adj[new_color][v] ^= (1 << u)
                
                total_violations += delta
                
                if total_violations == 0:
                    return {'n': n, 'colors': colors}
                    
                if total_violations < best_violations:
                    best_violations = total_violations
                    best_colors = colors[:]
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
                    
            else:
                no_improve_steps += 1
                
            temperature *= cooling_rate
            if temperature < min_temperature:
                temperature = min_temperature
                
            if no_improve_steps > max_no_improve:
                break
                
    return {'n': n, 'colors': best_colors} if best_colors and best_violations == 0 else None

