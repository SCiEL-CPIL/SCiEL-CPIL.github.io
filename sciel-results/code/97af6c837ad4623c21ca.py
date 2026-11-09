import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Try n from 22 down to 6
    for n in range(22, 5, -1):
        if time.time() > deadline:
            break
            
        # Try to find a valid graph for this n
        # Use local search / random restarts
        
        found = False
        # Try different p values and random seeds
        # For n=22, p=0.5 is standard.
        
        # Limit attempts based on time
        attempts = 0
        while time.time() < deadline - 5.0:
            attempts += 1
            # Vary p slightly around 0.5
            p = 0.5 + (random.random() - 0.5) * 0.2
            
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < p:
                        edges.append([u, v])
            
            if is_valid(n, edges):
                return {'n': n, 'edges': edges}
            
            # If not valid, try local search?
            # For simplicity and speed, just restart with new random graph first.
            # If we have time, we can add local search.
            
            # Optimization: If n is small, we can try more.
            if n <= 20:
                # More attempts for smaller n if we haven't found 22/21
                pass
            
        # If we didn't find one for this n, try smaller n
    
    # Fallback: return a small valid graph
    # n=5 is always valid
    return {'n': 5, 'edges': []}

def is_valid(n, edges):
    if n < 6:
        return True
        
    neighbors = [0] * n
    for u, v in edges:
        neighbors[u] |= (1 << v)
        neighbors[v] |= (1 << u)
    
    full_mask = (1 << n) - 1
    
    # Iterate through all masks
    # Optimization: Skip masks with < 6 bits
    # We can iterate masks by popcount?
    # It's faster to just iterate 1 to 2^n and check bit_count.
    # But 2^22 is 4 million. bit_count is fast.
    
    for mask in range(1, 1 << n):
        if mask.bit_count() < 6:
            continue
        
        # Check if induced subgraph on 'mask' is regular
        # Get first vertex
        lsb = mask & -mask
        u = lsb.bit_length() - 1
        target_deg = (neighbors[u] & mask).bit_count()
        
        # Check other vertices
        m = mask ^ lsb
        is_reg = True
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            deg = (neighbors[v] & mask).bit_count()
            if deg != target_deg:
                is_reg = False
                break
            m ^= lsb
        
        if is_reg:
            return False
            
    return True
