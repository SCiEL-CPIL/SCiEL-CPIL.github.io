import random
import time

def construct():
    n = 40
    start_time = time.time()
    time_limit = 115  # Safety margin
    
    best_edges = []
    best_count = 0
    
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Bitmask helpers
    # neighbors[i] is an integer bitmask
    
    for _ in range(2000):  # More restarts possible now
        if time.time() - start_time > time_limit:
            break
            
        random.shuffle(all_edges)
        
        neighbors = [0] * n
        edges = []
        
        for u, v in all_edges:
            creates_c6 = False
            
            # Iterate over neighbors of u
            mask_u = neighbors[u]
            mask_v = neighbors[v]
            
            # Extract bits for x in mask_u
            x_mask = mask_u
            while x_mask:
                x = (x_mask & -x_mask).bit_length() - 1
                x_mask &= x_mask - 1
                
                # Extract bits for z in mask_v
                z_mask = mask_v
                while z_mask:
                    z = (z_mask & -z_mask).bit_length() - 1
                    z_mask &= z_mask - 1
                    
                    if x == z:
                        continue
                    
                    # Check common neighbors of x and z
                    common = neighbors[x] & neighbors[z]
                    # Exclude u, v, x, z
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    common &= ~(1 << x)
                    common &= ~(1 << z)
                    
                    if common != 0:
                        creates_c6 = True
                        break
                
                if creates_c6:
                    break
            
            if not creates_c6:
                edges.append([u, v])
                neighbors[u] |= (1 << v)
                neighbors[v] |= (1 << u)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges.copy()
            
    return best_edges
