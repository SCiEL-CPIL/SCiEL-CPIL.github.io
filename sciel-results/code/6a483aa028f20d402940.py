import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # GF(4) arithmetic: represent as 2-bit integers
    # GF(4) = {0, 1, 2, 3} with 2=alpha, 3=alpha+1
    # Multiplication table:
    gf4_mult = {
        (0,0):0, (0,1):0, (0,2):0, (0,3):0,
        (1,0):0, (1,1):1, (1,2):2, (1,3):3,
        (2,0):0, (2,1):2, (2,2):3, (2,3):2,
        (3,0):0, (3,1):3, (3,2):2, (3,3):1,
    }
    # Inverse table (for non-zero elements)
    gf4_inv = {1: 1, 2: 3, 3: 2}
    
    def gf4_add(a, b):
        return a ^ b
    
    def gf4_mul(a, b):
        return gf4_mult[(a, b)]
    
    def gf4_div(a, b):
        return gf4_mul(a, gf4_inv[b])
    
    # Construction for n=16: vertices are elements of GF(4)^2
    # Edge color determined by the "direction" d = u XOR v in GF(4)^2
    # d = (d1, d2) where d1, d2 in GF(4)
    # If d1 == 0: color = 0 (horizontal)
    # If d2 == 0: color = 1 (vertical) 
    # Else: slope = d2 / d1 in GF(4), non-zero
    #   slope = 1 -> color 2
    #   slope = 2 -> color 3
    #   slope = 3 -> color 4
    
    n = 16
    # Map vertex index 0..15 to (a, b) in GF(4)^2
    # vertex i -> (i // 4, i % 4)
    
    colors = []
    for i in range(n):
        ai, bi = i // 4, i % 4
        for j in range(i + 1, n):
            aj, bj = j // 4, j % 4
            d1 = gf4_add(ai, aj)
            d2 = gf4_add(bi, bj)
            
            if d1 == 0 and d2 == 0:
                # Should not happen since i < j
                c = 0
            elif d1 == 0:
                c = 0  # horizontal
            elif d2 == 0:
                c = 1  # vertical
            else:
                slope = gf4_div(d2, d1)
                if slope == 1:
                    c = 2
                elif slope == 2:
                    c = 3
                else:  # slope == 3
                    c = 4
            colors.append(c)
    
    # Verify C4-freeness
    def has_c4(color_val, n, colors):
        """Check if any color class contains a C4."""
        # Build adjacency for this color
        adj = [set() for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if colors[idx] == color_val:
                    adj[i].add(j)
                    adj[j].add(i)
                idx += 1
        
        # Check for C4: two vertices with >= 2 common neighbors
        for i in range(n):
            for j in range(i + 1, n):
                common = adj[i] & adj[j]
                if len(common) >= 2:
                    return True
        return False
    
    # Verify all colors are C4-free
    valid = True
    for c in range(5):
        if has_c4(c, n, colors):
            valid = False
            break
    
    if not valid:
        # Fallback to known valid construction (should not happen)
        n = 11
        colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    
    return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
