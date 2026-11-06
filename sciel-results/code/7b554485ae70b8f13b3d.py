import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 16
    # Initialize with Z_16 additive coloring
    # Colors are in 0..15
    colors = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            colors[i][j] = (i + j) % 16
            colors[j][i] = colors[i][j]
    
    # List of all K4s
    k4s = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    k4s.append((i,j,k,l))
    
    def get_colors_in_k4(v1, v2, v3, v4):
        return {
            colors[v1][v2],
            colors[v1][v3],
            colors[v1][v4],
            colors[v2][v3],
            colors[v2][v4],
            colors[v3][v4]
        }
    
    def is_valid():
        for v1, v2, v3, v4 in k4s:
            if len(get_colors_in_k4(v1, v2, v3, v4)) < 5:
                return False
        return True
        
    # Check initial validity
    if is_valid():
        # Return flat list
        return [colors[i][j] for i in range(n) for j in range(i+1, n)]
        
    # Repair loop
    # We want to keep colors in 0..15
    palette = list(range(16))
    
    # Identify bad K4s
    bad_k4s = [k4 for k4 in k4s if len(get_colors_in_k4(*k4)) < 5]
    
    if not bad_k4s:
        return [colors[i][j] for i in range(n) for j in range(i+1, n)]

    # Try to fix bad K4s by changing one edge color
    # Randomized local search
    best_val = 17 # Fallback
    
    while time.time() < deadline:
        if is_valid():
            return [colors[i][j] for i in range(n) for j in range(i+1, n)]
            
        # Pick a random bad K4
        bad_k4s = [k4 for k4 in k4s if len(get_colors_in_k4(*k4)) < 5]
        if not bad_k4s:
            break
            
        v1, v2, v3, v4 = random.choice(bad_k4s)
        edges = [(v1,v2), (v1,v3), (v1,v4), (v2,v3), (v2,v4), (v3,v4)]
        random.shuffle(edges)
        
        fixed = False
        for u, v in edges:
            old_color = colors[u][v]
            # Try changing to a different color in palette
            # We need the new set of colors to have size >= 5
            # Current set S
            S = get_colors_in_k4(v1, v2, v3, v4)
            # Remove old color from S? No, we replace it.
            # If we change edge (u,v) to c_new, the new set is (S - {old_color}) U {c_new}
            # Note: if old_color appears multiple times in the K4, S is a set, so removing it might not reduce size if another edge has it.
            # We must be careful.
            
            # Let's just try all colors in palette
            for c_new in palette:
                if c_new == old_color:
                    continue
                # Simulate
                temp_color = colors[u][v]
                colors[u][v] = c_new
                colors[v][u] = c_new
                
                if len(get_colors_in_k4(v1, v2, v3, v4)) >= 5:
                    # Check if this breaks other K4s? 
                    # This is expensive. 
                    # For now, accept if it fixes this one. 
                    # We will re-check global validity in the outer loop.
                    fixed = True
                    break
                else:
                    # Revert
                    colors[u][v] = temp_color
                    colors[v][u] = temp_color
            
            if fixed:
                break
            else:
                # Revert if not fixed
                colors[u][v] = old_color
                colors[v][u] = old_color
        
        if not fixed:
            # Try random perturbation
            # Pick a random edge and change its color
            u, v = random.randint(0, n-1), random.randint(0, n-1)
            if u == v: continue
            if u > v: u, v = v, u
            colors[u][v] = random.choice(palette)
            colors[v][u] = colors[u][v]

    # If we fail to get 16 colors valid, return the 17-color seed
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
