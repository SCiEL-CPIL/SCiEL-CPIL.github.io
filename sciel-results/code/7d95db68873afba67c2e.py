import time
import random

def construct():
    start_time = time.time()
    time_limit = 140.0 # Safety margin
    
    # Base solution: Paley(29)
    # n=29
    n_base = 29
    squares = {x * x % 29 for x in range(1, 29)}
    colors_base = [0 if (j - i) % 29 in squares else 1 for i in range(n_base) for j in range(i + 1, n_base)]
    
    # Try to find n=30
    # We will represent the graph as a list of masks for each color
    # mask[color][v] is a bitmask of neighbors of v in that color
    
    def get_masks(colors, n):
        masks = [[0] * n for _ in range(2)]
        for i in range(n):
            for j in range(i + 1, n):
                idx = i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)
                # Correct index calculation:
                # Pairs (0,1)..(0,n-1), (1,2)..(1,n-1)
                # Count pairs before i: i * (n-1) - i*(i+1)/2
                # Offset in row i: j - i - 1
                c = colors[idx]
                masks[c][i] |= (1 << j)
                masks[c][j] |= (1 << i)
        return masks

    def count_bits(x):
        return bin(x).count('1')

    def find_violation(masks, n):
        # Returns a list of edges (u, v) to flip, or None if valid
        for c in range(2):
            for u in range(n):
                for v in range(u + 1, n):
                    M = masks[c][u] & masks[c][v]
                    # Get vertices in M
                    verts = []
                    temp = M
                    while temp:
                        lsb = temp & -temp
                        idx = lsb.bit_length() - 1
                        verts.append(idx)
                        temp ^= lsb
                    
                    if len(verts) < 2:
                        continue
                    
                    # Check pairs in verts
                    for i in range(len(verts)):
                        w = verts[i]
                        for j in range(i + 1, len(verts)):
                            x = verts[j]
                            # Check if |M & masks[c][w] & masks[c][x]| >= 4
                            I = M & masks[c][w] & masks[c][x]
                            if count_bits(I) >= 4:
                                # Violation found
                                # We can flip any edge in the K2,2,2
                                # Let's return the edge (u, v)
                                return (u, v)
        return None

    def get_color_idx(u, v, n):
        # Returns index in colors list
        # Pairs are ordered lexicographically (i, j) with i < j
        # Index = sum_{k=0}^{u-1} (n-1-k) + (v-u-1)
        # Sum_{k=0}^{u-1} (n-1-k) = u(n-1) - u(u-1)/2
        return u * (n - 1) - u * (u - 1) // 2 + (v - u - 1)

    best_solution = {'n': 29, 'colors': colors_base}
    
    # Local search for n=30
    n_target = 30
    max_restarts = 5
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit - 10:
            break
        
        # Initialize
        colors = colors_base.copy()
        # Add edges for vertex 29
        for i in range(29):
            colors.append(random.randint(0, 1))
        
        masks = get_masks(colors, n_target)
        
        # Local search
        steps = 0
        max_steps = 5000
        while steps < max_steps and time.time() - start_time < time_limit - 5:
            violation = find_violation(masks, n_target)
            if violation is None:
                return {'n': 30, 'colors': colors}
            
            u, v = violation
            # Flip edge (u, v)
            idx = get_color_idx(u, v, n_target)
            old_c = colors[idx]
            new_c = 1 - old_c
            
            # Update masks
            masks[old_c][u] &= ~(1 << v)
            masks[old_c][v] &= ~(1 << u)
            masks[new_c][u] |= (1 << v)
            masks[new_c][v] |= (1 << u)
            
            colors[idx] = new_c
            steps += 1
            
            # Occasional random restart or perturbation if stuck?
            # For now, simple hill climbing.
            
    return best_solution
