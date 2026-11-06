import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 115.0  # 5s safety margin
    
    # 1. Generate Paley 29
    n_29 = 29
    squares_29 = {x * x % 29 for x in range(1, 29)}
    colors_29 = [0 if (j - i) % 29 in squares_29 else 1 for i in range(n_29) for j in range(i + 1, n_29)]
    
    # 2. Exact verification of Paley 29
    # n=29, C(29,6) ~ 4.9M tuples.
    # We can check this exactly.
    def check_valid_exact(colors, n):
        # Build adjacency matrix for O(1) access
        mat = [[0]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                mat[i][j] = c
                mat[j][i] = c
                idx += 1
        
        # Check for K2,2,2 in color 0 and 1
        # K2,2,2 is defined by 3 pairs of vertices (a,b), (c,d), (e,f)
        # such that all edges between pairs are of the same color.
        # Vertices: a,b,c,d,e,f distinct.
        # Edges: a-c, a-d, b-c, b-d, a-e, a-f, b-e, b-f, c-e, c-f, d-e, d-f
        # Total 12 edges.
        
        # To speed up, we can iterate over 6-tuples.
        # 4.9M iterations is okay in 110s?
        # 4.9M * 12 checks = 58M operations.
        # Python might take ~5-10 seconds.
        # This is acceptable.
        
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    for d in range(c+1, n):
                        for e in range(d+1, n):
                            for f in range(e+1, n):
                                # Check color 0
                                if (mat[a][c] == 0 and mat[a][d] == 0 and mat[b][c] == 0 and mat[b][d] == 0 and
                                    mat[a][e] == 0 and mat[a][f] == 0 and mat[b][e] == 0 and mat[b][f] == 0 and
                                    mat[c][e] == 0 and mat[c][f] == 0 and mat[d][e] == 0 and mat[d][f] == 0):
                                    return False
                                # Check color 1
                                if (mat[a][c] == 1 and mat[a][d] == 1 and mat[b][c] == 1 and mat[b][d] == 1 and
                                    mat[a][e] == 1 and mat[a][f] == 1 and mat[b][e] == 1 and mat[b][f] == 1 and
                                    mat[c][e] == 1 and mat[c][f] == 1 and mat[d][e] == 1 and mat[d][f] == 1):
                                    return False
        return True

    if check_valid_exact(colors_29, n_29):
        # We have a valid n=29. Store it.
        result_29 = {'n': 29, 'colors': colors_29}
    else:
        # Fallback if Paley 29 is somehow invalid (should not happen)
        # Use a smaller known valid graph, e.g., Paley 17
        n_17 = 17
        squares_17 = {x * x % 17 for x in range(1, 17)}
        colors_17 = [0 if (j - i) % 17 in squares_17 else 1 for i in range(n_17) for j in range(i + 1, n_17)]
        result_29 = {'n': 17, 'colors': colors_17}
        
    # 3. Try to find n=30 using Simulated Annealing
    if time.time() < deadline - 5.0:
        n_30 = 30
        # Start with random coloring or extend Paley 29?
        # Extension is smarter.
        colors_30 = colors_29[:]
        for j in range(29):
            colors_30.append(random.randint(0, 1))
            
        mat_30 = [[0]*30 for _ in range(30)]
        idx = 0
        for i in range(30):
            for j in range(i+1, 30):
                c = colors_30[idx]
                mat_30[i][j] = c
                mat_30[j][i] = c
                idx += 1
                
        # Precompute edge indices for fast access
        edge_idx = {}
        idx = 0
        for i in range(30):
            for j in range(i+1, 30):
                edge_idx[(i,j)] = idx
                idx += 1
                
        # Precompute all 6-tuples for fast violation counting
        # This is expensive but we do it once
        tuples_6 = []
        for a in range(30):
            for b in range(a+1, 30):
                for c in range(b+1, 30):
                    for d in range(c+1, 30):
                        for e in range(d+1, 30):
                            for f in range(e+1, 30):
                                tuples_6.append((a,b,c,d,e,f))
                                
        # Fast violation count for a given state
        def count_violations(mat):
            count = 0
            for (a,b,c,d,e,f) in tuples_6:
                # Check color 0
                if (mat[a][c] == 0 and mat[a][d] == 0 and mat[b][c] == 0 and mat[b][d] == 0 and
                    mat[a][e] == 0 and mat[a][f] == 0 and mat[b][e] == 0 and mat[b][f] == 0 and
                    mat[c][e] == 0 and mat[c][f] == 0 and mat[d][e] == 0 and mat[d][f] == 0):
                    count += 1
                # Check color 1
                if (mat[a][c] == 1 and mat[a][d] == 1 and mat[b][c] == 1 and mat[b][d] == 1 and
                    mat[a][e] == 1 and mat[a][f] == 1 and mat[b][e] == 1 and mat[b][f] == 1 and
                    mat[c][e] == 1 and mat[c][f] == 1 and mat[d][e] == 1 and mat[d][f] == 1):
                    count += 1
            return count
        
        # Simulated Annealing
        current_mat = mat_30
        current_colors = colors_30[:]
        current_energy = count_violations(current_mat)
        
        best_mat = [row[:] for row in current_mat]
        best_colors = current_colors[:]
        best_energy = current_energy
        
        T = 10.0
        T_min = 0.01
        cooling_rate = 0.999
        
        while time.time() < deadline - 2.0 and T > T_min:
            # Pick a random edge to flip
            u = random.randint(0, 29)
            v = random.randint(0, 29)
            if u == v:
                continue
            if u > v:
                u, v = v, u
                
            old_c = current_mat[u][v]
            new_c = 1 - old_c
            
            # Calculate delta energy
            # This requires checking which 6-tuples are affected
            # For efficiency, we just recalculate (expensive but simple)
            # Or we can be smarter: only tuples containing edge (u,v)
            
            # For now, use the slow method
            current_mat[u][v] = new_c
            current_mat[v][u] = new_c
            
            new_energy = count_violations(current_mat)
            delta = new_energy - current_energy
            
            if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_mat = [row[:] for row in current_mat]
                    # Update best_colors
                    idx = edge_idx[(u,v)]
                    best_colors = current_colors[:]
                    best_colors[idx] = new_c
                    current_colors[idx] = new_c
                    
                    if best_energy == 0:
                        # Found a valid coloring!
                        return {'n': 30, 'colors': best_colors}
            else:
                # Reject move
                current_mat[u][v] = old_c
                current_mat[v][u] = old_c
            
            T *= cooling_rate
        
        # If we found a valid coloring, return it
        if best_energy == 0:
            return {'n': 30, 'colors': best_colors}
            
    return result_29
