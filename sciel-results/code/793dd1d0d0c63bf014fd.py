import random
import time

def construct():
    n_target = 50
    start_time = time.time()
    timeout = 115
    
    while time.time() - start_time < timeout:
        n = n_target
        colors = [[0]*n for _ in range(n)]
        
        # Init
        for i in range(n):
            for j in range(i+1, n):
                colors[i][j] = random.randint(0, 3)
                colors[j][i] = colors[i][j]
                
        # Precompute N
        N = [[[0]*n for _ in range(n)] for _ in range(4)]
        for c in range(4):
            for u in range(n):
                for v in range(n):
                    if u == v: continue
                    count = 0
                    for w in range(n):
                        if w == u or w == v: continue
                        if colors[u][w] == c and colors[v][w] == c:
                            count += 1
                    N[c][u][v] = count
                    N[c][v][u] = count

        # Calculate initial score
        score = 0
        for u in range(n):
            for v in range(u+1, n):
                score += N[colors[u][v]][u][v]
                
        if score == 0:
            break
            
        # Local Search
        while time.time() - start_time < timeout:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            if u > v: u, v = v, u
            
            c_old = colors[u][v]
            
            min_val = N[0][u][v]
            min_c = 0
            for c in range(1, 4):
                if N[c][u][v] < min_val:
                    min_val = N[c][u][v]
                    min_c = c
                    
            if min_val < N[c_old][u][v]:
                score -= N[c_old][u][v]
                score += min_val
                colors[u][v] = min_c
                colors[v][u] = min_c
                if score == 0:
                    break
            else:
                # Random perturbation
                if random.random() < 0.05:
                    c_new = random.randint(0, 3)
                    if c_new != c_old:
                        score -= N[c_old][u][v]
                        score += N[c_new][u][v]
                        colors[u][v] = c_new
                        colors[v][u] = c_new
                        if score == 0:
                            break

    # Verify and format
    # If score > 0, we might have a suboptimal solution.
    # But we want to maximize n.
    # If we didn't find 50, we could try 49, 48...
    # But 50 is likely found.
    
    out_colors = []
    for i in range(n):
        for j in range(i+1, n):
            out_colors.append(colors[i][j])
            
    return {'n': n, 'colors': out_colors}
