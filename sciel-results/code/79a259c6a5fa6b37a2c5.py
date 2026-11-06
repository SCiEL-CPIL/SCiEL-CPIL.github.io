import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for 150s limit
    
    # 1. Generate base n=32 coloring (Greenwood-Gleason)
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
        
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
        
    # Base colors for 32 vertices
    base_colors = {}
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                base_colors[(i, j)] = log[(i % 16) ^ (j % 16)] % 3
            else:
                base_colors[(i, j)] = 3
                
    N_TARGET = 51
    # Initialize full matrix for N_TARGET
    # mat[i][j] for i < j
    mat = [[-1] * N_TARGET for _ in range(N_TARGET)]
    
    # Fill base
    for (i, j), c in base_colors.items():
        mat[i][j] = c
        mat[j][i] = c
        
    # Greedy extension for vertices 32 to 50
    for v in range(32, N_TARGET):
        for u in range(v):
            if mat[u][v] != -1:
                continue
            best_color = 0
            min_violations = float('inf')
            for c in range(4):
                violations = 0
                # Check triangles (u, v, w) for w < u
                # Edges: (u,v) is c, (u,w) is mat[u][w], (v,w) is mat[v][w]
                for w in range(u):
                    if mat[u][w] == c and mat[v][w] == c:
                        violations += 1
                if violations < min_violations:
                    min_violations = violations
                    best_color = c
            mat[u][v] = best_color
            mat[v][u] = best_color
            
    # Now we have a candidate coloring for N=51.
    # Count monochromatic triangles
    def count_tris():
        cnt = 0
        for i in range(N_TARGET):
            for j in range(i+1, N_TARGET):
                c_ij = mat[i][j]
                for k in range(j+1, N_TARGET):
                    if mat[i][k] == c_ij and mat[j][k] == c_ij:
                        cnt += 1
        return cnt
        
    current_tris = count_tris()
    best_tris = current_tris
    best_mat = [row[:] for row in mat]
    
    # Local Search: Simulated Annealing
    # Precompute indices for faster access? 
    # For speed, we'll iterate edges randomly.
    
    # To speed up triangle count update, we can just recalculate locally.
    # But full recount is O(N^3) = 51^3 ~ 132k ops. 
    # 100k steps * 132k ops = 13 billion ops. Too slow in Python.
    
    # We need incremental update.
    # When changing mat[u][v], only triangles involving edge (u,v) change.
    # Triangles are (u, v, w) for all w != u, v.
    # There are N-2 such triangles.
    
    def get_edge(u, v):
        return mat[u][v] if u < v else mat[v][u]
        
    def set_edge(u, v, c):
        if u < v:
            mat[u][v] = c
        else:
            mat[v][u] = c
            
    # Incremental delta calculation
    def delta_tris(u, v, old_c, new_c):
        delta = 0
        for w in range(N_TARGET):
            if w == u or w == v:
                continue
            c_1 = get_edge(u, w)
            c_2 = get_edge(v, w)
            
            # Triangle (u, v, w)
            # Old state:
            if old_c == c_1 and old_c == c_2:
                delta -= 1 # Was mono, now might not be
            # New state:
            if new_c == c_1 and new_c == c_2:
                delta += 1 # Is mono now
            
        return delta
        
    # Main SA Loop
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick random edge
        u = random.randint(0, N_TARGET - 1)
        v = random.randint(0, N_TARGET - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        old_c = mat[u][v]
        new_c = random.randint(0, 3)
        if new_c == old_c:
            continue
            
        d = delta_tris(u, v, old_c, new_c)
        
        if d < 0 or random.random() < pow(2.71828, -d / T):
            mat[u][v] = new_c
            mat[v][u] = new_c
            current_tris += d
            if current_tris < best_tris:
                best_tris = current_tris
                best_mat = [row[:] for row in mat]
                if best_tris == 0:
                    break
        T *= cooling
        if T < T_min:
            T = T_min
            
    # Check if we found a valid solution for N=51
    if best_tris == 0:
        # Flatten best_mat
        colors = []
        for i in range(N_TARGET):
            for j in range(i+1, N_TARGET):
                colors.append(best_mat[i][j])
        return {'n': N_TARGET, 'colors': colors}
        
    # If not 0, try to find largest valid n
    # We can check subgraphs of the best_mat found
    # Since best_mat is a subgraph of the full graph, we can just take first k vertices
    # But the "best_mat" might have high violations in the whole graph, but a subgraph might be clean.
    # Let's verify validity for n from 50 down to 32
    
    # To speed up, we can just check the current `mat` or `best_mat`.
    # Let's use best_mat.
    
    def is_valid(mat_to_check, n):
        for i in range(n):
            for j in range(i+1, n):
                c = mat_to_check[i][j]
                for k in range(j+1, n):
                    if mat_to_check[i][k] == c and mat_to_check[j][k] == c:
                        return False
        return True
        
    # We should probably run the SA for each n? No, that's too slow.
    # The SA was run on N=51. The result is a 51-vertex graph.
    # If it's invalid, maybe a subgraph is valid.
    # Let's check n=50, 49, ...
    
    for n in range(50, 31, -1):
        if is_valid(best_mat, n):
            colors = []
            for i in range(n):
                for j in range(i+1, n):
                    colors.append(best_mat[i][j])
            return {'n': n, 'colors': colors}
            
    # Fallback: Return the base 32
    colors = []
    for i in range(32):
        for j in range(i+1, 32):
            colors.append(base_colors[(i,j)])
    return {'n': 32, 'colors': colors}
