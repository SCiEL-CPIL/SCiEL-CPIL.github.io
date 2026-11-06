import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Fallback: 32-vertex Greenwood-Gleason coloring
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log_map, x = {}, 1
    for e in range(15):
        log_map[x] = e
        x = mul(x, 2)

    def base_colors_32():
        n = 32
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                if i // 16 == j // 16:
                    colors.append(log_map[(i % 16) ^ (j % 16)] % 3)
                else:
                    colors.append(3)
        return {'n': n, 'colors': colors}

    best_result = base_colors_32()
    
    N = 50
    num_edges = N * (N - 1) // 2
    
    # NumPy optimization
    # Create a 50x50 matrix of colors
    # C[i,j] is the color of edge (i,j)
    
    def count_mono_triangles_vec(C):
        # C is 50x50 int array
        # Triangle (i,j,k) is mono if C[i,j]==C[i,k]==C[j,k]
        # Sum over all i<j<k
        count = 0
        # Vectorized check is tricky for all triples at once without large memory
        # But N=50 is small. 
        # We can loop over k and vectorize over i,j < k
        for k in range(2, N):
            sub = C[:k, :k]
            # We need pairs i < j < k
            # Let's just use a simple loop for counting, it's O(N^3) = 125k ops, fast enough
            # But we want speed.
            # Actually, 125k ops in Python is ~0.01s. 
            # The bottleneck is the delta calculation in local search if not vectorized.
            pass
        return count

    def local_search_numpy():
        # Random init
        C = np.random.randint(0, 4, size=(N, N))
        np.fill_diagonal(C, -1)
        
        # Symmetrize? No, we only use upper triangle.
        # Let's store colors in a flat array for the return, but use matrix for speed.
        
        def get_delta(u, v, new_color):
            # Calculate change in mono triangles if C[u,v] changes to new_color
            # Triangles are (u,v,w) for w != u,v
            # Old color: C[u,v]
            # New color: new_color
            # Triangles affected: those containing edge (u,v)
            # w ranges from 0 to N-1, w != u, w != v
            
            # We can vectorize this:
            # Let U = C[u, :], V = C[v, :]
            # Mask out u and v
            # old_mono = 1 if C[u,v] == U[w] == V[w]
            # new_mono = 1 if new_color == U[w] == V[w]
            
            U = C[u]
            V = C[v]
            
            # Create mask for valid w
            mask = np.ones(N, dtype=bool)
            mask[u] = False
            mask[v] = False
            
            U_valid = U[mask]
            V_valid = V[mask]
            
            old_c = C[u, v]
            
            old_mono = ((U_valid == old_c) & (V_valid == old_c)).sum()
            new_mono = ((U_valid == new_c) & (V_valid == new_c)).sum()
            
            return new_mono - old_mono

        # Initial count (approximate or exact)
        # We just need to improve.
        
        stuck = 0
        while time.time() < deadline:
            # Pick random edge
            u = random.randint(0, N-1)
            v = random.randint(0, N-1)
            if u == v: continue
            if u > v: u, v = v, u
            
            old_c = C[u, v]
            
            best_delta = 0
            best_c = -1
            
            for new_c in range(4):
                if new_c == old_c: continue
                d = get_delta(u, v, new_c)
                if d < best_delta:
                    best_delta = d
                    best_c = new_c
            
            if best_c != -1:
                C[u, v] = best_c
                C[v, u] = best_c
                stuck = 0
            else:
                stuck += 1
                if stuck > 500:
                    # Perturb
                    for _ in range(5):
                        ru = random.randint(0, N-1)
                        rv = random.randint(0, N-1)
                        if ru != rv:
                            C[ru, rv] = random.randint(0, 3)
                            C[rv, ru] = C[ru, rv]
                    stuck = 0
        
        # Check if solution found
        # Count mono triangles
        # If 0, return
        mono_count = 0
        for i in range(N):
            for j in range(i+1, N):
                c_ij = C[i, j]
                for k in range(j+1, N):
                    if C[i, k] == c_ij and C[j, k] == c_ij:
                        mono_count += 1
        
        if mono_count == 0:
            # Extract colors
            colors = []
            for i in range(N):
                for j in range(i+1, N):
                    colors.append(int(C[i, j]))
            return colors
        return None

    found = local_search_numpy()
    if found:
        return {'n': N, 'colors': found}
        
    return best_result
