import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin
    
    # Function to check validity and return colors if valid
    def is_valid(n, colors):
        # colors is a list of length n*(n-1)//2
        # index for (i, j) i<j is i*n - i*(i+1)//2 + (j-i-1)
        # This mapping is tricky. Let's use a 2D array for ease, then flatten.
        # But for speed in SA, we need fast access.
        # Let's stick to a list of lists or a flat list with precomputed indices.
        pass

    # We will implement a specialized SA for K_n
    def solve_sa(n, max_time=10.0):
        if n < 2:
            return [0] * (n*(n-1)//2)
        
        num_edges = n * (n - 1) // 2
        # colors[edge_index]
        # We need a fast way to get color(i, j)
        # Let's use a 2D list for colors to make triangle checking easier, 
        # and convert to flat list at the end.
        # 2D list access is slower than flat, but triangle checking involves 3 accesses.
        # Let's use a flat list and precompute edge indices for all pairs (i,j).
        
        # Precompute edge index for (i, j)
        # idx(i, j) = i*n - i*(i+1)//2 + (j-i-1) for i < j
        # This is O(1) but involves division.
        # Alternative: Store a matrix of indices.
        idx_matrix = [[0]*n for _ in range(n)]
        k = 0
        for i in range(n):
            for j in range(i+1, n):
                idx_matrix[i][j] = k
                idx_matrix[j][i] = k
                k += 1
        
        # Initialize colors randomly
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        # Calculate initial energy
        def calc_energy():
            e = 0
            for i in range(n):
                for j in range(i+1, n):
                    c_ij = colors[idx_matrix[i][j]]
                    for k in range(j+1, n):
                        # Triangle (i, j, k)
                        if c_ij == colors[idx_matrix[j][k]] and c_ij == colors[idx_matrix[i][k]]:
                            e += 1
                    for k in range(i):
                        # Triangle (k, i, j)
                        if c_ij == colors[idx_matrix[k][i]] and c_ij == colors[idx_matrix[k][j]]:
                            e += 1
                    for k in range(i+1, j):
                        # Triangle (i, k, j)
                        if c_ij == colors[idx_matrix[i][k]] and c_ij == colors[idx_matrix[k][j]]:
                            e += 1
            return e

        # This O(n^3) calculation is too slow for initial if n=50.
        # 50^3 = 125,000. In Python, this is ~0.1s. Acceptable for init.
        # But we need incremental update for SA.
        
        # Incremental update for changing edge (u, v) from c_old to c_new
        # Triangles containing (u, v) are (u, v, w) for all w != u, v.
        # There are n-2 such triangles.
        # For each w, check if (u, v, w) was mono before and is mono after.
        
        current_energy = calc_energy()
        
        T = 1.0
        T_min = 0.01
        cooling = 0.9999 # Adjust based on steps
        
        import math
        
        steps = 0
        max_steps = 500000 # Safety
        
        while time.time() < start_time + max_time and steps < max_steps:
            if current_energy == 0:
                return colors[:]
                
            # Pick random edge
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            
            idx = idx_matrix[u][v]
            c_old = colors[idx]
            c_new = random.randint(0, 3)
            if c_new == c_old:
                continue
            
            # Calculate Delta E
            delta = 0
            for w in range(n):
                if w == u or w == v:
                    continue
                # Triangle (u, v, w)
                # Edges: (u,v), (u,w), (v,w)
                c_uw = colors[idx_matrix[u][w]]
                c_vw = colors[idx_matrix[v][w]]
                
                # Before
                if c_old == c_uw and c_old == c_vw:
                    delta -= 1 # Was mono, now not (since c_new != c_old)
                
                # After
                if c_new == c_uw and c_new == c_vw:
                    delta += 1 # Was not mono, now is
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                colors[idx] = c_new
                current_energy += delta
            
            T *= cooling
            steps += 1
            
        # Check if found
        if current_energy == 0:
            return colors[:]
        return None

    # Try to find solution for n=50
    # We can try multiple restarts if it fails
    for n in [50, 49, 48]:
        for attempt in range(5):
            if time.time() > deadline:
                break
            sol = solve_sa(n, max_time=10.0)
            if sol:
                return {'n': n, 'colors': sol}
        
    # Fallback to seed program logic (n=32)
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
    
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
            
    return {'n': 32, 'colors': colors}
