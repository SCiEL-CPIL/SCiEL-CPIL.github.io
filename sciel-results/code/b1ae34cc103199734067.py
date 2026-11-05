import time
import random
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = t0 + 115.0  # Safety margin

    # --- Helper: GF(16) multiplication for the Clebsch graph ---
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

    def build_seed_32():
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return colors

    def solve(N, seed_colors=None):
        if time.time() > DEADLINE - 2:
            return None

        # Initialize state
        if seed_colors and len(seed_colors) == N * (N - 1) // 2:
            colors = np.array(seed_colors, dtype=np.int8).copy()
        else:
            # Random initialization
            colors = np.random.randint(0, 4, size=(N * (N - 1) // 2), dtype=np.int8)

        # Build M matrices (4 NxN matrices)
        M = [np.zeros((N, N), dtype=np.int8) for _ in range(4)]
        # Fill M from colors
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                c = colors[idx]
                M[c][i][j] = 1
                M[c][j][i] = 1
                idx += 1

        def compute_m2(M):
            return [M[c] @ M[c] for c in range(4)]

        M2 = compute_m2(M)

        def get_bad_edges():
            bad = []
            # Vectorized check for bad edges
            # For each color c, find indices (i,j) where M2[c][i,j] > 0
            # But we only care if the edge (i,j) is actually color c.
            # Since M[c][i,j] is 1 iff edge is color c, and M2[c][i,j] > 0 implies triangle,
            # we can just check M2[c][i,j] > 0 for all i<j.
            # Wait, if M2[c][i,j] > 0, it means there is a triangle of color c.
            # This implies M[c][i,j] must be 1?
            # Yes, because M2[c][i,j] = sum_k M[c][i,k]M[c][k,j].
            # If this sum is > 0, then M[c][i,k]=1 and M[c][k,j]=1 for some k.
            # Does it imply M[c][i,j]=1?
            # Not necessarily. M[c][i,j] could be 0.
            # But a "bad edge" is an edge (i,j) of color c that is in a triangle.
            # So we need M[c][i,j]==1 AND M2[c][i,j]>0.
            # However, if M[c][i,j]==0, it's not an edge of color c, so it can't be a "bad edge" of color c.
            # But it could be a bad edge of another color? No, an edge has only one color.
            # So we iterate over all edges (i,j) and check if M2[colors[i,j]][i,j] > 0.
            
            # Vectorized approach:
            # Create a mask for each color?
            # Or just loop. N=50, 1225 edges. Loop is fast enough.
            for i in range(N):
                for j in range(i + 1, N):
                    # Find index in colors array
                    # idx = i * N - i*(i+1)//2 + (j - i - 1)
                    # But we have M matrices, so we can just check M2
                    # We need to know the color of (i,j).
                    # We can reconstruct it or keep a colors matrix.
                    # Let's keep a colors matrix for speed.
                    pass
            
            # Actually, let's maintain a 2D colors matrix `C` for O(1) lookup.
            # I didn't define C in the outer scope yet. Let's do that.
            return None # Placeholder

        # Let's refactor to maintain C (NxN) as well.
        
        C = np.zeros((N, N), dtype=np.int8)
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                c = colors[idx]
                C[i][j] = c
                C[j][i] = c
                idx += 1

        def get_bad_edges():
            bad = []
            for i in range(N):
                for j in range(i + 1, N):
                    c = C[i][j]
                    if M2[c][i][j] > 0:
                        bad.append((i, j, c))
            return bad

        bad_edges = get_bad_edges()
        
        # If seed is already valid, return immediately
        if not bad_edges:
            return colors.tolist()

        current_T = 10.0
        cooling = 0.995
        min_T = 0.01
        
        # Local Search
        while time.time() < DEADLINE - 5:
            if not bad_edges:
                return colors.tolist()
            
            # Pick a random bad edge
            u, v, c_old = random.choice(bad_edges)
            
            best_delta = float('inf')
            best_c = -1
            
            for c_new in range(4):
                if c_new == c_old:
                    continue
                k_old = M2[c_old][u][v]
                k_new = M2[c_new][u][v]
                delta = k_new - k_old
                if delta < best_delta:
                    best_delta = delta
                    best_c = c_new
            
            # Acceptance
            accept = False
            if best_delta < 0:
                accept = True
            else:
                p = np.exp(-best_delta / current_T)
                if random.random() < p:
                    accept = True
            
            if accept:
                # Update C
                C[u][v] = best_c
                C[v][u] = best_c
                # Update colors array (flattened)
                # Find index of (u,v) in flattened array
                # idx = u * N - u*(u+1)//2 + (v - u - 1)
                idx_u_v = u * N - u * (u + 1) // 2 + (v - u - 1)
                colors[idx_u_v] = best_c
                
                # Update M
                M[c_old][u][v] = 0
                M[c_old][v][u] = 0
                M[best_c][u][v] = 1
                M[best_c][v][u] = 1
                
                # Update M2
                M2[c_old] = M[c_old] @ M[c_old]
                M2[best_c] = M[best_c] @ M[best_c]
                
                # Update bad_edges
                bad_edges = get_bad_edges()
            
            current_T *= cooling
            if current_T < min_T:
                current_T = min_T

        return None

    # Strategy:
    # 1. Try to solve N=50 starting from random init.
    # 2. If fails, try N=50 starting from seed 32 extended randomly? 
    #    Actually, seed 32 is fixed. Extending it to 50 requires adding 18 vertices.
    #    Random init for N=50 is likely better for SA.
    # 3. If N=50 fails, try N=48, 46, etc.
    
    # We will use the seed 32 construction as a potential starting point for N=32, 
    # but since we are optimizing N, we should jump to 50.
    # The seed 32 is valid. If we can't find 50, we should return 32 at least.
    
    # Let's try 50 first.
    # To save time, we can run multiple restarts for N=50.
    # But `solve` already runs until deadline.
    # We should split the time budget.
    
    time_budget_per_attempt = 20.0
    attempts = 0
    max_attempts = 5 # 5 * 20s = 100s
    
    best_result = None
    
    # Try N=50
    for i in range(max_attempts):
        if time.time() > DEADLINE - 10:
            break
        # Use a slice of the deadline? No, `solve` checks global DEADLINE.
        # We need to modify `solve` to accept a local deadline or just rely on global.
        # If we call `solve` multiple times, it will use the remaining time.
        # That's fine.
        
        # Random seed for diversity
        random.seed()
        
        res = solve(50)
        if res:
            return {'n': 50, 'colors': res}
        
        # If we have time, try again.
        # `solve` returns None if it hits the deadline or exhausts its internal loop?
        # My `solve` loops `while time.time() < DEADLINE - 5`.
        # So it will use ALL remaining time.
        # This means we can only call `solve` ONCE effectively, or we need to limit its time.
        
        # Let's modify `solve` to accept a `time_limit` parameter.
        
    # Okay, let's rewrite `solve` to accept a time limit.
    
    def solve_limited(N, time_limit, seed_colors=None):
        t_start = time.time()
        t_end = t_start + time_limit
        
        if seed_colors and len(seed_colors) == N * (N - 1) // 2:
            colors = np.array(seed_colors, dtype=np.int8).copy()
        else:
            colors = np.random.randint(0, 4, size=(N * (N - 1) // 2), dtype=np.int8)

        M = [np.zeros((N, N), dtype=np.int8) for _ in range(4)]
        C = np.zeros((N, N), dtype=np.int8)
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                c = colors[idx]
                M[c][i][j] = 1
                M[c][j][i] = 1
                C[i][j] = c
                C[j][i] = c
                idx += 1

        def compute_m2(M):
            return [M[c] @ M[c] for c in range(4)]

        M2 = compute_m2(M)

        def get_bad_edges():
            bad = []
            for i in range(N):
                for j in range(i + 1, N):
                    c = C[i][j]
                    if M2[c][i][j] > 0:
                        bad.append((i, j, c))
            return bad

        bad_edges = get_bad_edges()
        
        if not bad_edges:
            return colors.tolist()

        current_T = 10.0
        cooling = 0.995
        min_T = 0.01
        
        while time.time() < t_end:
            if not bad_edges:
                return colors.tolist()
            
            u, v, c_old = random.choice(bad_edges)
            
            best_delta = float('inf')
            best_c = -1
            
            for c_new in range(4):
                if c_new == c_old:
                    continue
                k_old = M2[c_old][u][v]
                k_new = M2[c_new][u][v]
                delta = k_new - k_old
                if delta < best_delta:
                    best_delta = delta
                    best_c = c_new
            
            accept = False
            if best_delta < 0:
                accept = True
            else:
                p = np.exp(-best_delta / current_T)
                if random.random() < p:
                    accept = True
            
            if accept:
                C[u][v] = best_c
                C[v][u] = best_c
                idx_u_v = u * N - u * (u + 1) // 2 + (v - u - 1)
                colors[idx_u_v] = best_c
                
                M[c_old][u][v] = 0
                M[c_old][v][u] = 0
                M[best_c][u][v] = 1
                M[best_c][v][u] = 1
                
                M2[c_old] = M[c_old] @ M[c_old]
                M2[best_c] = M[best_c] @ M[best_c]
                
                bad_edges = get_bad_edges()
            
            current_T *= cooling
            if current_T < min_T:
                current_T = min_T

        return None

    # Plan:
    # 1. Try N=50 with 5 restarts of 20s each.
    # 2. If fail, try N=48 with 5 restarts of 10s each?
    # 3. If fail, return Seed 32.
    
    seed_32 = build_seed_32()
    
    # Try N=50
    for i in range(5):
        if time.time() > DEADLINE - 10:
            break
        res = solve_limited(50, 20.0)
        if res:
            return {'n': 50, 'colors': res}
            
    # Try N=48
    for i in range(3):
        if time.time() > DEADLINE - 10:
            break
        res = solve_limited(48, 15.0)
        if res:
            return {'n': 48, 'colors': res}
            
    # Try N=46
    for i in range(2):
        if time.time() > DEADLINE - 10:
            break
        res = solve_limited(46, 10.0)
        if res:
            return {'n': 46, 'colors': res}
            
    # Fallback to Seed 32
    return {'n': 32, 'colors': seed_32}
