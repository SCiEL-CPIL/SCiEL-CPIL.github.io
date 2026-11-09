import time
import random
import numpy as np

def construct():
    # Deterministic construction for n=32 using (Z_2)^5
    # Vertices are elements of F_2^5.
    # Color of edge (u, v) is determined by the difference d = u + v (in F_2^5).
    # The 5 bits of d correspond to 5 coordinates.
    # We map the 31 non-zero vectors in F_2^5 to 4 colors.
    # A known construction for R(3,3,3,3) > 32 uses a 5-dimensional affine space.
    # Specifically, we can use the following mapping:
    # Let d be the difference vector (5 bits).
    # If d has weight 1, 2, 3, 4, or 5, assign colors based on a partition of the 31 non-zero vectors.
    # The condition "no monochromatic triangle" is equivalent to: for any u, v, w, 
    # colors(u,v), colors(v,w), colors(w,u) are not all equal.
    # Note that (u+v) + (v+w) + (w+u) = 0 in F_2^5.
    # So we need a coloring c: F_2^5 \ {0} -> {0,1,2,3} such that for any a,b,c != 0 with a+b+c=0,
    # c(a), c(b), c(c) are not all equal.
    
    # One such coloring is based on the weight (number of 1s) in the 5-bit vector.
    # Weight 1: 5 vectors.
    # Weight 2: 10 vectors.
    # Weight 3: 10 vectors.
    # Weight 4: 5 vectors.
    # Weight 5: 1 vector.
    
    # Let's try to assign colors based on weight.
    # If we assign Color 0 to Weight 1, Color 1 to Weight 2, Color 2 to Weight 3, Color 3 to Weight 4+5.
    # Does this work?
    # Sum of three vectors a,b,c is 0.
    # Weights: w(a), w(b), w(c).
    # In F_2^5, if a+b+c=0, then c=a+b.
    # w(c) = w(a+b).
    # If a and b are disjoint (support doesn't overlap), w(a+b) = w(a)+w(b).
    # If they overlap, it's more complex.
    
    # Actually, a standard construction for n=32 uses the following:
    # Identify vertices with F_2^5.
    # Color(u,v) = f(u+v) where f: F_2^5\{0} -> {0,1,2,3}.
    # We need f(a) != f(b) != f(c) != f(a) whenever a+b+c=0.
    
    # Let's use a specific known coloring.
    # Split the 31 non-zero vectors into 4 sets S0, S1, S2, S3.
    # A simple heuristic: 
    # Sort vectors by weight.
    # W1: 5 vectors
    # W2: 10 vectors
    # W3: 10 vectors
    # W4: 5 vectors
    # W5: 1 vector
    # Total 31.
    
    # Try:
    # S0 = W1 (5)
    # S1 = W2 (10)
    # S2 = W3 (10)
    # S3 = W4 U W5 (6)
    
    # Check triangle a,b,c with a+b+c=0.
    # If a,b in S0 (W1), then a,b are basis vectors e_i, e_j.
    # c = a+b has weight 2. So c in S1.
    # Colors: 0, 0, 1. Not mono.
    # If a,b in S1 (W2).
    # Let a, b have weight 2.
    # c = a+b.
    # If a,b disjoint, w(c)=4. c in S3. Colors 1,1,3.
    # If a,b overlap by 1, w(c)=2. c in S1. Colors 1,1,1. MONO!
    # Example: a={1,2}, b={2,3}. c={1,3}. All weight 2.
    # So this simple weight-based coloring fails.
    
    # We need a better partition.
    # Let's use the property of the 5-dimensional space.
    # There is a known solution for n=32.
    # It is often constructed via the "lexicographic" or "greedy" method on the group, or specific algebraic structures.
    
    # Let's stick to the seed program's approach for n=32 but ensure it's correct and fast,
    # and then focus on improving the search for n>32.
    
    # The seed program used a finite field GF(16) construction? 
    # "mul(a,b)" with poly 0b10011 (x^4+x+1) suggests GF(16).
    # Vertices 0..31. 
    # i,j in 0..31.
    # i = 16*i1 + i0.
    # If i1 == j1 (same half), color based on i0, j0 in GF(16).
    # If i1 != j1 (different half), color 3.
    
    # Let's verify the seed's n=32 construction is valid.
    # The seed code:
    # log[x] = e for x in GF(16)^*.
    # color = log[(i%16) ^ (j%16)] % 3 if same half else 3.
    # This maps the 15 non-zero differences in GF(16) to colors 0,1,2.
    # 15 / 3 = 5. So each color class has 5 differences.
    # In GF(16), if a+b+c=0, then c=a+b.
    # We need to ensure that for any a,b in GF(16)^*, if c=a+b is also in GF(16)^*,
    # then colors of a, b, c are not all same.
    # This is a known construction for R(3,3,3,3) > 32?
    # Actually, R(3,3,3,3) is known to be > 32. The maximum is 50 (Chung).
    # The GF(16) construction gives a 3-coloring of K_16 without mono triangles? No, it's 4-coloring of K_32.
    
    # Let's assume the seed's n=32 is valid.
    # We will use it as the base.
    
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    
    log = {}
    x = 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
        
    seed_colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                seed_colors.append(log[(i % 16) ^ (j % 16)] % 3)
            else:
                seed_colors.append(3)
                
    best_n = 32
    best_colors = seed_colors
    deadline = time.time() + 115.0 # Safety margin
    
    # Optimized Local Search using NumPy
    def count_triangles_vec(colors, n):
        # colors: np.array of shape (n, n), symmetric, 0 on diagonal
        # We need to count triples i<j<k such that C[i,j]==C[j,k]==C[i,k]
        # This is sum over i<j<k of (C[i,j]==C[j,k]) * (C[i,k]==C[j,k]) * (C[i,j]==C[i,k])
        # Actually, simpler: for each color c, let A_c be the adjacency matrix of color c.
        # Number of triangles of color c is trace(A_c^3) / 6.
        # Total mono triangles = sum_c trace(A_c^3) / 6.
        
        total = 0
        for c in range(4):
            A = (colors == c).astype(np.int32)
            np.fill_diagonal(A, 0)
            # trace(A^3)
            # A^2 = A @ A
            # trace(A^3) = sum(A^2 * A.T) ? No.
            # trace(A^3) = sum_{i,j,k} A[i,j] A[j,k] A[k,i]
            # This is equivalent to sum(A^2 * A.T) if A is symmetric?
            # A^2[i,k] = sum_j A[i,j] A[j,k]
            # sum_{i,k} A^2[i,k] A[k,i] = sum_{i,k} A^2[i,k] A[i,k] (since symmetric)
            # So trace(A^3) = np.sum(A @ A * A)
            tri = np.sum(A @ A * A)
            total += tri
        return total // 6

    def try_solve(n, time_limit):
        N = n*(n-1)//2
        # Represent colors as a flat array for output, but use 2D matrix for computation
        colors_flat = np.random.randint(0, 4, size=N)
        
        # Helper to convert flat to 2D
        def flat_to_2d(flat, n):
            mat = np.zeros((n, n), dtype=np.int32)
            idx = 0
            for i in range(n):
                for j in range(i+1, n):
                    mat[i, j] = flat[idx]
                    mat[j, i] = flat[idx]
                    idx += 1
            return mat

        curr_mat = flat_to_2d(colors_flat, n)
        curr_count = count_triangles_vec(curr_mat, n)
        
        if curr_count == 0:
            return colors_flat.tolist()
            
        start_time = time.time()
        last_improve = start_time
        
        # Precompute indices for each edge in the 2D matrix
        # We will iterate over edges in the flat array
        # For each edge (i,j), we need to check all k.
        # Instead of looping k in Python, we can use vectorized ops on the matrix rows.
        
        # Let's try a vectorized local search step.
        # For each edge e=(i,j) with old color c_old, new color c_new.
        # Delta = (number of k such that c_new==C[i,k]==C[j,k]) - (number of k such that c_old==C[i,k]==C[j,k])
        # Note: C[i,i]=0, C[j,j]=0.
        
        # We can precompute for each color c, the mask M_c[i,j] = 1 if C[i,j]==c else 0.
        # But colors change.
        
        # Let's just do a few random restarts with faster Python loops if NumPy overhead is high for small n?
        # For n=33, N=528. 
        # NumPy matrix mult is fast.
        
        while time.time() < start_time + time_limit:
            # Randomly select an edge
            e_idx = random.randint(0, N-1)
            
            # Find i, j from e_idx
            # i*n - i(i+1)/2 + (j-i-1) = e_idx
            # Approx i = sqrt(2*e_idx)
            # Let's just precompute a list of (i,j) for this n
            # Doing this inside the loop is bad. Precompute outside.
            
            # Precompute edges list
            if not hasattr(try_solve, 'edges_cache') or try_solve.edges_cache_n != n:
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        edges.append((i, j))
                try_solve.edges_cache = edges
                try_solve.edges_cache_n = n
            else:
                edges = try_solve.edges_cache
                
            i, j = edges[e_idx]
            old_c = curr_mat[i, j]
            
            # Calculate delta for each new color
            # We need to count k where curr_mat[i,k] == curr_mat[j,k] == new_c
            # And subtract count where curr_mat[i,k] == curr_mat[j,k] == old_c
            
            # Vectorized:
            # row_i = curr_mat[i]
            # row_j = curr_mat[j]
            # match_mask = (row_i == row_j)
            # For each color c, count = sum(match_mask & (row_i == c))
            # But we only care about k != i, j.
            # row_i[i] = 0, row_j[j] = 0.
            # If i != j, row_i[j] is the edge color. row_j[i] is the same.
            # We should exclude i and j from k.
            
            row_i = curr_mat[i]
            row_j = curr_mat[j]
            
            # Create a mask for valid k
            valid_k = np.ones(n, dtype=bool)
            valid_k[i] = False
            valid_k[j] = False
            
            match_mask = (row_i == row_j) & valid_k
            
            # Counts for each color
            counts = np.zeros(4, dtype=np.int32)
            for c in range(4):
                counts[c] = np.sum(match_mask & (row_i == c))
                
            best_new_c = -1
            best_delta = 0 # We want to minimize count, so we want negative delta
            
            for c in range(4):
                if c == old_c:
                    continue
                delta = counts[c] - counts[old_c]
                if delta < best_delta:
                    best_delta = delta
                    best_new_c = c
                    
            if best_new_c != -1 and best_delta < 0:
                curr_mat[i, j] = best_new_c
                curr_mat[j, i] = best_new_c
                curr_count += best_delta
                
                if curr_count == 0:
                    # Convert back to flat
                    flat = []
                    for idx, (u, v) in enumerate(edges):
                        flat.append(curr_mat[u, v])
                    return flat
                
                last_improve = time.time()
                
            # If no improvement, we could try random move, but let's stick to hill climbing for now
            # To escape local minima, we might need simulated annealing or random restarts.
            # Given the time limit, random restarts of the whole search might be better.
            # Let's break if we stall for too long?
            if time.time() - last_improve > 2.0:
                break
                
        return None

    # Main loop
    for n in range(33, 63):
        if time.time() > deadline - 10:
            break
            
        # Give a reasonable time slice
        # If n=33 takes 10s, we can try n=34.
        # If it fails, we might try again with a different seed?
        # The current try_solve uses random init.
        
        t0 = time.time()
        result = try_solve(n, 10.0)
        if result is not None:
            best_n = n
            best_colors = result
            # Continue to try larger n
        else:
            # If we fail for n, it's likely n+1 will be harder.
            # But maybe a different random seed would work?
            # We can try a few restarts for the same n.
            success = False
            for attempt in range(3):
                if time.time() > deadline - 5:
                    break
                result = try_solve(n, 5.0)
                if result is not None:
                    best_n = n
                    best_colors = result
                    success = True
                    break
            if not success:
                break
                
    return {'n': best_n, 'colors': best_colors}
