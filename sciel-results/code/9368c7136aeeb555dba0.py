"""
Direct combinatorial search for rank-39 exact tensor decomposition of 2x3 x 3x7 matrix multiplication.
Uses randomized local search (simulated annealing-like) to minimize decomposition error.
"""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]
DEADLINE = 150.0
START_TIME = time.time()

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T has shape (n*m, m*p, p*n)
    # Indices: (i*m+j, j*p+k, k*n+i)
    # We want to decompose T into sum of R rank-1 tensors: U_r, V_r, W_r
    # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    
    # Initialize target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # We search for rank R < 42. Theoretical lower bound is 38? 
    # Known best for 2x3x3 is 7 (Strassen is 7, schoolbook 9).
    # For 2x3x7, schoolbook is 42.
    # Let's try to find rank 39 first, as it's a common target for improvements over schoolbook.
    # If 39 fails, we can try 40, 41, but 39 is the "interesting" one.
    # Actually, let's start with a slightly higher rank to ensure we find *something* better than 42, 
    # then try to push lower. But the score is rank, so lower is better.
    # Let's try rank 39 directly. If we can't find it, we return the best found (maybe 41 or 42).
    
    best_rank = n * m * p
    best_decomp = None
    
    # Candidate values for entries: integers and half-integers
    # In exact decompositions, entries are often small integers.
    # Let's restrict to {-1, -0.5, 0, 0.5, 1} initially for speed.
    # If that fails, expand to {-1, 0, 1} or {-2, -1, 0, 1, 2}.
    
    candidate_vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    # Search for rank 39
    target_rank = 39
    
    # To make search tractable, we fix the structure partially?
    # No, let's do a full random search with local optimization.
    # Dimension of search space is R * (nm + mp + pn) = 39 * (6 + 21 + 14) = 39 * 41 = 1599 variables.
    # This is huge for random search.
    
    # Alternative: Use the known structure of "Strassen-like" decompositions.
    # For 2x3x7, we can view it as blocks.
    # A is 2x3, B is 3x7.
    # Split B into B1 (3x3) and B2 (3x4)?
    # Or split A?
    
    # Let's try a specific structural approach that is different from pure random search:
    # We know that for 2x2x2, rank 7 works.
    # For 2x3x3, rank 7 works (Strassen for 2x2 part, schoolbook for rest? No, 2x3x3 is 7? 
    # Actually 2x3x3 min rank is 7. 3x3x3 is 23 (Strassen) or 27 (schoolbook).
    # 2x3x3: 2*3*3 = 18 schoolbook.
    # Is there a rank 7 decomposition for 2x3x3?
    # If we embed 2x2x2 (rank 7) into 2x3x3, we use 7 terms for the 2x2x2 block.
    # The remaining terms must handle the 3rd column of A and 3rd row of B?
    # This gets complicated.
    
    # Let's stick to the directive: "Apply a direct combinatorial search ... using a randomized local search".
    # To make it feasible within 120s, we must be smart.
    # We will use a "greedy" construction with local refinement.
    # Start with schoolbook (42).
    # Try to merge or eliminate terms?
    
    # Actually, a simpler structural change:
    # Use the fact that the tensor is sparse.
    # We will perform a random search over a subset of the space defined by "sparse" vectors.
    
    # Let's try rank 41 first? No, we want to beat 42.
    # Let's try to find a rank 41 decomposition easily?
    # If we can find 41, score is 41/42 = 0.976 (wait, score is seed/value, so 42/41 = 1.024).
    # Yes, any rank < 42 improves the score.
    # Rank 41 is easier to find than 39.
    
    # Strategy:
    # 1. Start with Schoolbook (42).
    # 2. Try to find a rank 41 decomposition by perturbing the schoolbook and projecting to nearest integer/half-integer solution with one term removed?
    # Or simply: Random search for rank 41.
    
    # Let's implement a random search for rank R.
    # We will try R=41, then R=40, then R=39.
    # For each R, we run a time-limited search.
    
    def evaluate_decomp(U, V, W):
        # Compute T_est = sum_r U[:,r] * V[:,r] * W[:,r]
        # Efficiently: T_est[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # This is a tensor contraction.
        # Using einsum: 'ar,br,cr->abc'
        # But R is small, dimensions are small.
        # T_est = np.einsum('ar,br,cr->abc', U, V, W)
        # Error = ||T - T_est||_F^2
        T_est = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.sum((T - T_est)**2)
    
    def random_init(R, vals):
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        # Initialize with random sparse vectors
        for r in range(R):
            # Each vector is sparse
            # Pick a few non-zero entries
            nz_u = random.randint(1, 3)
            nz_v = random.randint(1, 3)
            nz_w = random.randint(1, 3)
            
            idx_u = random.sample(range(n*m), nz_u)
            idx_v = random.sample(range(m*p), nz_v)
            idx_w = random.sample(range(p*n), nz_w)
            
            for idx in idx_u:
                U[idx, r] = random.choice(vals)
            for idx in idx_v:
                V[idx, r] = random.choice(vals)
            for idx in idx_w:
                W[idx, r] = random.choice(vals)
        return U, V, W
    
    def local_search(U, V, W, R, vals, time_limit=10.0):
        current_error = evaluate_decomp(U, V, W)
        if current_error < 1e-6:
            return U, V, W, current_error
            
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_error = current_error
        
        start = time.time()
        while time.time() - start < time_limit:
            # Pick a random variable to change
            mat = random.choice(['U', 'V', 'W'])
            if mat == 'U':
                idx = random.randint(0, n*m - 1)
                r = random.randint(0, R - 1)
                old_val = U[idx, r]
                new_val = random.choice(vals)
                if new_val == old_val:
                    continue
                U[idx, r] = new_val
                new_error = evaluate_decomp(U, V, W)
                if new_error < best_error:
                    best_error = new_error
                    best_U[idx, r] = new_val
                else:
                    # Revert
                    U[idx, r] = old_val
            elif mat == 'V':
                idx = random.randint(0, m*p - 1)
                r = random.randint(0, R - 1)
                old_val = V[idx, r]
                new_val = random.choice(vals)
                if new_val == old_val:
                    continue
                V[idx, r] = new_val
                new_error = evaluate_decomp(U, V, W)
                if new_error < best_error:
                    best_error = new_error
                    best_V[idx, r] = new_val
                else:
                    V[idx, r] = old_val
            else:
                idx = random.randint(0, p*n - 1)
                r = random.randint(0, R - 1)
                old_val = W[idx, r]
                new_val = random.choice(vals)
                if new_val == old_val:
                    continue
                W[idx, r] = new_val
                new_error = evaluate_decomp(U, V, W)
                if new_error < best_error:
                    best_error = new_error
                    best_W[idx, r] = new_val
                else:
                    W[idx, r] = old_val
                    
        return best_U, best_V, best_W, best_error
    
    # Search for ranks 41, 40, 39
    # We have ~140 seconds.
    # Allocate time per rank.
    ranks_to_try = [41, 40, 39]
    
    for R in ranks_to_try:
        if time.time() - START_TIME > DEADLINE - 5.0:
            break
            
        # Try a few restarts for this rank
        found = False
        for restart in range(5):
            if time.time() - START_TIME > DEADLINE - 5.0:
                break
                
            U, V, W = random_init(R, candidate_vals)
            U, V, W, err = local_search(U, V, W, R, candidate_vals, time_limit=10.0)
            
            if err < 1e-6:
                # Found exact decomposition
                best_rank = R
                best_decomp = (U, V, W)
                found = True
                break
        
        if found:
            # If we found 41, we can try 40 and 39 in remaining time
            # But if we found 39, we are done (can't go lower easily)
            if R == 39:
                break
                
    # If no better rank found, return schoolbook
    if best_decomp is None:
        # Fallback to schoolbook
        U = [[0.0] * (n*m*p) for _ in range(n*m)]
        V = [[0.0] * (n*m*p) for _ in range(m*p)]
        W = [[0.0] * (n*m*p) for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx][col] = 1.0
                    V[v_idx][col] = 1.0
                    W[w_idx][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    else:
        # Convert numpy arrays to lists of floats
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}

# Keep the global variables for compatibility if needed, though construct uses locals
DIMENSIONS = [2, 3, 7]
DEADLINE = 150.0
START_TIME = time.time()
