"""Matrix multiplication 2x3x7 - rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 7 matrix.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    The score is the rank (lower is better).
    """
    n, m, p = DIMENSIONS
    size_a = n * m  # 6
    size_b = m * p  # 21
    size_c = p * n  # 14
    rank_schoolbook = n * m * p  # 42

    # Build target tensor indices (sparse representation)
    # target[i*m+j, j*p+k, k*n+i] = 1
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c, 1.0))

    # Precompute: for each pair (a,b), which c values and values are needed
    # target[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # For exact match, we need the tensor to be exactly reproduced.
    
    # Strategy: Direct constructive search for low rank.
    # Key insight: The tensor is sparse with exactly 42 non-zero entries.
    # We search for the minimal R such that there exist U (6xR), V (21xR), W (14xR)
    # with entries in {0, ±0.5, ±1, ±1.5, ...} such that the tensor matches.
    
    # Start with known good constructions and try to reduce rank.
    # Schoolbook is rank 42. Can we do better?
    
    # Let's try a hybrid approach:
    # 1. Construct a valid decomposition at some rank R.
    # 2. Try to reduce R by finding linear dependencies or using algebraic identities.
    
    # First, let's establish a baseline with a structured decomposition.
    # For 2x3x7, we can think of it as:
    # C[k,i] = sum_j A[i,j] * B[j,k]
    
    # Let's try to find a rank-36 or lower decomposition using combinatorial search.
    
    # Actually, let's use a different approach: 
    # The tensor M_{2,3,7} can be decomposed using the fact that n=2 is small.
    # We can write the multiplication as a sum of rank-1 tensors.
    
    # Let's try a greedy algorithm to find a low-rank decomposition.
    # We'll start with the schoolbook decomposition and try to merge terms.
    
    # Schoolbook decomposition:
    U_school = np.zeros((size_a, rank_schoolbook), dtype=float)
    V_school = np.zeros((size_b, rank_schoolbook), dtype=float)
    W_school = np.zeros((size_c, rank_schoolbook), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    # Verify schoolbook
    computed = np.einsum('ar,br,cr->abc', U_school, V_school, W_school)
    target_tensor = np.zeros((size_a, size_b, size_c), dtype=float)
    for a, b, c, val in target_indices:
        target_tensor[a, b, c] = val
    
    assert np.allclose(computed, target_tensor, atol=1e-10)

    # Now try to reduce rank using a combinatorial approach.
    # We'll try ranks from 30 to 42 and see if we can find a valid decomposition.
    
    def verify_decomp(U, V, W, R):
        """Verify that U, V, W of rank R exactly reproduce the target tensor."""
        if U.shape != (size_a, R) or V.shape != (size_b, R) or W.shape != (size_c, R):
            return False
        computed = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(computed, target_tensor, atol=1e-10, rtol=1e-10)

    def try_rank_greedy(R, max_iters=5000, timeout=10.0):
        """
        Try to find a rank-R decomposition using greedy coordinate descent
        over half-integer values.
        """
        start = time.time()
        
        # Initialize with random half-integer values
        U = np.random.choice([0, 0.5, 1, -0.5, -1], size=(size_a, R))
        V = np.random.choice([0, 0.5, 1, -0.5, -1], size=(size_b, R))
        W = np.random.choice([0, 0.5, 1, -0.5, -1], size=(size_c, R))
        
        # Compute error
        def compute_error():
            computed = np.einsum('ar,br,cr->abc', U, V, W)
            return np.sum((computed - target_tensor)**2)
        
        best_err = compute_error()
        
        # Greedy coordinate descent: try changing one entry at a time
        values = [0, 0.5, 1, -0.5, -1, 1.5, -1.5]
        
        for it in range(max_iters):
            if time.time() - start > timeout:
                break
            
            # Try to improve by changing a random entry
            improved = False
            
            # Choose which matrix to modify
            mat_choice = np.random.randint(3)
            if mat_choice == 0:
                mat = U
                mat_name = 'U'
            elif mat_choice == 1:
                mat = V
                mat_name = 'V'
            else:
                mat = W
                mat_name = 'W'
            
            # Choose a random entry
            i, j = np.random.randint(mat.shape[0]), np.random.randint(mat.shape[1])
            old_val = mat[i, j]
            
            # Try different values
            best_new_val = old_val
            best_new_err = best_err
            
            for new_val in values:
                if new_val == old_val:
                    continue
                mat[i, j] = new_val
                err = compute_error()
                if err < best_new_err:
                    best_new_err = err
                    best_new_val = new_val
            
            mat[i, j] = best_new_val
            
            if best_new_err < best_err:
                best_err = best_new_err
                improved = True
            
            if best_err < 1e-10:
                break
        
        if best_err < 1e-10:
            return U, V, W
        return None

    # Try to find a low-rank decomposition
    best_result = None
    best_rank = rank_schoolbook
    
    # Start with schoolbook
    U_best = U_school.copy()
    V_best = V_school.copy()
    W_best = W_school.copy()
    
    # Try ranks from 35 down to 30
    for R in range(35, 29, -1):
        if time.time() > 100:
            break
        
        # Try multiple restarts
        found = False
        for restart in range(5):
            if time.time() > 100:
                break
            
            result = try_rank_greedy(R, max_iters=2000, timeout=5.0)
            if result is not None:
                U_r, V_r, W_r = result
                if verify_decomp(U_r, V_r, W_r, R):
                    U_best = U_r
                    V_best = V_r
                    W_best = W_r
                    best_rank = R
                    found = True
                    break
        
        if found:
            # Try to go even lower
            continue
        else:
            break
    
    # Return the best result
    return {
        'U': U_best.tolist(),
        'V': V_best.tolist(),
        'W': W_best.tolist()
    }
# EVOLVE-BLOCK-END
