import time
import numpy as np

DIMENSIONS = [2, 2, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # First, try to find rank-20 decomposition
    rank_20 = try_find_rank_20(deadline)
    if rank_20 is not None:
        return rank_20
    
    # Fallback to rank-21 Strassen-based solution
    rank = 21
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # u_coefs[t][i][j]
    u_coefs = [
        { (0,0): 1, (1,1): 1 }, # M1
        { (1,0): 1, (1,1): 1 }, # M2
        { (0,0): 1 },           # M3
        { (1,1): 1 },           # M4
        { (0,0): 1, (0,1): 1 }, # M5
        { (1,0): 1, (0,0): -1 },# M6
        { (0,1): 1, (1,1): -1 } # M7
    ]
    
    # v_coefs[t][j][k_local] where k_local is 0 or 1
    v_coefs = [
        { (0,0): 1, (1,1): 1 }, # M1
        { (0,0): 1 },           # M2
        { (0,1): 1, (1,1): -1 },# M3
        { (1,0): 1, (0,0): -1 },# M4
        { (1,1): 1 },           # M5
        { (0,0): 1, (0,1): 1 }, # M6
        { (1,0): 1, (1,1): 1 }  # M7
    ]
    
    # w_coefs[t][i][k_local]
    w_coefs = [
        { (0,0): 1, (1,1): 1 }, # M1 -> C00, C11
        { (1,0): 1, (1,1): -1 },# M2 -> C10, -C11
        { (0,1): 1, (1,1): 1 }, # M3 -> C01, C11
        { (0,0): 1, (1,0): 1 }, # M4 -> C00, C10
        { (0,0): -1, (0,1): 1 },# M5 -> -C00, C01
        { (1,1): 1 },           # M6 -> C11
        { (0,0): 1 }            # M7 -> C00
    ]
    
    for s in range(3):
        base_col = 7 * s
        for t in range(7):
            col = base_col + t
            
            # Fill U
            for (i, j), val in u_coefs[t].items():
                row_u = i * m + j
                U[row_u][col] += val
                
            # Fill V
            for (j, k_loc), val in v_coefs[t].items():
                k_global = 2 * s + k_loc
                row_v = j * p + k_global
                V[row_v][col] += val
                
            # Fill W
            for (i, k_loc), val in w_coefs[t].items():
                k_global = 2 * s + k_loc
                row_w = k_global * n + i
                W[row_w][col] += val
                
    return {"U": U, "V": V, "W": W}

def verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    r = U.shape[1]
    
    # The tensor T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
    # should equal 1 if the multiplication is correctly represented
    # For matrix multiplication C = A*B: C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] should be 1 for all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for t in range(r):
                    val += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def try_find_rank_20(deadline):
    """Try to find a rank-20 decomposition using algebraic search."""
    n, m, p = DIMENSIONS
    r = 20
    
    # We need to find U (n*m x r), V (m*p x r), W (p*n x r)
    # such that for all i,j,k: sum_t U[i,j,t]*V[j,k,t]*W[k,i,t] = 1
    
    # This is a system of polynomial equations. We'll use a randomized approach:
    # 1. Randomly initialize some variables
    # 2. Use the bilinear structure to solve for others
    
    # Strategy: Use the fact that if we fix W, the problem becomes bilinear in U and V
    # For each (i,k), we need: sum_j A[i,j]*B[j,k] = C[i,k]
    # This means: for each t, the outer product U[:,t] @ V[:,t] contributes to the tensor
    
    # Let's use a different approach: exploit the structure
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i,j,k] = 1 (the multiplication tensor)
    
    # We can write: T = sum_t u_t ⊗ v_t ⊗ w_t
    # where u_t is a vector in R^{n*m}, v_t in R^{m*p}, w_t in R^{p*n}
    
    # For a 2x2x6 problem:
    # U is 4x20, V is 12x20, W is 12x20
    # Total unknowns: 4*20 + 12*20 + 12*20 = 80 + 240 + 240 = 560
    # Constraints: 2*2*6 = 24 equations (one for each (i,j,k))
    
    # This is underdetermined, so solutions should exist if rank 20 is achievable.
    
    # Approach: Use a structured initialization based on Strassen and try to merge
    # two of the 21 terms.
    
    # Alternative: Use randomized search with gradient descent on the polynomial system
    
    # Let's try a more direct approach: 
    # Fix random values for W, then solve for U and V using the bilinear constraints.
    
    # For each (i,k): sum_j sum_t U[i,j,t]*V[j,k,t]*W[k,i,t] = 1
    # Let w_ki = W[k*n+i, t] for fixed (k,i)
    # Then: sum_j (sum_t U[i,j,t]*V[j,k,t]*w_ki,t) = 1
    
    # This is still complex. Let's use a simpler randomized approach.
    
    # Generate random half-integer or integer matrices and check
    np.random.seed(42)
    
    # Try several random initializations
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Initialize with small random values
        U = np.random.randint(-2, 3, size=(n*m, r)) * 0.5
        V = np.random.randint(-2, 3, size=(m*p, r)) * 0.5
        W = np.random.randint(-2, 3, size=(p*n, r)) * 0.5
        
        # Verify
        if verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try a structured approach: start from Strassen rank-21 and try to eliminate one term
    # by expressing it as a combination of others
    
    # The Strassen solution has 3 blocks of 7 terms. 
    # If we can merge two terms across blocks, we get rank 20.
    
    # Let's try: Can we express one of the 21 Strassen terms as a linear combination
    # of the other 20 in the tensor product sense?
    
    # This is equivalent to checking if the 21 vectors (in the tensor space) are linearly dependent.
    
    # Build the tensor space: dimension n*m*p = 24
    # Each rank-1 term u_t ⊗ v_t ⊗ w_t gives a vector in R^24
    
    # Get the Strassen rank-21 solution
    U21, V21, W21 = get_rank_21_solution()
    
    # Build the 21 tensor vectors
    tensors = []
    for t in range(21):
        vec = np.zeros(n*m*p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[i*m*p + j*p + k] = U21[i*m+j][t] * V21[j*p+k][t] * W21[k*n+i][t]
        tensors.append(vec)
    
    tensors = np.array(tensors)
    
    # Check if these 21 vectors in R^24 are linearly dependent
    # If yes, we might be able to find a rank-20 solution
    rank = np.linalg.matrix_rank(tensors, tol=1e-10)
    
    if rank < 21:
        # There's linear dependence! Find a rank-20 solution
        # The null space of tensors gives coefficients c such that sum_t c_t * tensor_t = 0
        # We can use this to eliminate one term
        
        # Find null space
        from scipy.linalg import null_space
        null_vec = null_space(tensors, rcond=1e-10)
        
        if null_vec.shape[1] > 0:
            c = null_vec[:, 0]
            
            # Find the term with largest |c_t| to eliminate
            max_idx = np.argmax(np.abs(c))
            c_max = c[max_idx]
            
            # New solution: scale other terms and remove term max_idx
            # sum_{t != max_idx} (c_t / c_max) * tensor_t = tensor_max_idx
            # So we can express tensor_max_idx in terms of others
            
            # But this doesn't directly give us U, V, W for rank 20
            # We need to be more careful about the structure
            
            # For now, if we detect linear dependence, it suggests rank 20 might be possible
            # but constructing the explicit solution requires more work
    
    return None

def get_rank_21_solution():
    """Return the rank-21 Strassen-based solution."""
    n, m, p = DIMENSIONS
    rank = 21
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    u_coefs = [
        { (0,0): 1, (1,1): 1 },
        { (1,0): 1, (1,1): 1 },
        { (0,0): 1 },
        { (1,1): 1 },
        { (0,0): 1, (0,1): 1 },
        { (1,0): 1, (0,0): -1 },
        { (0,1): 1, (1,1): -1 }
    ]
    
    v_coefs = [
        { (0,0): 1, (1,1): 1 },
        { (0,0): 1 },
        { (0,1): 1, (1,1): -1 },
        { (1,0): 1, (0,0): -1 },
        { (1,1): 1 },
        { (0,0): 1, (0,1): 1 },
        { (1,0): 1, (1,1): 1 }
    ]
    
    w_coefs = [
        { (0,0): 1, (1,1): 1 },
        { (1,0): 1, (1,1): -1 },
        { (0,1): 1, (1,1): 1 },
        { (0,0): 1, (1,0): 1 },
        { (0,0): -1, (0,1): 1 },
        { (1,1): 1 },
        { (0,0): 1 }
    ]
    
    for s in range(3):
        base_col = 7 * s
        for t in range(7):
            col = base_col + t
            
            for (i, j), val in u_coefs[t].items():
                row_u = i * m + j
                U[row_u][col] += val
                
            for (j, k_loc), val in v_coefs[t].items():
                k_global = 2 * s + k_loc
                row_v = j * p + k_global
                V[row_v][col] += val
                
            for (i, k_loc), val in w_coefs[t].items():
                k_global = 2 * s + k_loc
                row_w = k_global * n + i
                W[row_w][col] += val
    
    return U, V, W
