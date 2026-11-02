import time
import sys
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for matrix multiplication.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    The rank is the number of columns in these matrices.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    n, m, p = DIMENSIONS
    
    # First, build the schoolbook decomposition as a fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = rank_school
    
    # Now try to find a lower rank using block-circulant embedding
    # Strategy: Use Strassen-like techniques for smaller sub-problems
    # and combine them with the remaining parts
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # For 3x5x7, we can try to decompose using known optimal algorithms
    # for sub-blocks. Let's try a structured approach:
    # Split the 5x7 part into blocks and use bilinear algorithms
    
    # Approach: Use the fact that we can sometimes achieve better than n*m*p
    # by using algebraic structure. For now, let's try a random search
    # with half-integer entries for a limited number of iterations.
    
    def verify_decomposition(U, V, W, n, m, p, max_check=100):
        """Verify that the decomposition correctly computes matrix multiplication."""
        # Check a few random matrices
        rng = np.random.RandomState(42)
        for _ in range(min(max_check, 20)):
            A = rng.randint(-5, 6, size=(n, m)).astype(float)
            B = rng.randint(-5, 6, size=(m, p)).astype(float)
            
            # Compute A*B directly
            AB_direct = A @ B
            
            # Compute using decomposition: sum_r U[:,r] * V[:,r] * W[:,r]
            # where U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] contributes to (i,k)
            AB_decomp = np.zeros((n, p), dtype=float)
            U_arr = np.array(U)
            V_arr = np.array(V)
            W_arr = np.array(W)
            
            for r in range(len(U[0])):
                # Extract vectors
                u_vec = U_arr[:, r]  # length n*m
                v_vec = V_arr[:, r]  # length m*p
                w_vec = W_arr[:, r]  # length p*n
                
                # Reshape
                u_mat = u_vec.reshape(n, m)
                v_mat = v_vec.reshape(m, p)
                w_mat = w_vec.reshape(p, n)
                
                # The contribution is: sum over j of u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
                # This is equivalent to: for each i,k, sum_j u[i,j]*v[j,k]*w[k,i]
                # Let's compute this efficiently
                for i in range(n):
                    for k in range(p):
                        s = 0.0
                        for j in range(m):
                            s += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
                        AB_decomp[i, k] += s
            
            if not np.allclose(AB_direct, AB_decomp, atol=1e-9):
                return False
        return True
    
    # Try random search for lower rank
    # Start with rank = 80 and try to find a valid decomposition
    # Use half-integer entries (multiply by 2 to work with integers)
    
    rng = np.random.RandomState(12345)
    search_ranks = [80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10]
    
    for target_rank in search_ranks:
        if time.time() > deadline:
            break
        
        # Try a few random attempts for this rank
        attempts = 5
        for attempt in range(attempts):
            if time.time() > deadline:
                break
            
            # Generate random half-integer entries
            # Use integers 0, 1, 2, 3 and divide by 2 at the end
            U_rand = rng.randint(0, 4, size=(n * m, target_rank)).astype(float) / 2.0
            V_rand = rng.randint(0, 4, size=(m * p, target_rank)).astype(float) / 2.0
            W_rand = rng.randint(0, 4, size=(p * n, target_rank)).astype(float) / 2.0
            
            # Verify
            if verify_decomposition(U_rand.tolist(), V_rand.tolist(), W_rand.tolist(), n, m, p, max_check=5):
                best_U = U_rand.tolist()
                best_V = V_rand.tolist()
                best_W = W_rand.tolist()
                best_rank = target_rank
                break
        
        if best_rank < rank_school:
            break
    
    # If we haven't found something better, try a more structured approach
    # Use the fact that for certain dimensions, known algorithms exist
    
    if best_rank == rank_school and time.time() < deadline - 10:
        # Try to use block structure
        # For 3x5x7, we can think of it as 3 blocks of 1x5x7
        # Or 5 blocks of 3x1x7
        # Let's try to use a bilinear algorithm for 1x5x7 or 3x1x7
        
        # Actually, let's try a different approach: use the fact that
        # we can sometimes share computations across blocks
        
        # For now, keep the schoolbook as it's guaranteed correct
        pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
