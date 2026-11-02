"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x5 matrix."""
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 30
    deadline = time.time() + 110  # safety margin before 120s limit

    # The tensor T_{i,j,k} = 1 for all i,j,k (matrix multiplication tensor).
    # We need to find factor matrices U (n*m x r), V (m*p x r), W (p*n x r)
    # such that for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Try various ranks from known lower bounds up to schoolbook
    # For 2x3x5, the schoolbook rank is 30.
    # Let's try to find a better decomposition.
    
    # One approach: use randomized initialization and gradient-free optimization
    # to minimize the error, then check if it's exactly zero.
    
    # First, let's try a simple approach: use the schoolbook decomposition as baseline
    best_rank = rank_schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Generate schoolbook decomposition
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    # Convert to list of lists for return
    def mat_to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    # Try to find lower rank decompositions
    # Strategy: try ranks from lower to upper, use random restarts
    
    for r in range(20, rank_schoolbook + 1):
        if time.time() > deadline:
            break
        if r >= best_rank:
            break
            
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            # Random initialization with small entries
            np.random.seed(42 + r * 100 + attempt)
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            # Compute error: for each (i,j,k), check if sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
            # This is equivalent to checking if the tensor T = sum_c U_c ⊗ V_c ⊗ W_c equals the all-ones tensor
            
            # Flatten the computation
            error = 0.0
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                        err = abs(val - 1.0)
                        error += err
                        if err > max_err:
                            max_err = err
            
            if max_err < 1e-10:
                # Found exact decomposition!
                best_rank = r
                best_U = U
                best_V = V
                best_W = W
                break
        
        if best_rank == r:
            break
    
    # If we didn't find anything better, use schoolbook
    if best_U is None:
        best_U = U_school
        best_V = V_school
        best_W = W_school
        best_rank = rank_schoolbook
    
    return {"U": mat_to_list(best_U), "V": mat_to_list(best_V), "W": mat_to_list(best_W)}
# EVOLVE-BLOCK-END
