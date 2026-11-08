"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 5 matrix 
    by a 5 by 6 matrix.
    
    The schoolbook algorithm has rank n*m*p = 2*5*6 = 60.
    We attempt to find a lower rank decomposition using HOOI (Higher-Order Orthogonal 
    Iteration) on the matrix multiplication tensor.
    
    However, for small matrices, it is known that the bilinear complexity of 2x5 by 5x6 
    matrix multiplication is likely 60 (no known improvement). But let's verify if 
    HOOI can find a lower rank approximation that is exact.
    
    Actually, for integer/rational entries, exact rank minimization is hard. 
    Let's use a structured approach:
    
    1. Construct the full matrix multiplication tensor T of shape (n*m, m*p, p*n).
       T[i*m+j, j*p+k, k*n+i] = 1, else 0.
       
    2. Use HOOI to find a CP decomposition of rank r for decreasing r until we find 
       one that exactly reproduces the tensor (within tight tolerance).
       
    3. If no lower rank is found, return the schoolbook decomposition.
    
    Note: The output factor W follows the official trace convention W[k*n+i, r].
    This matches our tensor construction.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # Leave 10 seconds margin
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_schoolbook = np.zeros((n * m, rank_schoolbook))
    V_schoolbook = np.zeros((m * p, rank_schoolbook))
    W_schoolbook = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j, col] = 1.0
                V_schoolbook[j * p + k, col] = 1.0
                W_schoolbook[k * n + i, col] = 1.0
                col += 1
    
    # Construct the full matrix multiplication tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Try to find a lower rank decomposition using HOOI
    # We'll try ranks from 1 up to rank_schoolbook - 1
    best_U, best_V, best_W, best_rank = U_schoolbook, V_schoolbook, W_schoolbook, rank_schoolbook
    
    # Try HOOI for various ranks
    max_rank_to_try = min(rank_schoolbook - 1, 50)  # Limit search for efficiency
    
    for r in range(1, max_rank_to_try + 1):
        if time.time() > deadline:
            break
            
        # Initialize with random factors
        np.random.seed(42 + r)  # Deterministic for reproducibility
        U_r = np.random.randn(n * m, r)
        V_r = np.random.randn(m * p, r)
        W_r = np.random.randn(p * n, r)
        
        # HOOI iterations
        converged = False
        for it in range(100):
            if time.time() > deadline:
                break
                
            # Update U
            T_matted = np.tensordot(T, np.outer(V_r, W_r), axes=([1, 2], [0, 0]))
            # T_matted shape: (n*m, r)
            U_r_new, _, _ = np.linalg.svd(T_matted, full_matrices=False)
            
            # Update V
            T_matted = np.tensordot(T, np.outer(W_r, U_r_new), axes=([2, 0], [0, 0]))
            # T_matted shape: (m*p, r)
            V_r_new, _, _ = np.linalg.svd(T_matted, full_matrices=False)
            
            # Update W
            T_matted = np.tensordot(T, np.outer(U_r_new, V_r_new), axes=([0, 1], [0, 0]))
            # T_matted shape: (p*n, r)
            W_r_new, _, _ = np.linalg.svd(T_matted, full_matrices=False)
            
            # Check convergence
            diff_U = np.linalg.norm(U_r_new - U_r)
            diff_V = np.linalg.norm(V_r_new - V_r)
            diff_W = np.linalg.norm(W_r_new - W_r)
            U_r, V_r, W_r = U_r_new, V_r_new, W_r_new
            
            if diff_U + diff_V + diff_W < 1e-8:
                converged = True
                break
        
        if not converged:
            continue
            
        # Check if the decomposition exactly reproduces the tensor
        # Reconstruct T from the factors
        T_reconstructed = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
        error = np.linalg.norm(T_reconstructed - T)
        
        if error < 1e-10:
            best_U, best_V, best_W, best_rank = U_r, V_r, W_r, r
            break  # Found a lower rank, stop searching
    
    # If we found a better rank, use it; otherwise use schoolbook
    if best_rank < rank_schoolbook:
        U_final, V_final, W_final = best_U, best_V, best_W
    else:
        U_final, V_final, W_final = U_schoolbook, V_schoolbook, W_schoolbook
    
    # Convert to lists as required by the interface
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }
# EVOLVE-BLOCK-END
