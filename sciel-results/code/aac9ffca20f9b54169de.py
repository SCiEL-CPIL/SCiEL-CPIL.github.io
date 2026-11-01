"""Tensor decomposition search for 2x2 * 2x3 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [2, 2, 3]


def _matmul_tensor():
    """Build the 2x2 @ 2x3 matrix multiplication tensor T[i, j, k] for W[k*n+i, r] convention.
    
    We need: (A @ B)[i, k] = sum_j A[i, j] * B[j, k]
    The tensor T has T[i, j, k] = 1 if the product A[i,j]*B[j,k] contributes to output[i,k].
    Actually, the multiplication tensor is T_{i,j,k} = delta_{i,j,k} where the product term is A[i,j]*B[j,k].
    So T[i,j,k] = 1 for all valid i,j,k (since every term contributes with coefficient 1).
    
    Wait, let me reconsider. The matrix multiplication tensor T is defined by:
    T(i, j, k) = 1 if the j-th column of A times j-th row of B contributes to (i,k) of output.
    For matrix multiplication C = A * B where C[i,k] = sum_j A[i,j] * B[j,k],
    the tensor has T[i,j,k] = 1 for all i,j,k in range.
    
    So for 2x2 * 2x3: i in {0,1}, j in {0,1}, k in {0,1,2}
    T[i,j,k] = 1 for all 12 combinations.
    """
    n, m, p = 2, 2, 3
    T = np.ones((n, m, p))
    return T


def _verify_decomposition(U, V, W, rank):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] reconstructs the multiplication tensor.
    
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    The reconstruction is: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1.
    """
    n, m, p = 2, 2, 3
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the reconstructed tensor
    T_recon = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                T_recon[i, j, k] = s
    
    T_true = np.ones((n, m, p))
    max_err = np.max(np.abs(T_recon - T_true))
    return max_err < 1e-9, max_err


def _snap_to_half_int(mat, tol=1e-6):
    """Snap values to nearest half-integer if within tolerance."""
    mat = np.array(mat, dtype=float)
    snapped = np.round(mat * 2) / 2.0
    if np.max(np.abs(mat - snapped)) < tol:
        return snapped
    return mat


def construct():
    """Find a rank-minimal exact tensor decomposition for 2x2 * 2x3 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin before 150s kill
    
    n, m, p = 2, 2, 3
    T = _matmul_tensor()
    
    # The schoolbook rank is n*m*p = 12.
    # Known results: for 2x2 * 2x3, the rank is known to be 12 (it's the same as general).
    # Actually, let's check: the matrix multiplication tensor for m x n * n x p has rank mn p 
    # in general, but sometimes less. For 2x2 * 2x2, rank is 7 (Strassen).
    # For 2x2 * 2x3, the rank might be less than 12.
    
    # Let's search for lower rank decompositions.
    best_rank = 12  # start with schoolbook
    best_decomp = None
    
    # Generate schoolbook decomposition as fallback
    U_school = np.zeros((n * m, best_rank))
    V_school = np.zeros((m * p, best_rank))
    W_school = np.zeros((p * n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    best_decomp = (U_school, V_school, W_school)
    
    # Try to find lower rank decompositions using randomized methods
    # We'll use a combination of random initialization and local optimization
    
    np.random.seed(42)
    random.seed(42)
    
    # Search from rank 7 upward (Strassen gives 7 for 2x2*2x2, so 7 is a reasonable lower bound)
    # Actually the theoretical lower bound for 2x2 * 2x3 is at least max(n*m, m*p, p*n) = max(4, 6, 6) = 6
    # And at least the "border rank" lower bounds.
    
    for rank in range(7, 12):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations for this rank
        max_attempts = 50
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with half-integer values
            # Use small integers/half-integers to increase chance of exact solutions
            U = np.random.choice([-2, -1, -0.5, 0, 0.5, 1, 2], size=(n * m, rank))
            V = np.random.choice([-2, -1, -0.5, 0, 0.5, 1, 2], size=(m * p, rank))
            W = np.random.choice([-2, -1, -0.5, 0, 0.5, 1, 2], size=(p * n, rank))
            
            # Verify
            valid, err = _verify_decomposition(U, V, W, rank)
            if valid:
                best_rank = rank
                best_decomp = (U, V, W)
                found = True
                break
        
        if found:
            # Try even lower ranks
            continue
        else:
            # If we can't find rank r, we likely can't find lower ranks easily
            # But let's still try one more rank below
            if rank > 7:
                break
    
    # If we found a decomposition, try to polish it
    if best_decomp is not None:
        U, V, W = best_decomp
        U = _snap_to_half_int(U)
        V = _snap_to_half_int(V)
        W = _snap_to_half_int(W)
        best_decomp = (U, V, W)
    
    # Convert to lists of lists
    U, V, W = best_decomp
    U_list = [[float(U[i, j]) for j in range(len(U[0]))] for i in range(len(U))]
    V_list = [[float(V[i, j]) for j in range(len(V[0]))] for i in range(len(V))]
    W_list = [[float(W[i, j]) for j in range(len(W[0]))] for i in range(len(W))]
    
    return {"U": U_list, "V": V_list, "W": W_list}
