"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 2 matrix
    by a 2 by 8 matrix.
    
    Returns:
        dict: {'U': list of lists, 'V': list of lists, 'W': list of lists}
              U shape (n*m, rank)
              V shape (m*p, rank)
              W shape (p*n, rank)
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Known lower bound: 26 (from literature)
    # Known upper bound: 28 (Strassen per 2x2 block)
    # We'll try to find a rank 26 or 27 decomposition if possible,
    # otherwise fall back to rank 28.
    
    # First, generate the rank 28 Strassen solution as a baseline
    best_solution = _strassen_28(n, m, p)
    best_rank = 28
    
    # Try to improve using ALS optimization
    # We'll attempt to find a lower rank decomposition
    
    # For rank 26 or 27, we need to search the space of possible decompositions
    # This is a hard problem, but we can try some heuristics
    
    # Strategy: Start from rank 28 and try to merge columns
    # Or: Try random initialization and ALS for rank 26, 27
    
    # Let's try ALS for rank 27 first
    if time.time() < deadline - 30:
        rank_27 = _als_decompose(n, m, p, 27, time_limit=30)
        if rank_27 is not None:
            best_solution = rank_27
            best_rank = 27
    
    # Try rank 26
    if time.time() < deadline - 10:
        rank_26 = _als_decompose(n, m, p, 26, time_limit=10)
        if rank_26 is not None:
            best_solution = rank_26
            best_rank = 26
    
    # Convert to list of lists with integer/half-integer entries if possible
    U, V, W = best_solution
    U = _round_to_half_int(U)
    V = _round_to_half_int(V)
    W = _round_to_half_int(W)
    
    # Verify the solution
    if not _verify_decomposition(n, m, p, U, V, W):
        # Fall back to the guaranteed rank 28 solution
        U, V, W = _strassen_28(n, m, p)
        U = _round_to_half_int(U)
        V = _round_to_half_int(V)
        W = _round_to_half_int(W)
    
    return {"U": U, "V": V, "W": W}

def _strassen_28(n, m, p):
    """
    Generate a rank-28 decomposition using Strassen for each 2x2 block of B.
    """
    num_blocks = p // 2  # 4 blocks
    rank = num_blocks * 7  # 28
    
    U = np.zeros((n * m, rank))  # 4 x 28
    V = np.zeros((m * p, rank))  # 16 x 28
    W = np.zeros((p * n, rank))  # 16 x 28
    
    for b in range(num_blocks):
        base = b * 7
        # M1 = (A[0,0]+A[1,1]) * (B[0,2b]+B[1,2b+1])
        U[0, base + 0] += 1
        U[3, base + 0] += 1
        V[0 * p + 2 * b, base + 0] += 1
        V[1 * p + 2 * b + 1, base + 0] += 1
        W[2 * b * n + 0, base + 0] += 1
        W[(2 * b + 1) * n + 1, base + 0] += 1
        
        # M2 = (A[1,0]+A[1,1]) * B[0,2b]
        U[2, base + 1] += 1
        U[3, base + 1] += 1
        V[0 * p + 2 * b, base + 1] += 1
        W[2 * b * n + 1, base + 1] += 1
        W[(2 * b + 1) * n + 1, base + 1] -= 1
        
        # M3 = A[0,0] * (B[0,2b+1]-B[1,2b+1])
        U[0, base + 2] += 1
        V[0 * p + 2 * b + 1, base + 2] += 1
        V[1 * p + 2 * b + 1, base + 2] -= 1
        W[(2 * b + 1) * n + 0, base + 2] += 1
        W[(2 * b + 1) * n + 1, base + 2] += 1
        
        # M4 = A[1,1] * (B[1,2b]-B[0,2b])
        U[3, base + 3] += 1
        V[1 * p + 2 * b, base + 3] += 1
        V[0 * p + 2 * b, base + 3] -= 1
        W[2 * b * n + 0, base + 3] += 1
        W[2 * b * n + 1, base + 3] += 1
        
        # M5 = (A[0,0]+A[0,1]) * B[1,2b+1]
        U[0, base + 4] += 1
        U[1, base + 4] += 1
        V[1 * p + 2 * b + 1, base + 4] += 1
        W[2 * b * n + 0, base + 4] -= 1
        W[(2 * b + 1) * n + 0, base + 4] += 1
        
        # M6 = (A[1,0]-A[0,0]) * (B[0,2b]+B[0,2b+1])
        U[2, base + 5] += 1
        U[0, base + 5] -= 1
        V[0 * p + 2 * b, base + 5] += 1
        V[0 * p + 2 * b + 1, base + 5] += 1
        W[(2 * b + 1) * n + 1, base + 5] += 1
        
        # M7 = (A[0,1]-A[1,1]) * (B[1,2b]+B[1,2b+1])
        U[1, base + 6] += 1
        U[3, base + 6] -= 1
        V[1 * p + 2 * b, base + 6] += 1
        V[1 * p + 2 * b + 1, base + 6] += 1
        W[2 * b * n + 0, base + 6] += 1
    
    return U, V, W

def _round_to_half_int(arr):
    """
    Round array entries to nearest half-integer (multiple of 0.5).
    """
    arr = np.round(arr * 2) / 2.0
    # Convert to list of lists
    return arr.tolist()

def _verify_decomposition(n, m, p, U, V, W):
    """
    Verify that U, V, W form a valid tensor decomposition.
    """
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # The tensor T_{i,k,j} should equal sum_r U_{i,k,r} * V_{k,j,r} * W_{j,i,r}
    # Note: W uses convention W[k*n+i, r] where k is column index of C (0..p-1), i is row index (0..n-1)
    # So W_{j,i,r} in tensor notation corresponds to W[j*n+i, r] in the array
    
    # Build the tensor from the decomposition
    rank = U.shape[1]
    T_decomp = np.zeros((n, m, p))
    
    for r in range(rank):
        # U[:, r] is a vector of length n*m, reshape to (n, m)
        U_r = U[:, r].reshape(n, m)
        # V[:, r] is a vector of length m*p, reshape to (m, p)
        V_r = V[:, r].reshape(m, p)
        # W[:, r] is a vector of length p*n, reshape to (p, n)
        W_r = W[:, r].reshape(p, n)
        
        # T_{i,k,j} += U_{i,k,r} * V_{k,j,r} * W_{j,i,r}
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    T_decomp[i, k, j] += U_r[i, k] * V_r[k, j] * W_r[j, i]
    
    # Build the true matrix multiplication tensor
    # T_{i,k,j} = 1 if this term appears in C_{i,j} = sum_k A_{i,k} B_{k,j}
    # For matrix multiplication, T_{i,k,j} = 1 for all i, k, j
    T_true = np.ones((n, m, p))
    
    # Check if they match
    return np.allclose(T_decomp, T_true, atol=1e-9)

def _als_decompose(n, m, p, target_rank, time_limit=30):
    """
    Try to find a decomposition of the given rank using ALS optimization.
    Returns (U, V, W) if successful, None otherwise.
    """
    start_time = time.time()
    
    # Initialize with random values
    best_U = None
    best_V = None
    best_W = None
    best_error = float('inf')
    
    num_restarts = 5
    max_iter = 200
    
    for restart in range(num_restarts):
        if time.time() > start_time + time_limit:
            break
        
        # Random initialization
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)
        
        # Normalize columns to avoid scaling issues
        for r in range(target_rank):
            norm_u = np.linalg.norm(U[:, r])
            norm_v = np.linalg.norm(V[:, r])
            norm_w = np.linalg.norm(W[:, r])
            if norm_u > 1e-10:
                U[:, r] /= norm_u
            if norm_v > 1e-10:
                V[:, r] /= norm_v
            if norm_w > 1e-10:
                W[:, r] /= norm_w
        
        error_history = []
        
        for it in range(max_iter):
            if time.time() > start_time + time_limit:
                break
            
            # Update U
            U_new = _update_U(n, m, p, target_rank, V, W)
            # Update V
            V_new = _update_V(n, m, p, target_rank, U_new, W)
            # Update W
            W_new = _update_W(n, m, p, target_rank, U_new, V_new)
            
            # Compute error
            error = _compute_error(n, m, p, U_new, V_new, W_new)
            error_history.append(error)
            
            # Check convergence
            if len(error_history) > 10:
                if abs(error_history[-1] - error_history[-11]) < 1e-12:
                    break
            
            U, V, W = U_new, V_new, W_new
        
        # Check if we found a valid solution
        if _verify_decomposition(n, m, p, U, V, W):
            return U, V, W
        
        # Track best
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # If we didn't find an exact solution, return None
    # (We could try to round and verify, but for now just return None)
    return None

def _update_U(n, m, p, rank, V, W):
    """
    Update U given V and W.
    For each (i, k), we need:
    sum_r U[i*m+k, r] * (sum_j V[k*p+j, r] * W[j*n+i, r]) = 1
    """
    U_new = np.zeros((n * m, rank))
    
    # Precompute for each (i, k) the vector b_r = sum_j V[k*p+j, r] * W[j*n+i, r]
    # Then solve U[i*m+k, :] such that U[i*m+k, :] @ b = 1
    # This is an underdetermined system; we find the minimum-norm solution
    
    for i in range(n):
        for k in range(m):
            b = np.zeros(rank)
            for j in range(p):
                b += V[k * p + j, :] * W[j * n + i, :]
            
            # Solve U_row @ b = 1 with minimum norm
            # If b is zero, this is infeasible; set to zero
            if np.linalg.norm(b) < 1e-10:
                U_new[i * m + k, :] = 0
            else:
                # Minimum norm solution: U_row = b / (b @ b)
                U_new[i * m + k, :] = b / (b @ b)
    
    return U_new

def _update_V(n, m, p, rank, U, W):
    """
    Update V given U and W.
    For each (k, j), we need:
    sum_r V[k*p+j, r] * (sum_i U[i*m+k, r] * W[j*n+i, r]) = 1
    """
    V_new = np.zeros((m * p, rank))
    
    for k in range(m):
        for j in range(p):
            b = np.zeros(rank)
            for i in range(n):
                b += U[i * m + k, :] * W[j * n + i, :]
            
            if np.linalg.norm(b) < 1e-10:
                V_new[k * p + j, :] = 0
            else:
                V_new[k * p + j, :] = b / (b @ b)
    
    return V_new

def _update_W(n, m, p, rank, U, V):
    """
    Update W given U and V.
    For each (j, i), we need:
    sum_r W[j*n+i, r] * (sum_k U[i*m+k, r] * V[k*p+j, r]) = 1
    """
    W_new = np.zeros((p * n, rank))
    
    for j in range(p):
        for i in range(n):
            b = np.zeros(rank)
            for k in range(m):
                b += U[i * m + k, :] * V[k * p + j, :]
            
            if np.linalg.norm(b) < 1e-10:
                W_new[j * n + i, :] = 0
            else:
                W_new[j * n + i, :] = b / (b @ b)
    
    return W_new

def _compute_error(n, m, p, U, V, W):
    """
    Compute the error of the decomposition.
    """
    rank = U.shape[1]
    T_decomp = np.zeros((n, m, p))
    
    for r in range(rank):
        U_r = U[:, r].reshape(n, m)
        V_r = V[:, r].reshape(m, p)
        W_r = W[:, r].reshape(p, n)
        
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    T_decomp[i, k, j] += U_r[i, k] * V_r[k, j] * W_r[j, i]
    
    T_true = np.ones((n, m, p))
    return np.linalg.norm(T_decomp - T_true)
# EVOLVE-BLOCK-END
