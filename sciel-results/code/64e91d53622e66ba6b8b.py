"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomp(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct using numpy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape for efficient verification
    # U: (n*m, rank) -> (n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)
    # W: (p*n, rank) -> (p, n, rank)
    U_t = U_arr.reshape(n, m, rank)
    V_t = V_arr.reshape(m, p, rank)
    W_t = W_arr.reshape(p, n, rank)
    
    # For each (i, l, k): sum_r U[i,l,r] * V[l,k,r] * W[k,i,r] == 1
    # Use einsum: result[i,l,k] = sum_r U[i,l,r] * V[l,k,r] * W[k,i,r]
    result = np.einsum('ilr,lkr,kir->ilk', U_t, V_t, W_t)
    
    if not np.allclose(result, 1.0, atol=1e-9):
        diff = np.abs(result - 1.0)
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)
        return False, (max_idx[0], max_idx[1], max_idx[2], float(result[max_idx]))
    return True, None

def _strassen_block(j0, j1, k0, k1, U, V, W, col, n, m, p):
    """Add 7 Strassen products for a 2x2x2 block and return new col."""
    # P1 = (A[0,j0]+A[1,j1]) * (B[j0,k0]+B[j1,k1])
    U[0*m + j0][col] += 1
    U[1*m + j1][col] += 1
    V[j0*p + k0][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    # P2 = (A[1,j0]+A[1,j1]) * B[j0,k0]
    U[1*m + j0][col] += 1
    U[1*m + j1][col] += 1
    V[j0*p + k0][col] += 1
    W[k0*n + 1][col] += 1
    W[k1*n + 1][col] -= 1
    col += 1
    
    # P3 = A[0,j0] * (B[j0,k1]-B[j1,k1])
    U[0*m + j0][col] += 1
    V[j0*p + k1][col] += 1
    V[j1*p + k1][col] -= 1
    W[k1*n + 0][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    # P4 = A[1,j1] * (B[j1,k0]-B[j0,k0])
    U[1*m + j1][col] += 1
    V[j1*p + k0][col] += 1
    V[j0*p + k0][col] -= 1
    W[k0*n + 0][col] += 1
    W[k0*n + 1][col] += 1
    col += 1
    
    # P5 = (A[0,j0]+A[0,j1]) * B[j1,k1]
    U[0*m + j0][col] += 1
    U[0*m + j1][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] -= 1
    W[k1*n + 0][col] += 1
    col += 1
    
    # P6 = (A[1,j0]-A[0,j0]) * (B[j0,k0]+B[j0,k1])
    U[1*m + j0][col] += 1
    U[0*m + j0][col] -= 1
    V[j0*p + k0][col] += 1
    V[j0*p + k1][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    # P7 = (A[0,j1]-A[1,j1]) * (B[j1,k0]+B[j1,k1])
    U[0*m + j1][col] += 1
    U[1*m + j1][col] -= 1
    V[j1*p + k0][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] += 1
    col += 1
    
    return col

def _try_eliminate_term(U, V, W, n, m, p, rank, target_col):
    """
    Try to eliminate the term at target_col by expressing it as a combination of others.
    Returns (success, new_U, new_V, new_W, new_rank)
    """
    # The idea: if we can find coefficients c_r for r != target_col such that
    # the tensor contribution of term target_col equals sum_{r != target_col} c_r * term_r,
    # then we can eliminate it.
    
    # This is equivalent to checking if the vector of tensor contributions from term target_col
    # lies in the span of the other terms' contributions.
    
    # Build the "contribution vectors" for each term
    # Term r contributes: U[:,r] ⊗ V[:,r] ⊗ W[:,r] to the tensor
    # Flatten each into a vector of size n*m*p
    
    nmp = n * m * p
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Get the target term's contribution vector
    target_vec = np.zeros(nmp)
    for i in range(n):
        for l in range(m):
            for k in range(p):
                target_vec[i*m*p + l*p + k] = U_arr[i*m+l, target_col] * V_arr[l*p+k, target_col] * W_arr[k*n+i, target_col]
    
    # Build matrix of other terms' contributions
    other_cols = [r for r in range(rank) if r != target_col]
    A_mat = np.zeros((nmp, len(other_cols)))
    for idx, r in enumerate(other_cols):
        for i in range(n):
            for l in range(m):
                for k in range(p):
                    A_mat[i*m*p + l*p + k, idx] = U_arr[i*m+l, r] * V_arr[l*p+k, r] * W_arr[k*n+i, r]
    
    # Try to solve A_mat @ c = target_vec
    try:
        c, residuals, rank_mat, sv = np.linalg.lstsq(A_mat, target_vec, rcond=None)
        # Check if the solution is good
        if np.allclose(A_mat @ c, target_vec, atol=1e-9):
            # We can eliminate the target term
            # New U, V, W with the target column removed and others scaled
            new_rank = rank - 1
            new_U = [[0] * new_rank for _ in range(n * m)]
            new_V = [[0] * new_rank for _ in range(m * p)]
            new_W = [[0] * new_rank for _ in range(p * n)]
            
            for idx, r in enumerate(other_cols):
                for i in range(n * m):
                    new_U[i][idx] = U[i][r] + c[idx] * U[i][target_col]
                for i in range(m * p):
                    new_V[i][idx] = V[i][r] + c[idx] * V[i][target_col]
                for i in range(p * n):
                    new_W[i][idx] = W[i][r] + c[idx] * W[i][target_col]
            
            return True, new_U, new_V, new_W, new_rank
    except:
        pass
    
    return False, None, None, None, rank

def construct():
    n, m, p = DIMENSIONS
    _start_time = time.time()
    deadline = _start_time + 140  # Leave 10s margin
    
    # Start with the known rank-54 decomposition
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            col = _strassen_block(j0, j1, k0, k1, U, V, W, col, n, m, p)
    
    # Remaining j=4
    for k in range(p):
        U[0*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 0][col] += 1
        col += 1
        
        U[1*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 1][col] += 1
        col += 1
    
    # Verify the initial decomposition
    valid, _ = _verify_decomp(U, V, W, n, m, p, rank)
    if not valid:
        pass
    
    best_rank = rank
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    # Try to improve by eliminating terms
    # Try each column as a potential elimination target
    for target in range(rank):
        if time.time() > deadline:
            break
        
        success, new_U, new_V, new_W, new_rank = _try_eliminate_term(U, V, W, n, m, p, rank, target)
        
        if success:
            # Verify the new decomposition
            valid_new, _ = _verify_decomp(new_U, new_V, new_W, n, m, p, new_rank)
            if valid_new:
                if new_rank < best_rank:
                    best_rank = new_rank
                    best_U, best_V, best_W = new_U, new_V, new_W
                    # Update U, V, W, rank for further elimination attempts
                    U, V, W, rank = new_U, new_V, new_W, new_rank
                    # After elimination, the column indices shift, so restart the loop
                    break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
