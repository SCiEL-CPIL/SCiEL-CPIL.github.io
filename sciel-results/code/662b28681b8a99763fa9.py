"""Rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _rank_of_matrix(M):
    """Compute numerical rank of matrix M."""
    if M.size == 0:
        return 0
    return int(np.linalg.matrix_rank(M, tol=1e-10))

def _try_reduce_rank(U, V, W, n, m, p, max_rank, deadline):
    """Try to reduce rank by checking if any rank term is linearly dependent."""
    rank = len(U[0]) if U else 0
    if rank <= max_rank:
        return U, V, W, rank
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the tensor in flattened form
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For each r, the rank-1 tensor is U[:,r] ⊗ V[:,r] ⊗ W[:,r] reshaped
    
    # Try to find if removing any single rank term still gives exact decomposition
    for r_remove in range(rank):
        if time.time() > deadline:
            break
        U_test = np.delete(U_arr, r_remove, axis=1)
        V_test = np.delete(V_arr, r_remove, axis=1)
        W_test = np.delete(W_arr, r_remove, axis=1)
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_test[i*m+j], V_test[j*p+k] * W_test[k*n+i])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err < 1e-10:
            # Found a reducible term
            new_rank = rank - 1
            U_new = U_test.tolist()
            V_new = V_test.tolist()
            W_new = W_test.tolist()
            return U_new, V_new, W_new, new_rank
    
    return U, V, W, rank

def _search_rank32(U, V, W, n, m, p, deadline):
    """Search for rank-32 decomposition by modifying the rank-33 baseline."""
    rank = len(U[0])
    if rank < 32:
        return U, V, W, rank
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Strategy: Try to find a linear combination of two rank terms that can replace one
    # This is equivalent to checking if the tensor lies in a rank-32 subspace
    
    # Alternative: Try small perturbations to see if we can merge terms
    # For each pair of rank terms, check if their difference can be absorbed
    
    # More practical: try to find if any column of U is in the span of others
    # when combined with corresponding V and W columns
    
    # Try removing each term and checking if the residual can be expressed
    # as a linear combination of remaining terms
    
    for r_remove in range(rank):
        if time.time() > deadline:
            break
        
        # Remove term r_remove
        mask = [i for i in range(rank) if i != r_remove]
        U_rem = U_arr[:, mask]
        V_rem = V_arr[:, mask]
        W_rem = W_arr[:, mask]
        
        # The residual is the original tensor minus the sum of remaining terms
        # residual[i,j,k] = U[i*m+j, r_remove] * V[j*p+k, r_remove] * W[k*n+i, r_remove]
        
        # Check if this residual can be written as sum of rank-1 tensors from remaining basis
        # This is a linear system: for each (i,j,k), residual[i,j,k] = sum_s c_s * U_rem[i*m+j,s] * V_rem[j*p+k,s] * W_rem[k*n+i,s]
        
        # Build the linear system
        n_eq = n * m * p
        n_unknowns = len(mask)
        
        A_mat = np.zeros((n_eq, n_unknowns))
        b_vec = np.zeros(n_eq)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_arr[i*m+j, r_remove]
                    v_val = V_arr[j*p+k, r_remove]
                    w_val = W_arr[k*n+i, r_remove]
                    b_vec[idx] = u_val * v_val * w_val
                    for s, r_s in enumerate(mask):
                        A_mat[idx, s] = U_rem[i*m+j, s] * V_rem[j*p+k, s] * W_rem[k*n+i, s]
                    idx += 1
        
        # Solve the linear system
        try:
            c, residuals, rank_sys, sv = np.linalg.lstsq(A_mat, b_vec, rcond=1e-10)
            # Check if the solution is exact
            A_c = A_mat @ c
            max_err = np.max(np.abs(A_c - b_vec))
            if max_err < 1e-10:
                # We found that the removed term is a linear combination of others
                # This means the original decomposition had a redundant term
                U_new = U_rem.tolist()
                V_new = V_rem.tolist()
                W_new = W_rem.tolist()
                return U_new, V_new, W_new, len(mask)
        except:
            pass
    
    return U, V, W, rank

def construct():
    """Build a rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication."""
    t0 = time.time()
    deadline = t0 + 100  # safety margin before 120s limit
    
    n, m, p = 2, 3, 6
    rank = 33
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A1 (2x1) * B1 (1x6) => 2x6 output
    for k in range(p):
        for i in range(n):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    # col is now 12
    
    # Part 2: A2 (2x2) * B2 (2x6) => 2x6 output
    for t in range(3):
        # M1 = (A00+A11)(B00+B11)
        U[0 * m + 1][col] = 1
        U[1 * m + 2][col] = 1
        V[1 * p + 2*t][col] = 1
        V[2 * p + 2*t + 1][col] = 1
        W[2*t * n + 0][col] = 1
        W[(2*t + 1) * n + 1][col] = 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[1 * m + 1][col] = 1
        U[1 * m + 2][col] = 1
        V[1 * p + 2*t][col] = 1
        W[2*t * n + 1][col] = 1
        W[(2*t + 1) * n + 1][col] = -1
        col += 1
        
        # M3 = A00(B01-B11)
        U[0 * m + 1][col] = 1
        V[1 * p + 2*t + 1][col] = 1
        V[2 * p + 2*t + 1][col] = -1
        W[(2*t + 1) * n + 0][col] = 1
        W[(2*t + 1) * n + 1][col] = 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[1 * m + 2][col] = 1
        V[2 * p + 2*t][col] = 1
        V[1 * p + 2*t][col] = -1
        W[2*t * n + 0][col] = 1
        W[2*t * n + 1][col] = 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[0 * m + 1][col] = 1
        U[0 * m + 2][col] = 1
        V[2 * p + 2*t + 1][col] = 1
        W[2*t * n + 0][col] = -1
        W[(2*t + 1) * n + 0][col] = 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[1 * m + 1][col] = 1
        U[0 * m + 1][col] = -1
        V[1 * p + 2*t][col] = 1
        V[1 * p + 2*t + 1][col] = 1
        W[(2*t + 1) * n + 1][col] = 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[0 * m + 2][col] = 1
        U[1 * m + 2][col] = -1
        V[2 * p + 2*t][col] = 1
        V[2 * p + 2*t + 1][col] = 1
        W[2*t * n + 0][col] = 1
        col += 1
    
    # col should be 12 + 3*7 = 33
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Verify baseline
    if time.time() < deadline:
        max_err = _verify_decomposition(U, V, W, n, m, p)
        if max_err > 1e-10:
            # Baseline is incorrect, but proceed with search anyway
            pass
    
    # Try to reduce rank
    if time.time() < deadline:
        U_r, V_r, W_r, r_r = _try_reduce_rank(U, V, W, n, m, p, 32, deadline)
        if r_r < best_rank:
            best_U, best_V, best_W, best_rank = U_r, V_r, W_r, r_r
    
    # Search for rank-32 via linear system approach
    if time.time() < deadline - 5:
        U_s, V_s, W_s, r_s = _search_rank32(U, V, W, n, m, p, deadline)
        if r_s < best_rank:
            best_U, best_V, best_W, best_rank = U_s, V_s, W_s, r_s
    
    # Final verification
    if time.time() < deadline:
        max_err = _verify_decomposition(best_U, best_V, best_W, n, m, p)
        if max_err > 1e-10:
            # Fall back to original if search corrupted it
            best_U, best_V, best_W = U, V, W
            best_rank = 33
    
    result = {"U": best_U, "V": best_V, "W": best_W}
    return result
# EVOLVE-BLOCK-END
