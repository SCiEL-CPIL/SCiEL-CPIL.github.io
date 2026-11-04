import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140
    
    n, m, p = DIMENSIONS
    
    # Start with rank 50 decomposition (6 Strassen blocks + 8 schoolbook)
    rank_init = 50
    U = np.zeros((n * m, rank_init), dtype=np.float64)
    V = np.zeros((m * p, rank_init), dtype=np.float64)
    W = np.zeros((p * n, rank_init), dtype=np.float64)
    
    col = 0
    
    def emit_strassen(i0, j0, k0):
        nonlocal col
        # M1: (A11+A22)(B11+B22)
        U[i0*m + j0, col] = 1.0
        U[(i0+1)*m + (j0+1), col] = 1.0
        V[j0*p + k0, col] = 1.0
        V[(j0+1)*p + (k0+1), col] = 1.0
        W[k0*n + i0, col] = 1.0
        W[(k0+1)*n + (i0+1), col] = 1.0
        col += 1
        
        # M2: (A21+A22)B11
        U[(i0+1)*m + j0, col] = 1.0
        U[(i0+1)*m + (j0+1), col] = 1.0
        V[j0*p + k0, col] = 1.0
        W[k0*n + (i0+1), col] = 1.0
        W[(k0+1)*n + (i0+1), col] = -1.0
        col += 1
        
        # M3: A11(B12-B22)
        U[i0*m + j0, col] = 1.0
        V[j0*p + (k0+1), col] = 1.0
        V[(j0+1)*p + (k0+1), col] = -1.0
        W[(k0+1)*n + i0, col] = 1.0
        W[(k0+1)*n + (i0+1), col] = 1.0
        col += 1
        
        # M4: A22(B21-B11)
        U[(i0+1)*m + (j0+1), col] = 1.0
        V[(j0+1)*p + k0, col] = 1.0
        V[j0*p + k0, col] = -1.0
        W[k0*n + i0, col] = 1.0
        W[k0*n + (i0+1), col] = 1.0
        col += 1
        
        # M5: (A11+A12)B22
        U[i0*m + j0, col] = 1.0
        U[i0*m + (j0+1), col] = 1.0
        V[(j0+1)*p + (k0+1), col] = 1.0
        W[k0*n + i0, col] = -1.0
        W[(k0+1)*n + i0, col] = 1.0
        col += 1
        
        # M6: (A21-A11)(B11+B12)
        U[(i0+1)*m + j0, col] = 1.0
        U[i0*m + j0, col] = -1.0
        V[j0*p + k0, col] = 1.0
        V[j0*p + (k0+1), col] = 1.0
        W[(k0+1)*n + (i0+1), col] = 1.0
        col += 1
        
        # M7: (A12-A22)(B21+B22)
        U[i0*m + (j0+1), col] = 1.0
        U[(i0+1)*m + (j0+1), col] = -1.0
        V[(j0+1)*p + k0, col] = 1.0
        V[(j0+1)*p + (k0+1), col] = 1.0
        W[k0*n + i0, col] = 1.0
        col += 1

    def emit_schoolbook_2x2x1(i0, j0, k0):
        nonlocal col
        U[i0*m + j0, col] = 1.0
        V[j0*p + k0, col] = 1.0
        W[k0*n + i0, col] = 1.0
        col += 1
        
        U[i0*m + (j0+1), col] = 1.0
        V[(j0+1)*p + k0, col] = 1.0
        W[k0*n + i0, col] = 1.0
        col += 1
        
        U[(i0+1)*m + j0, col] = 1.0
        V[j0*p + k0, col] = 1.0
        W[k0*n + (i0+1), col] = 1.0
        col += 1
        
        U[(i0+1)*m + (j0+1), col] = 1.0
        V[(j0+1)*p + k0, col] = 1.0
        W[k0*n + (i0+1), col] = 1.0
        col += 1

    # 6 Strassen blocks for k=0..5
    emit_strassen(0, 0, 0)
    emit_strassen(0, 0, 2)
    emit_strassen(0, 2, 0)
    emit_strassen(0, 2, 2)
    emit_strassen(0, 0, 4)
    emit_strassen(0, 2, 4)
    
    # 8 schoolbook terms for k=6
    for i in range(2):
        for j in range(4):
            U[i*m + j, col] = 1.0
            V[j*p + 6, col] = 1.0
            W[6*n + i, col] = 1.0
            col += 1
    
    # Verify initial decomposition
    U_3d = U.reshape(n, m, rank_init)
    V_3d = V.reshape(m, p, rank_init)
    W_3d = W.reshape(p, n, rank_init)
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    if not np.allclose(T, 1.0, atol=1e-9):
        return None
    
    # Now try to find linear dependencies to reduce rank
    # For each column r, check if it can be expressed as a linear combination of others
    # T_r[i,j,k] = U_r[i,j] * V_r[j,k] * W_r[k,i]
    # We want: T_r = sum_{s != r} c_s * T_s
    
    best_rank = rank_init
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try eliminating each column
    for elim_col in range(rank_init):
        if time.time() > DEADLINE - 10:
            break
        
        # Build the tensor for the column to eliminate
        T_elim = U[:, elim_col].reshape(n, m) * 0  # placeholder
        # Actually, compute the full tensor contribution
        U_e = U[:, elim_col].reshape(n, m)
        V_e = V[:, elim_col].reshape(m, p)
        W_e = W[:, elim_col].reshape(p, n)
        
        # T_elim[i,j,k] = U_e[i,j] * V_e[j,k] * W_e[k,i]
        T_elim = np.einsum('ij,jk,ki->ijk', U_e, V_e, W_e)
        
        # Build matrix of other columns' tensor contributions
        other_cols = [c for c in range(rank_init) if c != elim_col]
        # For each other column, compute its tensor
        # We need to solve: T_elim = sum_{s in other} c_s * T_s
        
        # Flatten tensors
        T_elim_flat = T_elim.flatten()
        
        # Build coefficient matrix
        A_mat = np.zeros((n*m*p, len(other_cols)))
        for idx, c in enumerate(other_cols):
            U_c = U[:, c].reshape(n, m)
            V_c = V[:, c].reshape(m, p)
            W_c = W[:, c].reshape(p, n)
            T_c = np.einsum('ij,jk,ki->ijk', U_c, V_c, W_c)
            A_mat[:, idx] = T_c.flatten()
        
        # Solve least squares
        try:
            coeffs, residuals, rank_mat, sv = np.linalg.lstsq(A_mat, T_elim_flat, rcond=1e-10)
        except:
            continue
        
        # Check if the fit is good
        T_recon = A_mat @ coeffs
        if np.allclose(T_recon, T_elim_flat, atol=1e-9):
            # We can eliminate this column!
            new_rank = rank_init - 1
            if new_rank < best_rank:
                # Build new U, V, W without this column
                new_U = np.zeros((n*m, new_rank))
                new_V = np.zeros((m*p, new_rank))
                new_W = np.zeros((p*n, new_rank))
                
                new_col = 0
                for c in other_cols:
                    new_U[:, new_col] = U[:, c]
                    new_V[:, new_col] = V[:, c]
                    new_W[:, new_col] = W[:, c]
                    new_col += 1
                
                best_rank = new_rank
                best_U, best_V, best_W = new_U, new_V, new_W
                
                # Update U, V, W for subsequent elimination attempts
                U, V, W = new_U, new_V, new_W
                rank_init = new_rank
    
    # Final verification
    if best_rank < rank_init:
        U_3d = best_U.reshape(n, m, best_rank)
        V_3d = best_V.reshape(m, p, best_rank)
        W_3d = best_W.reshape(p, n, best_rank)
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        if not np.allclose(T, 1.0, atol=1e-9):
            # Fallback to original
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
