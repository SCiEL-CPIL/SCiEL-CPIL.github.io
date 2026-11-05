"""Rank-reduced exact decomposition for 3x4 times 4x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact decomposition for C = A(3x4) * B(4x4) using Strassen-based
    block decomposition. Rank-44 via 2x2 Strassen + 1x2 schoolbook.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = 3, 4, 4
    
    def schoolbook():
        """Rank-48 schoolbook fallback."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    total_rank = 44
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(i, k): return k * n + i
    
    def add_strassen_2x2x2(i0, j0, k0, col_start):
        a11 = u_idx(i0, j0)
        a12 = u_idx(i0, j0+1)
        a21 = u_idx(i0+1, j0)
        a22 = u_idx(i0+1, j0+1)
        b11 = v_idx(j0, k0)
        b12 = v_idx(j0, k0+1)
        b21 = v_idx(j0+1, k0)
        b22 = v_idx(j0+1, k0+1)
        c11 = w_idx(i0, k0)
        c12 = w_idx(i0, k0+1)
        c21 = w_idx(i0+1, k0)
        c22 = w_idx(i0+1, k0+1)
        c = col_start
        U[a11][c] += 1; U[a22][c] += 1; V[b11][c] += 1; V[b22][c] += 1; W[c11][c] += 1; W[c22][c] += 1; c += 1
        U[a21][c] += 1; U[a22][c] += 1; V[b11][c] += 1; W[c21][c] += 1; W[c22][c] -= 1; c += 1
        U[a11][c] += 1; V[b12][c] += 1; V[b22][c] -= 1; W[c12][c] += 1; W[c22][c] += 1; c += 1
        U[a22][c] += 1; V[b21][c] += 1; V[b11][c] -= 1; W[c11][c] += 1; W[c21][c] += 1; c += 1
        U[a11][c] += 1; U[a12][c] += 1; V[b22][c] += 1; W[c12][c] += 1; W[c11][c] -= 1; c += 1
        U[a21][c] += 1; U[a11][c] -= 1; V[b11][c] += 1; V[b12][c] += 1; W[c22][c] += 1; c += 1
        U[a12][c] += 1; U[a22][c] -= 1; V[b21][c] += 1; V[b22][c] += 1; W[c11][c] += 1; c += 1
        return c
    
    def add_schoolbook_1x2x2(i0, j0, k0, col_start):
        c = col_start
        for dj in range(2):
            for dk in range(2):
                U[u_idx(i0, j0+dj)][c] = 1
                V[v_idx(j0+dj, k0+dk)][c] = 1
                W[w_idx(i0, k0+dk)][c] = 1
                c += 1
        return c
    
    col = 0
    col = add_strassen_2x2x2(0, 0, 0, col)
    col = add_strassen_2x2x2(0, 2, 0, col)
    col = add_strassen_2x2x2(0, 0, 2, col)
    col = add_strassen_2x2x2(0, 2, 2, col)
    col = add_schoolbook_1x2x2(2, 0, 0, col)
    col = add_schoolbook_1x2x2(2, 2, 0, col)
    col = add_schoolbook_1x2x2(2, 0, 2, col)
    col = add_schoolbook_1x2x2(2, 2, 2, col)
    
    assert col == total_rank, f"Column mismatch: {col} != {col}"
    
    def verify(U_list, V_list, W_list, rank):
        try:
            Ua = np.array(U_list, dtype=np.float64)
            Va = np.array(V_list, dtype=np.float64)
            Wa = np.array(W_list, dtype=np.float64)
            U3 = Ua.reshape(n, m, rank)
            V3 = Va.reshape(m, p, rank)
            W3 = Wa.reshape(p, n, rank)
            T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            return np.allclose(T, 1.0, atol=1e-10)
        except Exception:
            return False
    
    if not verify(U, V, W, total_rank):
        return schoolbook()
    
    # Try to reduce rank by checking for linearly dependent columns
    # Use a more robust approach: check if removing a column and adjusting
    # others can preserve the tensor
    
    def try_remove_column(U_list, V_list, W_list, col_to_remove, target_rank):
        """Try to remove a specific column and see if the remaining columns
        can be adjusted to preserve the tensor."""
        if time.time() > deadline - 10.0:
            return None
        
        Ua = np.array(U_list, dtype=np.float64)
        Va = np.array(V_list, dtype=np.float64)
        Wa = np.array(W_list, dtype=np.float64)
        
        remaining = [c for c in range(total_rank) if c != col_to_remove]
        
        # Build the system to solve for adjustments
        # The tensor contribution of column c is:
        # T_c[i,j,k] = U[i,j,c] * V[j,k,c] * W[k,i,c]
        # We need to find new U', V', W' for remaining columns such that
        # sum_{c' in remaining} U'[i,j,c'] * V'[j,k,c'] * W'[k,i,c'] = T_c[i,j,k]
        # plus the original contributions from remaining columns
        
        # This is a nonlinear problem. Instead, let's check if the column
        # is already redundant by checking if its tensor is in the span
        # of the other columns' tensors.
        
        # Compute tensor for column to remove
        T_remove = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_remove[i,j,k] = Ua[i*m+j, col_to_remove] * Va[j*p+k, col_to_remove] * Wa[k*n+i, col_to_remove]
        
        if np.allclose(T_remove, 0, atol=1e-10):
            # Column contributes nothing, safe to remove
            U_new = [[0] * target_rank for _ in range(n * m)]
            V_new = [[0] * target_rank for _ in range(m * p)]
            W_new = [[0] * target_rank for _ in range(p * n)]
            for ci, c in enumerate(remaining):
                for i in range(n * m):
                    U_new[i][ci] = Ua[i, c]
                for j in range(m * p):
                    V_new[j][ci] = Va[j, c]
                for k in range(p * n):
                    W_new[k][ci] = Wa[k, c]
            if verify(U_new, V_new, W_new, target_rank):
                return U_new, V_new, W_new
        
        # Check if T_remove is in the span of other columns' tensors
        # Build matrix where each column is a flattened tensor
        num_eqs = n * m * p
        tensors = np.zeros((num_eqs, target_rank))
        for ci, c in enumerate(remaining):
            for idx, (i, j, k) in enumerate([(i, j, k) for i in range(n) for j in range(m) for k in range(p)]):
                tensors[idx, ci] = Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
        
        b = T_remove.flatten()
        
        try:
            alpha, residuals, rank_sol, sv = np.linalg.lstsq(tensors, b, rcond=None)
        except Exception:
            return None
        
        if np.allclose(tensors @ alpha, b, atol=1e-8):
            # The column is in the span, but we can't just add alpha * tensors
            # because the decomposition is trilinear, not linear in each factor
            # separately. This approach doesn't directly work.
            pass
        
        return None
    
    # Try removing columns from the end
    for elim in range(total_rank - 1, max(40, total_rank - 4), -1):
        if time.time() > deadline - 15.0:
            break
        
        result = try_remove_column(U, V, W, elim, total_rank - 1)
        if result is not None:
            U, V, W = result
            total_rank -= 1
    
    # Final verification
    if not verify(U, V, W, total_rank):
        return schoolbook()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
