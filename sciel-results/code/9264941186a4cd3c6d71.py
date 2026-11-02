import numpy as np
import time

DIMENSIONS = [3, 5, 6]
DEADLINE_SECONDS = 120

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i, j, k.
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
        
    rank = U.shape[1]
    
    # Reshape to 3D arrays for einsum
    # U[i, j, r], V[j, k, r], W[k, i, r]
    U_reshaped = U.reshape(n, m, rank)
    V_reshaped = V.reshape(m, p, rank)
    W_reshaped = W.reshape(p, n, rank)
    
    # Compute the tensor contraction
    # C[i, j, k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    try:
        C = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped, optimize=True)
    except Exception:
        return False
        
    # Check if C is all 1s
    return np.allclose(C, 1.0, atol=tol)

def _check_integer_half_integer(U, V, W, tol=1e-6):
    """Check if entries are close to integers or half-integers."""
    def check_matrix(M):
        # M * 2 should be close to integer
        M2 = M * 2
        diff = np.abs(M2 - np.round(M2))
        return np.max(diff) < tol
    
    return check_matrix(U) and check_matrix(V) and check_matrix(W)

def _als_decompose(T, n, m, p, rank, max_iters=50, tol=1e-12, seed=0):
    """
    Alternating Least Squares for 3-way tensor decomposition.
    T: (n, m, p) tensor of 1s.
    Returns U, V, W or None.
    """
    np.random.seed(seed)
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    # Pre-compute target tensor T (all ones)
    T = np.ones((n, m, p))
    
    for it in range(max_iters):
        # Update U
        # For each (i, j), U[i*m+j, :] is solved from:
        # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = T[i,j,k] for all k
        # This is a linear system: U_row @ (V_col @ W_col^T) = T_row
        # Actually, let's flatten.
        # Let M_VW be a matrix of size (m*p) x (p*n) ? No.
        # For fixed i, j:
        # sum_r U[i*m+j, r] * (sum_k V[j*p+k, r] W[k*n+i, r] delta_{k,k'}) = T[i,j,k']
        # This is: U[i*m+j, :] @ (V[j, :, :] * W[:, i, :]).T ?
        # Let B_j_i be a matrix of size (rank, p) where B_j_i[r, k] = V[j*p+k, r] * W[k*n+i, r]
        # Then sum_r U[i*m+j, r] B_j_i[r, k] = T[i, j, k]
        # So U[i*m+j, :] @ B_j_i = T[i, j, :]
        
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                # Construct B_j_i: (rank, p)
                # V[j, :, :] is (p, rank) -> transpose to (rank, p)
                # W[:, i, :] is (p, rank) -> transpose to (rank, p)
                B = (V.reshape(m, p, rank)[j].T * W.reshape(p, n, rank)[:, i].T)
                # Solve U_row @ B = T[i, j, :]
                # U_row = T[i, j, :] @ pinv(B)
                try:
                    U_new[i*m+j, :] = np.linalg.lstsq(B.T, T[i, j, :].T, rcond=None)[0]
                except:
                    return None
        U = U_new
        
        # Update V
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                # Construct C_k_j: (rank, n) ? No.
                # sum_r V[j*p+k, r] U[i*m+j, r] W[k*n+i, r] = T[i, j, k] for all i
                # Let A_i_j be (rank, n) where A_i_j[r, i] = U[i*m+j, r] * W[k*n+i, r]
                # Then V[j*p+k, :] @ A_i_j = T[:, j, k]
                A = (U.reshape(n, m, rank)[:, j, :].T * W.reshape(p, n, rank)[k, :].T)
                # A is (rank, n)
                try:
                    V_new[j*p+k, :] = np.linalg.lstsq(A.T, T[:, j, k].T, rcond=None)[0]
                except:
                    return None
        V = V_new
        
        # Update W
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                # sum_r W[k*n+i, r] U[i*m+j, r] V[j*p+k, r] = T[i, j, k] for all j
                # Let D_j be (rank, m) where D_j[r, j] = U[i*m+j, r] * V[j*p+k, r]
                # Then W[k*n+i, :] @ D_j = T[i, :, k]
                D = (U.reshape(n, m, rank)[i, :, :].T * V.reshape(m, p, rank)[:, k, :].T)
                # D is (rank, m)
                try:
                    W_new[k*n+i, :] = np.linalg.lstsq(D.T, T[i, :, k].T, rcond=None)[0]
                except:
                    return None
        W = W_new
        
        # Check convergence
        if it % 5 == 0:
            if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
                # Check if it's close to integer/half-integer
                if _check_integer_half_integer(U, V, W):
                    # Snap to integer/half-integer
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, tol=1e-9):
                        return U_snap, V_snap, W_snap

    return None

def _schoolbook_decomposition(n, m, p):
    """
    Returns the standard schoolbook decomposition with rank n*m*p.
    """
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # 1. Generate Schoolbook Decomposition
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_U, best_V, best_W = U, V, W
    best_rank = U.shape[1]
    
    # 2. Try to find lower rank decomposition using ALS
    # Try ranks from a reasonable upper bound down to schoolbook
    # We don't know the exact lower bound, so we try a range.
    # Start from 80 down to 50?
    # Each ALS run takes time. Limit to ~20s per rank attempt?
    
    for rank in range(80, 40, -1):
        if time.time() - start_time > DEADLINE_SECONDS - 20:
            break
        
        # Try a few seeds
        for seed in range(5):
            if time.time() - start_time > DEADLINE_SECONDS - 10:
                break
            try:
                res = _als_decompose(None, n, m, p, rank, max_iters=30, seed=seed)
                if res is not None:
                    U_c, V_c, W_c = res
                    if U_c.shape[1] < best_rank:
                        best_U, best_V, best_W = U_c, V_c, W_c
                        best_rank = U_c.shape[1]
                        # If we found a very low rank, we can break early
                        if best_rank <= 50:
                            break
            except Exception:
                pass
        if best_rank <= 50:
            break
            
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
