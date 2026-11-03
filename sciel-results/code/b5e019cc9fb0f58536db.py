import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [4, 4, 5]

def _verify_fast(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = 4, 4, 5
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the tensor T[i,j,k] = delta_{i,k} for C[i,k] = sum_j A[i,j]*B[j,k]
    # In the decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Efficient verification using einsum
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # Compute sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
    # This should equal 1 if i==k, 0 otherwise
    result = np.einsum('ijr,jkr,kir->ik', U_r, V_r, W_r)
    
    # Expected: identity matrix of size n x p? No.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] should be 1 if this is the "identity" contribution, 0 otherwise.
    # Actually, the tensor T[i,j,k] = delta_{i,k} is not right.
    # The matrix multiplication tensor has T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k].
    # So T[i,j,k] = 1 for all valid (i,j,k), and the decomposition must satisfy:
    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
    
    # Wait, let me reconsider. The tensor for matrix multiplication M_{n,m,p} is:
    # T_{ijk} = 1 if the output entry (i,k) depends on input entries (i,j) and (j,k)
    # In standard form, T_{ijk} = delta_{i,i'} * delta_{j,j'} * delta_{k,k'} where we're just saying
    # the trilinear form is sum_{i,j,k} A_{ij} B_{jk} C_{ki} = 0 for the identity check.
    
    # Actually, the correct tensor is: T[i,j,k] = 1 for all i,j,k in valid range.
    # No wait. Let me think again.
    
    # The matrix multiplication tensor M_{n,m,p} is defined such that:
    # <M, A \otimes B \otimes C> = trace(A B C^T) = sum_{i,j,k} A_{ij} B_{jk} C_{ki}
    # So M_{ijk} = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    # A rank-1 term is u \otimes v \otimes w where:
    # <u \otimes v \otimes w, A \otimes B \otimes C> = (sum_{ij} u_{ij} A_{ij})(sum_{jk} v_{jk} B_{jk})(sum_{ki} w_{ki} C_{ki})
    # = sum_{ijk} u_{ij} v_{jk} w_{ki} A_{ij} B_{jk} C_{ki}
    
    # So we need: sum_r U_{ij,r} V_{jk,r} W_{ki,r} = 1 for all i,j,k
    
    # My einsum above computes exactly this (with W indexed as W[k,i,r] = W_r[k,i,r])
    # And it should equal 1 for all i,j,k.
    
    expected = np.ones((n, p))  # This is wrong, let me reconsider
    
    # Actually result[i,k] = sum_j sum_r U_r[i,j,r] V_r[j,k,r] W_r[k,i,r]
    # But we need sum_r U_r[i,j,r] V_r[j,k,r] W_r[k,i,r] = 1 for each (i,j,k)
    # The result I computed sums over j as well, which is not what we want.
    
    # Let me redo: we need the full 3D tensor to match.
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Build T as a 3D array
    T = np.zeros((n, m, p))
    for r in range(rank):
        T += U_r[:, :, r] * V_r[:, :, r][np.newaxis, :, :] * W_r[:, :, r][:, np.newaxis, :]
        # Wait, this indexing is getting confused. Let me be explicit.
    
    # U_r has shape (n, m, rank) -> U_r[i, j, r]
    # V_r has shape (m, p, rank) -> V_r[j, k, r]  
    # W_r has shape (p, n, rank) -> W_r[k, i, r]
    
    # T[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
    
    # Use einsum properly:
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    
    # T should be all ones
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=1e-6)

def _compute_error(U, V, W):
    """Compute the squared error of the decomposition."""
    n, m, p = 4, 4, 5
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    expected = np.ones((n, m, p))
    return np.sum((T - expected) ** 2)

def _round_to_half_integer(x):
    """Round a value to the nearest half-integer."""
    return 2.0 * np.round(x * 0.5) * 0.5

def _round_matrix_to_half_int(M):
    """Round all entries of a matrix to half-integers."""
    return _round_to_half_integer(np.array(M, dtype=float))

def _strassen_2x2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is (a_coeffs, b_coeffs, c_coeffs) where each is a 4-tuple
    for the 4 entries of the respective 2x2 matrix."""
    # S1 = (A00+A11)(B00+B11)
    # S2 = (A10+A11)B00
    # S3 = A00(B01-B11)
    # S4 = A11(B10-B00)
    # S5 = (A00+A01)B11
    # S6 = (A10-A00)(B00+B01)
    # S7 = (A01-A11)(B10+B11)
    
    # C00 = S1 - S2 + S4 - S5
    # C01 = S3 + S5
    # C10 = S2 + S7
    # C11 = S1 - S3 + S6 - S7
    
    terms = [
        # (a00,a01,a10,a11, b00,b01,b10,b11, c00,c01,c10,c11)
        ((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1)),   # S1
        ((0, 0, 1, 1), (1, 0, 0, 0), (-1, 0, 1, 0)),   # S2
        ((1, 0, 0, 0), (0, 1, 0, -1), (0, 1, 0, -1)),  # S3
        ((0, 0, 0, 1), (-1, 0, 1, 0), (1, 0, 0, 0)),   # S4
        ((1, 1, 0, 0), (0, 0, 0, 1), (-1, 1, 0, 0)),   # S5
        ((-1, 0, 1, 0), (1, 1, 0, 0), (0, 0, 0, -1)),  # S6
        ((0, 1, 0, -1), (0, 0, 1, 1), (0, 0, 1, -1)),  # S7
    ]
    return terms

def _kron_strassen_construct():
    """Construct a rank-49 decomposition of 4x4x4 using Kronecker product of Strassen 2x2x2.
    Then add 8 more terms for the 5th column (schoolbook).
    Total rank: 57."""
    n, m, p = 4, 4, 5
    strassen = _strassen_2x2x2_terms()
    
    # For 4x4x4, we use Kronecker: the 4x4x4 tensor = kron(2x2x2, 2x2x2)
    # This gives rank 7*7 = 49
    rank_4x4 = 49
    rank_total = rank_4x4 + 8  # 8 for the 5th column
    
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    # Kronecker Strassen for 4x4x4 (first 4 columns of B)
    for r1 in range(7):
        for r2 in range(7):
            col = r1 * 7 + r2
            a1, b1, c1 = strassen[r1]
            a2, b2, c2 = strassen[r2]
            
            # The Kronecker product: A = kron(A1, A2), B = kron(B1, B2), C = kron(C1, C2)
            # where A1, A2 are 2x2 blocks of the 4x4 matrix
            # A[i,j] with i = i1*2+i2, j = j1*2+j2
            # A[i,j] = A1[i1,j1] * A2[i2,j2]
            
            # For U: index is i*m+j = (i1*2+i2)*4 + (j1*2+j2)
            # For V: index is j*p+k = (j1*2+j2)*5 + (k1*2+k2)
            # For W: index is k*n+i = (k1*2+k2)*4 + (i1*2+i2)
            
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            a_val = (a1[i1*2+j1] if i1 == 0 and j1 == 0 else 0) + \
                                    (a1[i1*2+j1] if i1 == 1 and j1 == 0 else 0) + \
                                    (a1[i1*2+j1] if i1 == 0 and j1 == 1 else 0) + \
                                    (a1[i1*2+j1] if i1 == 1 and j1 == 1 else 0)
                            # This is getting complicated. Let me use a cleaner approach.
                            pass
            
            # Cleaner: use the Kronecker structure directly
            # A[i,j] = A1[i//2, j//2] * A2[i%2, j%2]
            # U[i*m+j, col] = A1[i//2, j//2] * A2[i%2, j%2]
            
            for i in range(4):
                for j in range(4):
                    i1, i2 = i // 2, i % 2
                    j1, j2 = j // 2, j % 2
                    U[i * 4 + j, col] = a1[i1 * 2 + j1] * a2[i2 * 2 + j2]
            
            for j in range(4):
                for k in range(4):
                    j1, j2 = j // 2, j % 2
                    k1, k2 = k // 2, k % 2
                    V[j * 5 + k, col] = b1[j1 * 2 + k1] * b2[j2 * 2 + k2]
            
            for k in range(4):
                for i in range(4):
                    k1, k2 = k // 2, k % 2
                    i1, i2 = i // 2, i % 2
                    W[k * 4 + i, col] = c1[k1 * 2 + i1] * c2[k2 * 2 + i2]
    
    # Add schoolbook terms for the 5th column (k=4)
    for i in range(n):
        for j in range(m):
            col = rank_4x4 + i * m + j
            U[i * m + j, col] = 1
            V[j * p + 4, col] = 1
            W[4 * n + i, col] = 1
    
    return U, V, W

def _als_decompose(U_init, V_init, W_init, max_iter=200, tol=1e-10, time_limit=10):
    """ALS decomposition to reduce rank by zeroing out columns with small norm."""
    n, m, p = 4, 4, 5
    U = np.array(U_init, dtype=float)
    V = np.array(V_init, dtype=float)
    W = np.array(W_init, dtype=float)
    rank = U.shape[1]
    
    start = time.time()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = _compute_error(U, V, W)
    
    for iteration in range(max_iter):
        if time.time() - start > time_limit:
            break
        
        # Try to remove columns with smallest contribution
        # Compute the norm of each column's contribution
        col_norms = np.zeros(rank)
        for r in range(rank):
            # Temporarily zero out column r and compute error increase
            U_test = U.copy()
            V_test = V.copy()
            W_test = W.copy()
            U_test[:, r] = 0
            V_test[:, r] = 0
            W_test[:, r] = 0
            err_test = _compute_error(U_test, V_test, W_test)
            col_norms[r] = err_test - best_err
        
        # Find the column with smallest norm (easiest to remove)
        if np.all(col_norms > 0.01):
            break
        
        min_idx = np.argmin(col_norms)
        
        # Try removing this column and re-optimizing
        U_new = np.delete(U, min_idx, axis=1)
        V_new = np.delete(V, min_idx, axis=1)
        W_new = np.delete(W, min_idx, axis=1)
        
        # Optimize the remaining columns using gradient descent
        # Flatten parameters
        def objective(params):
            U_opt = params[:n*m*U_new.shape[1]].reshape(n*m, U_new.shape[1])
            V_opt = params[n*m*U_new.shape[1]:n*m*U_new.shape[1]+m*p*U_new.shape[1]].reshape(m*p, U_new.shape[1])
            W_opt = params[n*m*U_new.shape[1]+m*p*U_new.shape[1]:].reshape(p*n, U_new.shape[1])
            return _compute_error(U_opt, V_opt, W_opt)
        
        params = np.concatenate([U_new.flatten(), V_new.flatten(), W_new.flatten()])
        
        # Use Nelder-Mead for small problems
        result = minimize(objective, params, method='Nelder-Mead', 
                         options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if result.fun < best_err:
            U = result.x[:n*m*U_new.shape[1]].reshape(n*m, U_new.shape[1])
            V = result.x[n*m*U_new.shape[1]:n*m*U_new.shape[1]+m*p*U_new.shape[1]].reshape(m*p, U_new.shape[1])
            W = result.x[n*m*U_new.shape[1]+m*p*U_new.shape[1]:].reshape(p*n, U_new.shape[1])
            best_err = result.fun
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            rank = U.shape[1]
        else:
            # Can't remove this column, try next smallest
            col_norms[min_idx] = np.inf
            if np.all(col_norms > 0.01):
                break
    
    return best_U, best_V, best_W, best_err

def _schoolbook():
    """Return the schoolbook decomposition with rank 80."""
    n, m, p = 4, 4, 5
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = 140  # seconds, leave 10s margin
    
    # Start with Kronecker Strassen baseline (rank 57)
    U, V, W = _kron_strassen_construct()
    err = _compute_error(U, V, W)
    
    if not _verify_fast(U, V, W):
        # Fallback to schoolbook if baseline is invalid
        U, V, W = _schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    # Try ALS-based rank reduction
    if time.time() - start_time < deadline - 30:
        U_red, V_red, W_red, err_red = _als_decompose(U, V, W, max_iter=100, time_limit=80)
        if U_red.shape[1] < best_rank and _verify_fast(U_red, V_red, W_red):
            best_U, best_V, best_W = U_red, V_red, W_red
            best_rank = U_red.shape[1]
    
    # Try additional random restarts if time permits
    if time.time() - start_time < deadline - 20:
        for restart in range(3):
            if time.time() - start_time > deadline - 10:
                break
            
            # Perturb the baseline
            U_pert = best_U + 0.1 * np.random.randn(*best_U.shape)
            V_pert = best_V + 0.1 * np.random.randn(*best_V.shape)
            W_pert = best_W + 0.1 * np.random.randn(*best_W.shape)
            
            U_r, V_r, W_r, err_r = _als_decompose(U_pert, V_pert, W_pert, max_iter=50, time_limit=15)
            if U_r.shape[1] < best_rank and _verify_fast(U_r, V_r, W_r):
                best_U, best_V, best_W = U_r, V_r, W_r
                best_rank = U_r.shape[1]
    
    # Round to half-integers and verify
    U_final = _round_matrix_to_half_int(best_U)
    V_final = _round_matrix_to_half_int(best_V)
    W_final = _round_matrix_to_half_int(best_W)
    
    if _verify_fast(U_final, V_final, W_final):
        return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
    
    # If rounding broke it, try without rounding
    if _verify_fast(best_U, best_V, best_W):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
