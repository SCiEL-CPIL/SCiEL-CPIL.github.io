"""Model-and-optimize approach for 2x4x7 matrix multiplication tensor decomposition."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [2, 4, 7]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 @ 4x7.
    
    Strategy: Build rank-50 Strassen-based decomposition, then attempt
    rank reduction via linear dependency detection with fast verification.
    """
    n, m, p = 2, 4, 7
    deadline = time.time() + 130  # safety margin before 150s limit
    
    # Build the rank-50 decomposition as starting point
    rank_init = 50
    U = np.zeros((n * m, rank_init), dtype=float)
    V = np.zeros((m * p, rank_init), dtype=float)
    W = np.zeros((p * n, rank_init), dtype=float)
    
    col = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1: (A11+A22)(B11+B22)
        U[i0*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M2: (A21+A22)B11
        U[(i0+1)*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        W[k0*n + (i0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] -= 1
        col += 1
        
        # M3: A11(B12-B22)
        U[i0*m + j0, col] += 1
        V[j0*p + (k0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] -= 1
        W[(k0+1)*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M4: A22(B21-B11)
        U[(i0+1)*m + (j0+1), col] += 1
        V[(j0+1)*p + k0, col] += 1
        V[j0*p + k0, col] -= 1
        W[k0*n + i0, col] += 1
        W[k0*n + (i0+1), col] += 1
        col += 1
        
        # M5: (A11+A12)B22
        U[i0*m + j0, col] += 1
        U[i0*m + (j0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] -= 1
        W[(k0+1)*n + i0, col] += 1
        col += 1
        
        # M6: (A21-A11)(B11+B12)
        U[(i0+1)*m + j0, col] += 1
        U[i0*m + j0, col] -= 1
        V[j0*p + k0, col] += 1
        V[j0*p + (k0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M7: (A12-A22)(B21+B22)
        U[i0*m + (j0+1), col] += 1
        U[(i0+1)*m + (j0+1), col] -= 1
        V[(j0+1)*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        col += 1

    # 6 Strassen blocks for the 2x4x6 part (k=0..5)
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 4)
    
    # 8 rank-1 terms for k=6
    for i in range(2):
        for j in range(4):
            U[i*m + j, col] = 1
            V[j*p + 6, col] = 1
            W[6*n + i, col] = 1
            col += 1
    
    assert col == rank_init, f"Expected rank {rank_init}, got {col}"
    
    def verify_fast(U_m, V_m, W_m, n, m, p, num_tests=10):
        """Fast verification using vectorized tensor operations."""
        np.random.seed(42)
        for _ in range(num_tests):
            A = np.random.randint(-3, 4, size=(n, m)).astype(float)
            B = np.random.randint(-3, 4, size=(m, p)).astype(float)
            C_expected = A @ B
            
            # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # Vectorize: for each r, contribution is (u_r reshaped) * (v_r reshaped) * w_r
            # u_r is (n*m,), v_r is (m*p,), w_r is (p*n,)
            # C_contrib_r[i,k] = sum_j U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            
            # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
            U_r = U_m.reshape(n, m, -1)  # (n, m, rank)
            V_r = V_m.reshape(m, p, -1)  # (m, p, rank)
            W_r = W_m.reshape(p, n, -1)  # (p, n, rank)
            
            # C[i,k] = sum_j sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
            # = sum_r sum_j U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
            
            # For each r: term_r[i,k] = sum_j U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
            # Use einsum: 'ijr,jkr,kir->ik'
            C_comp = np.einsum('ijr,jkr,kir->ik', U_r, V_r, W_r, optimize=True)
            
            if not np.allclose(C_comp, C_expected, atol=1e-9):
                return False
        return True
    
    # Verify initial decomposition
    if not verify_fast(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_init
    
    # Try greedy rank reduction: check if any column is in the span of others
    # across all three matrices simultaneously
    def try_reduce_rank(U_arr, V_arr, W_arr, n, m, p):
        """Attempt to reduce rank by eliminating linearly dependent columns."""
        U_arr = U_arr.copy()
        V_arr = V_arr.copy()
        W_arr = W_arr.copy()
        current_rank = U_arr.shape[1]
        
        eliminated = set()
        for iteration in range(50):
            if time.time() > deadline - 20:
                break
            found_elim = False
            for c in range(current_rank):
                if c in eliminated:
                    continue
                other_cols = [oc for oc in range(current_rank) if oc != c and oc not in eliminated]
                if len(other_cols) < 1:
                    continue
                
                # Stack columns from all three matrices
                U_sub = U_arr[:, other_cols]  # (n*m, len(other))
                V_sub = V_arr[:, other_cols]  # (m*p, len(other))
                W_sub = W_arr[:, other_cols]  # (p*n, len(other))
                
                u_target = U_arr[:, c]
                v_target = V_arr[:, c]
                w_target = W_arr[:, c]
                
                # Least squares: minimize ||[U_sub; V_sub; W_sub] @ x - [u_t; v_t; w_t]||
                stacked = np.vstack([U_sub, V_sub, W_sub])  # (n*m+m*p+p*n, len(other))
                target = np.concatenate([u_target, v_target, w_target])
                
                try:
                    coeffs, residuals, rank_s, sv = np.linalg.lstsq(stacked, target, rcond=1e-12)
                    recon = stacked @ coeffs
                    if np.allclose(recon, target, atol=1e-10):
                        eliminated.add(c)
                        found_elim = True
                        break
                except:
                    continue
            
            if not found_elim:
                break
        
        keep_cols = [c for c in range(current_rank) if c not in eliminated]
        new_rank = len(keep_cols)
        if new_rank < current_rank:
            return U_arr[:, keep_cols], V_arr[:, keep_cols], W_arr[:, keep_cols], new_rank
        return U_arr, V_arr, W_arr, current_rank
    
    # Attempt reduction
    if time.time() < deadline - 30:
        U_red, V_red, W_red, new_rank = try_reduce_rank(U, V, W, n, m, p)
        if new_rank < best_rank and verify_fast(U_red, V_red, W_red, n, m, p):
            best_U, best_V, best_W = U_red, V_red, W_red
            best_rank = new_rank
    
    # Try a second round of reduction on the reduced result
    if best_rank < rank_init and time.time() < deadline - 15:
        U_red2, V_red2, W_red2, new_rank2 = try_reduce_rank(best_U, best_V, best_W, n, m, p)
        if new_rank2 < best_rank and verify_fast(U_red2, V_red2, W_red2, n, m, p):
            best_U, best_V, best_W = U_red2, V_red2, W_red2
            best_rank = new_rank2
    
    # Try alternative: schoolbook rank-56 decomposition as baseline comparison
    # Actually rank-50 is already better than 56, so skip
    
    # Final verification
    if not verify_fast(best_U, best_V, best_W, n, m, p):
        best_U, best_V, best_W = U, V, W
        best_rank = rank_init
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
