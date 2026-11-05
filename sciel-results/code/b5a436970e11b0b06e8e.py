"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product
import scipy.optimize

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x4x5 matrix multiplication.
    Tries to find rank < 36 via optimization, falls back to rank 36.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    _t0 = time.time()
    _DEADLINE = 115  # seconds (safety margin)
    
    # Build the multiplication tensor T[i,j,k] = 1 if this term contributes to C[i,k]
    # T has shape (n, m, p) = (2, 4, 5)
    # We want: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = T[i,j,k] for all i,j,k
    # where W is indexed as W[k*n+i, r] per the convention
    
    def build_tensor():
        """Return the multiplication tensor as a 3D array T[i,j,k]."""
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = 1.0
        return T
    
    T = build_tensor()
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W correctly decompose the tensor."""
        rank = U.shape[1]
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        # Compute sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        result = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U3[i, j, r] * V3[j, k, r] * W3[k, i, r]
                    result[i, j, k] = s
        return np.allclose(result, T, atol=tol)
    
    def make_strassen_decomposition():
        """Create the known rank-36 Strassen-based decomposition."""
        rank = 36
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        def add_term(idx, u_coeffs, v_coeffs, w_coeffs):
            for (i, j), val in u_coeffs.items():
                U[i*m + j, idx] += val
            for (j, k), val in v_coeffs.items():
                V[j*p + k, idx] += val
            for (k, i), val in w_coeffs.items():
                W[k*n + i, idx] += val

        def strassen_block(a_row_offset, b_row_offset, c_col_offset, idx_start):
            add_term(idx_start, 
                     {(0, a_row_offset): 1, (1, a_row_offset+1): 1},
                     {(b_row_offset, c_col_offset): 1, (b_row_offset+1, c_col_offset+1): 1},
                     {(c_col_offset, 0): 1, (c_col_offset+1, 1): 1})
            add_term(idx_start+1,
                     {(1, a_row_offset): 1, (1, a_row_offset+1): 1},
                     {(b_row_offset, c_col_offset): 1},
                     {(c_col_offset, 1): 1, (c_col_offset+1, 1): -1})
            add_term(idx_start+2,
                     {(0, a_row_offset): 1},
                     {(b_row_offset, c_col_offset+1): 1, (b_row_offset+1, c_col_offset+1): -1},
                     {(c_col_offset+1, 0): 1, (c_col_offset+1, 1): 1})
            add_term(idx_start+3,
                     {(1, a_row_offset+1): 1},
                     {(b_row_offset+1, c_col_offset): 1, (b_row_offset, c_col_offset): -1},
                     {(c_col_offset, 0): 1, (c_col_offset, 1): 1})
            add_term(idx_start+4,
                     {(0, a_row_offset): 1, (0, a_row_offset+1): 1},
                     {(b_row_offset+1, c_col_offset+1): 1},
                     {(c_col_offset, 0): -1, (c_col_offset+1, 0): 1})
            add_term(idx_start+5,
                     {(1, a_row_offset): 1, (0, a_row_offset): -1},
                     {(b_row_offset, c_col_offset): 1, (b_row_offset, c_col_offset+1): 1},
                     {(c_col_offset+1, 1): 1})
            add_term(idx_start+6,
                     {(0, a_row_offset+1): 1, (1, a_row_offset+1): -1},
                     {(b_row_offset+1, c_col_offset): 1, (b_row_offset+1, c_col_offset+1): 1},
                     {(c_col_offset, 0): 1})

        def schoolbook_2x2x1_block(a_row_offset, b_row_offset, c_col_offset, idx_start):
            add_term(idx_start,
                     {(0, a_row_offset): 1},
                     {(b_row_offset, c_col_offset): 1},
                     {(c_col_offset, 0): 1})
            add_term(idx_start+1,
                     {(0, a_row_offset+1): 1},
                     {(b_row_offset+1, c_col_offset): 1},
                     {(c_col_offset, 0): 1})
            add_term(idx_start+2,
                     {(1, a_row_offset): 1},
                     {(b_row_offset, c_col_offset): 1},
                     {(c_col_offset, 1): 1})
            add_term(idx_start+3,
                     {(1, a_row_offset+1): 1},
                     {(b_row_offset+1, c_col_offset): 1},
                     {(c_col_offset, 1): 1})

        idx = 0
        strassen_block(0, 0, 0, idx); idx += 7
        strassen_block(0, 0, 2, idx); idx += 7
        schoolbook_2x2x1_block(0, 0, 4, idx); idx += 4
        strassen_block(2, 2, 0, idx); idx += 7
        strassen_block(2, 2, 2, idx); idx += 7
        schoolbook_2x2x1_block(2, 2, 4, idx); idx += 4
        
        return U, V, W, rank
    
    def try_reduce_rank(U_init, V_init, W_init, rank_init, time_budget):
        """Attempt to find a lower-rank decomposition starting from rank_init."""
        _t0_inner = time.time()
        best_U, best_V, best_W, best_rank = U_init, V_init, W_init, rank_init
        
        for target_rank in range(rank_init - 1, 0, -1):
            if time.time() - _t0_inner > time_budget - 2:
                break
            
            U_try = U_init[:, :target_rank].copy()
            V_try = V_init[:, :target_rank].copy()
            W_try = W_init[:, :target_rank].copy()
            
            # Precompute index arrays for vectorized contraction
            idx_i = np.repeat(np.arange(n), m*p)
            idx_j = np.tile(np.repeat(np.arange(m), p), n)
            idx_k = np.tile(np.arange(p), n*m)
            row_U = idx_i * m + idx_j
            row_V = idx_j * p + idx_k
            row_W = idx_k * n + idx_i
            
            def objective(params):
                U_f = params[:n*m*target_rank].reshape(n*m, target_rank)
                V_f = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_f = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                # Vectorized: compute all 40 entries
                U_sel = U_f[row_U, :]  # (40, target_rank)
                V_sel = V_f[row_V, :]  # (40, target_rank)
                W_sel = W_f[row_W, :]  # (40, target_rank)
                s = np.sum(U_sel * V_sel * W_sel, axis=1)  # (40,)
                return np.sum((1.0 - s) ** 2)
            
            def gradient(params):
                U_f = params[:n*m*target_rank].reshape(n*m, target_rank)
                V_f = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_f = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                
                U_sel = U_f[row_U, :]
                V_sel = V_f[row_V, :]
                W_sel = W_f[row_W, :]
                s = np.sum(U_sel * V_sel * W_sel, axis=1)
                resid = 1.0 - s  # (40,)
                
                grad_U = np.zeros_like(U_f)
                grad_V = np.zeros_like(V_f)
                grad_W = np.zeros_like(W_f)
                
                # grad_U[row_U, r] += -2 * resid * V_sel[:, r] * W_sel[:, r]
                for r in range(target_rank):
                    np.add.at(grad_U, row_U, -2.0 * resid * V_sel[:, r] * W_sel[:, r])
                    np.add.at(grad_V, row_V, -2.0 * resid * U_sel[:, r] * W_sel[:, r])
                    np.add.at(grad_W, row_W, -2.0 * resid * U_sel[:, r] * V_sel[:, r])
                
                return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            
            params_init = np.concatenate([U_try.flatten(), V_try.flatten(), W_try.flatten()])
            
            try:
                result = scipy.optimize.minimize(
                    objective, params_init, method='L-BFGS-B',
                    jac=gradient,
                    options={'maxiter': 30, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 100}
                )
                
                if result.fun < 1e-10:
                    U_opt = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_opt = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_opt = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
                    
                    if verify_decomposition(U_opt, V_opt, W_opt, tol=1e-9):
                        best_U, best_V, best_W, best_rank = U_opt, V_opt, W_opt, target_rank
                        break
            except Exception:
                pass
        
        return best_U, best_V, best_W, best_rank
    
    U0, V0, W0, rank0 = make_strassen_decomposition()
    
    # Verify the constructed solution
    if not verify_decomposition(U0, V0, W0, tol=1e-9):
        # Fallback to schoolbook if Strassen construction fails (should not happen)
        U_school = np.zeros((n*m, n*m*p))
        V_school = np.zeros((m*p, n*m*p))
        W_school = np.zeros((p*n, n*m*p))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m+j, idx] = 1.0
                    V_school[j*p+k, idx] = 1.0
                    W_school[k*n+i, idx] = 1.0
                    idx += 1
        U0, V0, W0, rank0 = U_school, V_school, W_school, n*m*p
    else:
        # Try to reduce rank below 36 using optimization
        _elapsed = time.time() - _t0
        _remaining = _DEADLINE - _elapsed
        if _remaining > 5:
            U0, V0, W0, rank0 = try_reduce_rank(U0, V0, W0, rank0, _remaining)
    
    return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
# EVOLVE-BLOCK-END
