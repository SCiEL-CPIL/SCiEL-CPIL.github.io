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
    Primary: rank-36 Strassen+schoolbook block construction (known exact).
    Secondary: attempt rank reduction via structured integer/half-integer search
    using the bilinear constraint structure and rank-deficiency testing.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    _t0 = time.time()
    _DEADLINE = 115  # seconds (safety margin before 120s kill)
    
    def _elapsed():
        return time.time() - _t0
    
    # Build the multiplication tensor T[i,j,k] = 1 for all i,j,k
    # We need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    def build_tensor():
        T = np.ones((n, m, p))
        return T
    
    T = build_tensor()
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W correctly decompose the tensor."""
        rank = U.shape[1]
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
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
    
    U0, V0, W0, rank0 = make_strassen_decomposition()
    best_U, best_V, best_W, best_rank = U0, V0, W0, rank0
    
    # Try to reduce rank by optimizing the decomposition
    def try_rank_reduction(target_rank, U_init, V_init, W_init, max_iter=200):
        """Try to find a decomposition of target_rank by optimizing from initialization."""
        if _elapsed() > _DEADLINE - 5:
            return None
        
        n_vars = n*m*target_rank + m*p*target_rank + p*n*target_rank
        params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def objective(params):
            r = target_rank
            U_p = params[:n*m*r].reshape(n*m, r)
            V_p = params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
            W_p = params[n*m*r + m*p*r:].reshape(p*n, r)
            
            U3 = U_p.reshape(n, m, r)
            V3 = V_p.reshape(m, p, r)
            W3 = W_p.reshape(p, n, r)
            
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for rr in range(r):
                            s += U3[i, j, rr] * V3[j, k, rr] * W3[k, i, rr]
                        err += (s - 1.0) ** 2
            return err
        
        try:
            result = scipy.optimize.minimize(
                objective, params0, method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.fun < 1e-8:
                r = target_rank
                U_sol = result.x[:n*m*r].reshape(n*m, r)
                V_sol = result.x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W_sol = result.x[n*m*r + m*p*r:].reshape(p*n, r)
                if verify_decomposition(U_sol, V_sol, W_sol, tol=1e-6):
                    return (U_sol, V_sol, W_sol, target_rank)
        except Exception:
            pass
        return None
    
    # Try to reduce from rank 36 to rank 35
    if _elapsed() < _DEADLINE - 20:
        U_init_35 = U0[:, :35]
        V_init_35 = V0[:, :35]
        W_init_35 = W0[:, :35]
        
        result = try_rank_reduction(35, U_init_35, V_init_35, W_init_35, max_iter=100)
        if result is not None:
            U, V, W, rank = result
            best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Try rank 34
    if _elapsed() < _DEADLINE - 20:
        U_init_34 = U0[:, :34]
        V_init_34 = V0[:, :34]
        W_init_34 = W0[:, :34]
        
        result = try_rank_reduction(34, U_init_34, V_init_34, W_init_34, max_iter=100)
        if result is not None:
            U, V, W, rank = result
            best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Deadline guard
    if _elapsed() > _DEADLINE - 5:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
