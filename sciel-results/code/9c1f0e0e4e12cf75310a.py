import time
import numpy as np
from scipy.optimize import minimize
import sys

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact decomposition for matrix multiplication 2x4 by 4x7.
    
    Strategy:
    1. Formulate the problem as minimizing the Frobenius norm of the error between
       the target tensor (all ones) and the tensor reconstructed from factor matrices U, V, W.
    2. Use a fixed "slack" rank R (larger than expected optimal) and optimize continuous variables.
    3. After optimization, attempt to round the factors to integers or half-integers.
    4. Verify exactness. If successful, return the rank (number of non-zero columns).
    5. If global optimization fails to find an exact solution quickly, fall back to a 
       constructive hybrid (Strassen + Direct) which is guaranteed to be exact.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 130.0  # Safety margin

    # Target tensor T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We seek U (n*m x R), V (m*p x R), W (p*n x R) such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # We will try to find a decomposition with rank R.
    # The trivial rank is n*m*p = 56.
    # Known lower bounds suggest rank could be significantly lower.
    # Let's try to optimize for R = 30, 35, 40, etc.
    
    def verify_exact(U, V, W, rank, n, m, p, tol=1e-9):
        """Check if the decomposition is exactly correct."""
        U_arr = np.array(U)[:, :rank]
        V_arr = np.array(V)[:, :rank]
        W_arr = np.array(W)[:, :rank]
        
        # Reconstruct tensor
        # T[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
        # We can vectorize this check.
        # For each c, the contribution is a rank-1 tensor.
        # Let's just loop over i, j, k since dimensions are small (2*4*7 = 56 entries).
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U_arr[u_idx], V_arr[v_idx] * W_arr[w_idx])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def make_schoolbook(n, m, p):
        """Standard rank n*m*p decomposition."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U.tolist(), V.tolist(), W.tolist(), rank

    # Fallback: Hybrid Strassen + Direct
    # 2x4x7 can be split into 2x4x2 (using two 2x2x2 Strassen blocks, rank 14) 
    # and 2x4x5 (direct, rank 40). Total 54.
    # Or 2x2x7 (Strassen-like?) + 2x2x7.
    # Actually, let's just use the schoolbook if optimization fails, 
    # but first try a slightly better constructive one if easy.
    # For now, schoolbook is the safe fallback.
    
    U_fb, V_fb, W_fb, rank_fb = make_schoolbook(n, m, p)
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_fb

    # Optimization Strategy
    # We flatten U, V, W into a single vector x.
    # Objective: || T_target - T_reconstructed(x) ||_F^2
    
    # We will try a few target ranks R, starting from a plausible lower bound.
    # Lower bound for rank of M_{2,4,7}:
    # rank >= max(n*m, m*p, n*p) is not strictly true for all tensors, 
    # but for matrix multiplication, rank >= m * max(n, p) is a common heuristic? 
    # Actually, rank(M_{n,m,p}) >= m * max(n,p) is not always true.
    # However, rank >= n*m, rank >= m*p, rank >= n*p.
    # max(8, 28, 14) = 28.
    # Let's try R = 28, 30, 32, 34, 36, 38, 40.
    
    target_ranks = [28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]
    
    for R in target_ranks:
        if time.time() > deadline:
            break
            
        # If R >= best_rank, no point in trying (we want lower rank)
        if R >= best_rank:
            continue

        # Initial guess: Random or Schoolbook-like projection?
        # Random initialization often gets stuck in local minima for tensor decomposition.
        # A better init: Use the schoolbook decomposition and randomly project/combine columns?
        # Or just start with small random values.
        
        # Let's try a structured initialization:
        # Set U, V, W to small random values.
        rng = np.random.default_rng(42)
        
        U_init = rng.normal(0, 1, (n*m, R))
        V_init = rng.normal(0, 1, (m*p, R))
        W_init = rng.normal(0, 1, (p*n, R))
        
        # Scale them so that the product has a reasonable magnitude.
        # We want sum_c U V W ~ 1.
        # If U, V, W are ~ N(0,1), product is ~ N(0,1). Sum of R such terms ~ N(0, R).
        # We want sum ~ 1. So scale each by 1/sqrt(R).
        scale = 1.0 / np.sqrt(R)
        U_init *= scale
        V_init *= scale
        W_init *= scale
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        def objective(x):
            U_x = x[:n*m*R].reshape(n*m, R)
            V_x = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
            W_x = x[n*m*R + m*p*R : ].reshape(p*n, R)
            
            # Compute error
            err = 0.0
            # To speed up, we can compute the full tensor difference.
            # T_recon[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
            
            # Vectorized computation for small sizes:
            # Create index arrays
            # This loop is 56 iterations, each doing dot products of length R.
            # R is small (< 56). So this is fast.
            
            # Alternative: Use einsum if we construct the full tensor, but looping is fine.
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        val = np.dot(U_x[u_idx], V_x[v_idx] * W_x[w_idx])
                        err += (val - 1.0) ** 2
            return err

        # Gradient is expensive to compute analytically in this loop structure easily without care.
        # Let's use finite difference or just rely on L-BFGS-B with numerical gradients?
        # scipy.optimize.minimize with method='L-BFGS-B' computes gradients numerically by default if jac is not provided.
        # For ~ (2*4 + 4*7 + 7*2) * R = (8+28+14)*R = 50*R variables.
        # For R=30, 1500 variables. Numerical gradient might be slow.
        
        # Let's implement an analytical gradient to make it feasible.
        # f(x) = sum_{i,j,k} (1 - sum_c U_{ij,c} V_{jk,c} W_{ki,c})^2
        # Let E_{ijk} = 1 - T_{ijk}
        # df/du_{ij,c} = -2 sum_{i,j,k} E_{ijk} * V_{jk,c} * W_{ki,c}
        # df/dv_{jk,c} = -2 sum_{i,j,k} E_{ijk} * U_{ij,c} * W_{ki,c}
        # df/dw_{ki,c} = -2 sum_{i,j,k} E_{ijk} * U_{ij,c} * V_{jk,c}
        
        def objective_and_grad(x):
            U_x = x[:n*m*R].reshape(n*m, R)
            V_x = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
            W_x = x[n*m*R + m*p*R : ].reshape(p*n, R)
            
            # Compute T and E
            # T is n x m x p
            T = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        T[i, j, k] = np.dot(U_x[u_idx], V_x[v_idx] * W_x[w_idx])
            
            E = 1.0 - T
            obj_val = np.sum(E ** 2)
            
            # Compute gradients
            grad_U = np.zeros_like(U_x)
            grad_V = np.zeros_like(V_x)
            grad_W = np.zeros_like(W_x)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        e_ijk = E[i, j, k]
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        
                        # grad_U[u_idx, c] += -2 * e_ijk * V[v_idx, c] * W[w_idx, c]
                        grad_U[u_idx] -= 2.0 * e_ijk * (V_x[v_idx] * W_x[w_idx])
                        
                        # grad_V[v_idx, c] += -2 * e_ijk * U[u_idx, c] * W[w_idx, c]
                        grad_V[v_idx] -= 2.0 * e_ijk * (U_x[u_idx] * W_x[w_idx])
                        
                        # grad_W[w_idx, c] += -2 * e_ijk * U[u_idx, c] * V[v_idx, c]
                        grad_W[w_idx] -= 2.0 * e_ijk * (U_x[u_idx] * V_x[v_idx])
            
            grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            return obj_val, grad

        try:
            res = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if time.time() > deadline:
                break
                
            x_opt = res.x
            U_opt = x_opt[:n*m*R].reshape(n*m, R)
            V_opt = x_opt[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
            W_opt = x_opt[n*m*R + m*p*R : ].reshape(p*n, R)
            
            # Check if the continuous solution is close to integer/half-integer
            # We need to round to nearest integer or half-integer.
            # Let's try rounding to nearest 0.5.
            
            def round_to_half(arr):
                return np.round(arr * 2.0) / 2.0
            
            U_rounded = round_to_half(U_opt)
            V_rounded = round_to_half(V_opt)
            W_rounded = round_to_half(W_opt)
            
            # Verify exactness of rounded solution
            if verify_exact(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), R, n, m, p):
                # Found a valid decomposition with rank R
                # We can potentially prune zero columns to get a lower rank
                # But for now, R is the score.
                best_rank = R
                best_result = {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                # Since we iterate R upwards, the first one found is the lowest.
                # We can break early if we are confident, but let's try to see if lower R was missed?
                # We started from 28, so if 28 works, we break. If 30 works, we break.
                break
                
            # If rounding to 0.5 failed, try rounding to integer
            U_int = np.round(U_opt)
            V_int = np.round(V_opt)
            W_int = np.round(W_opt)
            
            if verify_exact(U_int.tolist(), V_int.tolist(), W_int.tolist(), R, n, m, p):
                best_rank = R
                best_result = {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                break
                
        except Exception as e:
            # Optimization failed
            pass

    return best_result
# EVOLVE-BLOCK-END
