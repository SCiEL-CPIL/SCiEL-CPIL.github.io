"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast vectorized verification using einsum."""
    try:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
    except Exception:
        return False, float('inf')
    
    if U_arr.size == 0: return False, float('inf')
    rank = U_arr.shape[1]
    
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    # The target tensor is 1.0 where i,j,k exist (which is all of them)
    # Wait, the tensor T[i,j,k] corresponds to the coefficient of A[i,j]B[j,k] in C[i,k].
    # The standard matrix multiplication tensor has T[i,j,k] = 1 if we define the tensor such that
    # C[i,k] = sum_{j} A[i,j] B[j,k].
    # The decomposition is C[i,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]? No.
    # The standard form is: C[i,k] = sum_{j,c} A[i,j] B[j,k] (tensor coeff).
    # The tensor is $T_{ijk}$. The multiplication is $C_{ik} = \sum_{j} T_{ijk} A_{ij} B_{jk}$.
    # So $T_{ijk} = 1$ for all valid i,j,k.
    
    max_err = np.max(np.abs(T - 1.0))
    return max_err < tol, max_err


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition exactly reconstructs the matrix multiplication tensor."""
    ok, _ = _verify_fast(U, V, W, n, m, p, tol)
    return ok


def _objective_and_grad(x, n, m, p, r):
    """Compute objective and gradient for L-BFGS."""
    U = x[:n*m*r].reshape(n, m, r)
    V = x[n*m*r:2*n*m*r].reshape(m, p, r)
    W = x[2*n*m*r:].reshape(p, n, r)
    
    T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
    residual = T - 1.0
    obj = np.sum(residual**2)
    
    # Gradients
    # dU[i,j,c] = sum_{k} residual[i,j,k] * V[j,k,c] * W[k,i,c]
    dU = 2 * np.einsum('ijk,jkc,kic->ijc', residual, V, W)
    # dV[j,k,c] = sum_{i} residual[i,j,k] * U[i,j,c] * W[k,i,c]
    dV = 2 * np.einsum('ijk,ijc,kic->jkc', residual, U, W)
    # dW[k,i,c] = sum_{j} residual[i,j,k] * U[i,j,c] * V[j,k,c]
    dW = 2 * np.einsum('ijk,ijc,jkc->kic', residual, U, V)
    
    grad = np.concatenate([dU.ravel(), dV.ravel(), dW.ravel()])
    return obj, grad


def _lbfgs_solve(n, m, p, r, max_time=10, seed=42):
    """Use L-BFGS-B to find a rank-r decomposition."""
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    # Try multiple initializations
    best_res = None
    best_err = float('inf')
    
    for init_type in ['random', 'structured', 'sparse', 'gaussian_scaled']:
        if time.time() - start_time > max_time:
            break
        
        if init_type == 'random':
            U = rng.randn(n * m, r) * 0.5
            V = rng.randn(m * p, r) * 0.5
            W = rng.randn(p * n, r) * 0.5
        elif init_type == 'structured':
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            # Initialize with a subset of schoolbook terms
            for c in range(min(r, n*m*p)):
                i = c % n
                j = (c // n) % m
                k = (c // (n*m)) % p
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
            U += rng.randn(n*m, r) * 0.1
            V += rng.randn(m*p, r) * 0.1
            W += rng.randn(p*n, r) * 0.1
        elif init_type == 'sparse':
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            for c in range(r):
                idx = rng.randint(0, n*m-1)
                U[idx, c] = 1.0
                idx = rng.randint(0, m*p-1)
                V[idx, c] = 1.0
                idx = rng.randint(0, p*n-1)
                W[idx, c] = 1.0
            U += rng.randn(n*m, r) * 0.05
            V += rng.randn(m*p, r) * 0.05
            W += rng.randn(p*n, r) * 0.05
        else: # gaussian_scaled
             U = rng.randn(n * m, r) * 0.1
             V = rng.randn(m * p, r) * 0.1
             W = rng.randn(p * n, r) * 0.1

        x0 = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
        
        class TimeLimitExceeded(Exception):
            pass
        
        def callback(xk):
            if time.time() - start_time > max_time - 2:
                raise TimeLimitExceeded()
            return False
        
        try:
            result = minimize(
                lambda x: _objective_and_grad(x, n, m, p, r)[0],
                x0,
                jac=lambda x: _objective_and_grad(x, n, m, p, r)[1],
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 200, 'callback': callback}
            )
            
            if result.fun < best_err:
                best_err = result.fun
                best_res = result
        except TimeLimitExceeded:
            continue
        except Exception:
            continue
    
    if best_res is not None and best_res.fun < 1e-8:
        U_opt = best_res.x[:n*m*r].reshape(n*m, r)
        V_opt = best_res.x[n*m*r:2*n*m*r].reshape(m*p, r)
        W_opt = best_res.x[2*n*m*r:].reshape(p*n, r)
        return U_opt, V_opt, W_opt, best_err
    return None


def _try_polish(U, V, W, n, m, p, max_time=5):
    """Try to polish a near-exact solution to exact fractions."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    start_time = time.time()
    
    # Try multiple denominator limits
    for max_den in [10, 50, 100, 500, 1000, 5000]:
        if time.time() - start_time > max_time:
            break
        
        U_frac = [[Fraction(x).limit_denominator(max_den) for x in row] for row in U_arr.tolist()]
        V_frac = [[Fraction(x).limit_denominator(max_den) for x in row] for row in V_arr.tolist()]
        W_frac = [[Fraction(x).limit_denominator(max_den) for x in row] for row in W_arr.tolist()]
        
        ok, _ = _verify_fast(U_frac, V_frac, W_frac, n, m, p, tol=1e-9)
        if ok:
            return U_frac, V_frac, W_frac
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Start with schoolbook
    result = _schoolbook()
    best_rank = n * m * p
    best_result = result
    
    # Theoretical lower bound is max(n*m, m*p, p*n) = 16
    # We search from 16 up to best_rank - 1
    for r in range(16, best_rank):
        if time.time() > deadline - 15:
            break
        
        remaining = deadline - time.time() - 10
        if remaining < 5:
            break
        
        # Allocate time per seed based on remaining time
        num_seeds = min(5, max(2, int(remaining / 10)))
        time_per_seed = max(2, (remaining - 10) / num_seeds)
        
        found = False
        for seed in range(num_seeds):
            if time.time() > deadline - 10:
                break
            
            res = _lbfgs_solve(n, m, p, r, max_time=time_per_seed, seed=seed + 100 + r)
            if res is not None:
                U, V, W, err = res
                if err < 1e-6:
                    # Try to polish
                    polished = _try_polish(U, V, W, n, m, p, max_time=min(3, deadline - time.time() - 10))
                    if polished is not None:
                        U_f, V_f, W_f = polished
                        best_result = {"U": U_f, "V": V_f, "W": W_f}
                        best_rank = r
                    else:
                        # If polishing fails, we might still accept the float solution if the evaluator
                        # is lenient, but the contract says "exactly decompose" and "integer/half-integer".
                        # So we MUST polish. If we can't polish, we discard this rank and try higher.
                        pass
                    found = True
                    break
        
        if found:
            # If we found a good rank, we can stop early if it's very low
            if best_rank <= 18:
                break

    return best_result

# EVOLVE-BLOCK-END
