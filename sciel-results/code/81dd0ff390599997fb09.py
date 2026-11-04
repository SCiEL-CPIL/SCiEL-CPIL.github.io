import time
import numpy as np
from fractions import Fraction
import sys

DIMENSIONS = [2, 5, 5]
_DEADLINE = 135.0

def _remaining():
    return _DEADLINE - time.time()

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify(n, m, p, U, V, W):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _try_rank_49(n, m, p):
    """
    Attempt to find rank-49 decomposition for 2x5x5 matrix multiplication.
    Uses numerical optimization initialized near the schoolbook solution.
    """
    if time.time() > _DEADLINE - 20:
        return None
    
    target_rank = 49
    np.random.seed(42)
    
    # Initialize with schoolbook terms, but drop one and perturb
    # We start with 50 terms, drop the last one, and add noise to the rest
    U0 = np.zeros((n * m, target_rank))
    V0 = np.zeros((m * p, target_rank))
    W0 = np.zeros((p * n, target_rank))
    
    # Fill first 49 terms with schoolbook pattern
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if c < target_rank:
                    U0[i * m + j, c] = 1.0
                    V0[j * p + k, c] = 1.0
                    W0[k * n + i, c] = 1.0
                    c += 1
    
    # The 50th term (i=1, j=4, k=4) is missing.
    # We need to adjust the first 49 terms to compensate.
    # Add small random perturbation to help escape local minima
    noise_scale = 0.01
    U0 += np.random.normal(0, noise_scale, U0.shape)
    V0 += np.random.normal(0, noise_scale, V0.shape)
    W0 += np.random.normal(0, noise_scale, W0.shape)
    
    # Flatten parameters
    params = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
    
    def objective(params):
        U_flat, V_flat, W_flat = np.split(params, [n*m*target_rank, 2*n*m*target_rank])
        U = U_flat.reshape(n*m, target_rank)
        V = V_flat.reshape(m*p, target_rank)
        W = W_flat.reshape(p*n, target_rank)
        
        # Compute tensor entries: T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Vectorized computation
        # U shape: (n*m, R) -> (n, m, R)
        U_3d = U.reshape(n, m, target_rank)
        # V shape: (m*p, R) -> (m, p, R)
        V_3d = V.reshape(m, p, target_rank)
        # W shape: (p*n, R) -> (p, n, R)
        W_3d = W.reshape(p, n, target_rank)
        
        # We need sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # This is a contraction over c, j, k? No, j and k are free indices in the output?
        # Output is C[i, k] = sum_j A[i,j] B[j,k].
        # The tensor entry is for fixed i, j, k.
        # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # Let's compute this efficiently.
        # For each c, we have outer products?
        # T = sum_c U_col_c (reshaped to n,m) * V_col_c (reshaped to m,p) * W_col_c (reshaped to p,n) ?
        # No, the indices are linked: U[i,j], V[j,k], W[k,i].
        # This is a cyclic contraction.
        
        # We can do:
        # Temp[j, k, c] = V[j, k, c] * W[k, i, c] ... wait i is free.
        # Let's just loop over i and k, and sum over j and c.
        # Or loop over c, and add to the tensor.
        
        T = np.zeros((n, m, p))
        for c in range(target_rank):
            u_c = U[:, c].reshape(n, m)
            v_c = V[:, c].reshape(m, p)
            w_c = W[:, c].reshape(p, n)
            
            # T[i, j, k] += u_c[i, j] * v_c[j, k] * w_c[k, i]
            # This is not a standard matmul.
            # For fixed i, k:
            # T[i, :, k] += u_c[i, :] * (v_c * w_c[k, :].T ?)
            # v_c[j, k] * w_c[k, i] is an outer product of v_c[:, k] and w_c[k, :]?
            # Let's compute M_c = u_c (n x m) * diag(v_c[:, k]?) ...
            
            # Efficient way:
            # For each i, k:
            # term = u_c[i, :] * (v_c * w_c[k, :][None, :].T) ???
            # v_c[j, k] * w_c[k, i] for all j.
            # This is u_c[i, j] * v_c[j, k] * w_c[k, i].
            # Sum over j.
            # For fixed i, k:
            # sum_j u_c[i, j] * v_c[j, k] * w_c[k, i]
            # = w_c[k, i] * sum_j u_c[i, j] * v_c[j, k]
            # = w_c[k, i] * (u_c[i, :] @ v_c[:, k])
            
            # So for fixed c:
            # M_c = u_c @ v_c  (shape n x p)
            # Then T[i, j, k] += u_c[i, j] * v_c[j, k] * w_c[k, i]
            # Wait, the formula above sums over j. But T[i, j, k] has j as an index.
            # The tensor entry T[i, j, k] is NOT the sum over j.
            # T[i, j, k] is the coefficient for A[i, j] B[j, k] in the expansion?
            # No, the decomposition is:
            # C[i, k] = sum_j A[i, j] B[j, k] = sum_c (sum_j U[i, j, c] A[i, j]) (sum_k V[j, k, c] B[j, k]) W[k, i, c]
            # This is incorrect. The standard decomposition is:
            # C[i, k] = sum_c (sum_j U[i, j, c] A[i, j]) (sum_k V[j, k, c] B[j, k]) ... no.
            
            # The correct tensor decomposition form is:
            # T_{i, j, k} = sum_c U_{i, j}^{(c)} V_{j, k}^{(c)} W_{k, i}^{(c)}
            # Where T_{i, j, k} = 1 for all i, j, k.
            # So we need to compute T[i, j, k] for all i, j, k and check if it equals 1.
            
            # My loop:
            # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            
            # Let's compute this T efficiently.
            # T is n x m x p.
            # For fixed c:
            # T_c[i, j, k] = U_c[i, j] * V_c[j, k] * W_c[k, i]
            # This is a rank-1 tensor in (i, j, k) space? No, indices are linked.
            # It's a "cyclic" product.
            
            # We can compute T_c by:
            # 1. Create matrix M_c of size n x p: M_c[i, k] = sum_j U_c[i, j] V_c[j, k] ? No, that sums over j.
            # We need to keep j.
            
            # Let's just use numpy broadcasting if memory allows.
            # U_c shape (n, m)
            # V_c shape (m, p)
            # W_c shape (p, n)
            
            # T_c[i, j, k] = U_c[i, j] * V_c[j, k] * W_c[k, i]
            # We can do:
            # Temp1[i, j, k] = U_c[i, j, None] * V_c[None, j, k]
            # Temp2[i, j, k] = Temp1 * W_c[None, None, k, i] ... W_c is (p, n), so W_c[k, i].
            
            # This creates an n x m x p array for each c.
            # n=2, m=5, p=5. Size 50. Very small.
            # R=49. Total operations 49 * 50 * 3 mults. Trivial.
            
            U_c_3d = u_c[:, :, None] # (n, m, 1)
            V_c_3d = v_c[None, :, :] # (1, m, p)
            # Product U_c[i,j] * V_c[j,k]
            UV = U_c_3d * V_c_3d # (n, m, p)
            
            # Now multiply by W_c[k, i]
            # W_c is (p, n). We need W_c[k, i] for each (i, j, k).
            # W_c[k, i] can be broadcast as (n, 1, p) if we transpose?
            # W_c.T is (n, p). W_c.T[i, k] = W_c[k, i].
            W_c_T = w_c.T # (n, p)
            W_c_3d = W_c_T[:, None, :] # (n, 1, p)
            
            T_c = UV * W_c_3d
            T += T_c
        
        # Compute error
        error = np.sum((T - 1.0)**2)
        return error

    # Use L-BFGS-B for optimization
    try:
        from scipy.optimize import minimize
        
        # Initial check
        init_err = objective(params)
        if init_err < 1e-6:
            pass # Already good? Unlikely with noise.
        
        res = minimize(objective, params, method='L-BFGS-B', 
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.fun < 1e-8:
            U_flat, V_flat, W_flat = np.split(res.x, [n*m*target_rank, 2*n*m*target_rank])
            U = U_flat.reshape(n*m, target_rank)
            V = V_flat.reshape(m*p, target_rank)
            W = W_flat.reshape(p*n, target_rank)
            
            # Try rounding to rationals
            # The entries should be integers or half-integers.
            for denom in [1, 2, 4, 8]:
                U_r = np.round(U * denom) / denom
                V_r = np.round(V * denom) / denom
                W_r = np.round(W * denom) / denom
                
                # Verify with exact arithmetic
                if _verify(n, m, p, U_r.tolist(), V_r.tolist(), W_r.tolist()):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            
            # If rounding didn't work, try higher precision or different denom
            # Sometimes the solution is integer.
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            if _verify(n, m, p, U_int.tolist(), V_int.tolist(), W_int.tolist()):
                return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                
    except Exception as e:
        pass
    
    return None

def _construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    
    # Try to find rank 49
    result = _try_rank_49(n, m, p)
    if result is not None:
        if _verify(n, m, p, result['U'], result['V'], result['W']):
            return result
    
    return best

def construct():
    try:
        return _construct()
    except Exception:
        n, m, p = DIMENSIONS
        return _schoolbook(n, m, p)
