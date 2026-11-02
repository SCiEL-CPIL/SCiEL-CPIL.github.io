import numpy as np
import time
import random

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    def get_indices():
        """Precompute indices for the tensor identity check."""
        # We need to check: sum_r U[i, j, r] V[j, k, r] W[k, i, r] == delta_i_i' delta_j_j' delta_k_k'
        # Wait, the indices in the prompt are:
        # U shape (n*m, rank). Index: i*m + j.
        # V shape (m*p, rank). Index: j*p + k.
        # W shape (p*n, rank). Index: k*n + i. (Trace convention W[k*n+i, r])
        
        # Identity:
        # C[i, k] = sum_j A[i, j] B[j, k]
        # C_recon[i, k] = sum_r (sum_{i', j'} U[i'*m+j', r] A[i', j']) * (sum_{j'', k'} V[j''*p+k', r] B[j'', k']) * W[k*n+i, r]
        
        # Let's expand C_recon[i, k]:
        # sum_{i', j', j'', k'} A[i', j'] B[j'', k'] W[k*n+i, r] U[i'*m+j', r] V[j''*p+k', r]
        
        # We want C_recon[i, k] = sum_j A[i, j] B[j, k]
        # Coeff of A[i', j'] B[j'', k'] in C_recon[i, k] must be:
        # 1 if i'=i, k'=k, j'=j''
        # 0 otherwise
        
        # So we need:
        # sum_r U[i'*m+j', r] V[j''*p+k', r] W[k*n+i, r] = delta_{i', i} delta_{k', k} delta_{j', j''}
        
        # We can check this identity by evaluating it for specific values.
        # Since it's a polynomial in U, V, W, if it holds for enough random A, B, it likely holds exactly.
        # Or we can directly check the tensor contraction.
        
        pass

    def check_identity(U, V, W, R):
        """
        Check if the factors satisfy the tensor identity.
        U: (R, n, m)
        V: (R, m, p)
        W: (R, p, n)
        """
        # U_flat: (R, n*m) -> reshape to (R, n, m)
        # V_flat: (R, m*p) -> reshape to (R, m, p)
        # W_flat: (R, p*n) -> reshape to (R, p, n)
        
        U_t = U.reshape(R, n, m)
        V_t = V.reshape(R, m, p)
        W_t = W.reshape(R, p, n)
        
        # Compute LHS tensor T[i, j, k] = sum_r U[r, i, j] V[r, j, k] ... wait.
        # The identity is:
        # sum_r U[r, i, j] V[r, j, k] W[r, k, i] ? No.
        
        # Let's re-derive carefully.
        # S_A[r] = sum_{i,j} U[r, i, j] A[i, j]
        # S_B[r] = sum_{j,k} V[r, j, k] B[j, k]
        # C[i, k] = sum_r S_A[r] S_B[r] W[r, k, i]
        
        # Expand C[i, k]:
        # sum_r (sum_{i',j'} U[r,i',j'] A[i',j']) (sum_{j'',k'} V[r,j'',k'] B[j'',k']) W[r,k,i]
        # = sum_{i',j',j'',k'} A[i',j'] B[j'',k'] W[r,k,i] sum_r U[r,i',j'] V[r,j'',k']
        
        # We want C[i,k] = sum_j A[i,j] B[j,k].
        # So coeff of A[i',j'] B[j'',k'] in C[i,k] must be delta_{i',i} delta_{j',j} delta_{j'',j} delta_{k',k}
        # = delta_{i',i} delta_{k',k} delta_{j',j''}
        
        # So we need:
        # sum_r U[r,i',j'] V[r,j'',k'] W[r,k,i] = delta_{i',i} delta_{k',k} delta_{j',j''}
        
        # Let's check this for all i, i', j, j', k, k'.
        # Total checks: n*n*m*m*p*p = 2*2*4*4*4*4 = 1024 equations.
        
        max_err = 0.0
        for i in range(n):
            for k in range(p):
                for i_prime in range(n):
                    for k_prime in range(p):
                        for j_prime in range(m):
                            for j_double_prime in range(m):
                                target = 1.0 if (i == i_prime and k == k_prime and j_prime == j_double_prime) else 0.0
                                val = 0.0
                                for r in range(R):
                                    val += U_t[r, i_prime, j_prime] * V_t[r, j_double_prime, k_prime] * W_t[r, k, i]
                                err = abs(val - target)
                                if err > max_err:
                                    max_err = err
        return max_err

    def round_to_half_int(arr):
        return np.round(arr * 2) / 2.0

    def try_rank(R):
        # Initialize
        U = np.random.randn(R, nm)
        V = np.random.randn(R, mp)
        W = np.random.randn(R, pn)
        
        # Try a few random restarts
        best_err = float('inf')
        best_params = None
        
        num_restarts = 20
        for _ in range(num_restarts):
            if time.time() > deadline:
                return None
            
            U = np.random.randn(R, nm) * 0.5
            V = np.random.randn(R, mp) * 0.5
            W = np.random.randn(R, pn) * 0.5
            
            # Simple gradient descent step?
            # Or just check if random init works (unlikely).
            # Let's use a few steps of a simple optimizer.
            
            # For speed, we will just try random inits and rounding.
            # If R is small, random inits are unlikely to work.
            # We need a real optimizer.
            
            # Let's use scipy.optimize.minimize with L-BFGS-B
            from scipy.optimize import minimize
            
            def objective(params):
                U_flat = params[:R*nm].reshape(R, nm)
                V_flat = params[R*nm:R*nm+R*mp].reshape(R, mp)
                W_flat = params[R*nm+R*mp:R*nm+R*mp+R*pn].reshape(R, pn)
                return check_identity(U_flat, V_flat, W_flat, R)
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50, 'maxfun': 100})
                if res.fun < best_err:
                    best_err = res.fun
                    best_params = res.x
            except Exception:
                pass
            
            if best_err < 1e-8:
                break
        
        if best_params is not None and best_err < 1e-6:
            U_opt = best_params[:R*nm].reshape(R, nm)
            V_opt = best_params[R*nm:R*nm+R*mp].reshape(R, mp)
            W_opt = best_params[R*nm+R*mp:R*nm+R*mp+R*pn].reshape(R, pn)
            
            # Round to half-integers
            U_r = round_to_half_int(U_opt)
            V_r = round_to_half_int(V_opt)
            W_r = round_to_half_int(W_opt)
            
            # Verify rounded solution
            err = check_identity(U_r, V_r, W_r, R)
            if err < 1e-9:
                return {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
        
        return None

    # Try ranks from 28 down to 24
    # If none found, return schoolbook 32
    for R in [28, 27, 26, 25, 24]:
        if time.time() > deadline:
            break
        result = try_rank(R)
        if result is not None:
            return result

    # Fallback: Schoolbook rank 32
    R = n * m * p
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
