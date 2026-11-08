import numpy as np
import time
import itertools
from scipy.optimize import minimize

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # A: n×m, B: m×p, C: n×p
    # U: (n*m) × rank, V: (m*p) × rank, W: (p*n) × rank
    # W index: k*n + i
    
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin for 150s limit
    
    # 1. Verification function
    def verify(U, V, W):
        U3 = U.reshape(n, m, -1)
        V3 = V.reshape(m, p, -1)
        W3 = W.reshape(p, n, -1)
        # Target tensor is all ones for valid (i,j,k)
        target = np.ones((n, m, p))
        computed = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        # Check for exact match with tolerance for floating point errors
        return np.allclose(computed, target, atol=1e-6)

    # 2. Schoolbook construction (Rank 96)
    def get_schoolbook():
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

    # 3. Optimization Strategy
    # We attempt to find a decomposition of rank R < 96.
    # We start with R=24 (theoretical lower bound max(nm, mp, pn) = 24).
    # If that fails, we try higher ranks up to a limit where it becomes computationally expensive or time runs out.
    
    # The objective function is the squared Frobenius norm of the error:
    # E = || T - sum_r u_r \otimes v_r \otimes w_r ||_F^2
    
    # We flatten U, V, W into a single vector x.
    # x[0:R*n*m] -> U
    # x[R*n*m:R*n*m + R*m*p] -> V
    # x[R*n*m + R*m*p : R*n*m + R*m*p + R*p*n] -> W
    
    def objective(x, R):
        U = x[:R*n*m].reshape(n, m, R)
        V = x[R*n*m:R*n*m + R*m*p].reshape(m, p, R)
        W = x[R*n*m + R*m*p:].reshape(p, n, R)
        
        # Compute tensor contraction
        # T_computed[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Efficiently:
        # Temp[j,k,r] = V[j,k,r] * W[k,i,r] ? No, W depends on i.
        # Let's use einsum for speed, though it might be slow for large R.
        # For small R and dimensions, it's fine.
        
        # W is (p, n, R). We need W[k, i, r].
        # V is (m, p, R). We need V[j, k, r].
        # U is (n, m, R). We need U[i, j, r].
        
        # Sum over r: U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Note: W in the matrix is indexed [k*n+i, r]. In 3D it is W3[k, i, r].
        
        computed = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        target = np.ones((n, m, p))
        
        diff = computed - target
        return np.sum(diff**2)

    def gradient(x, R):
        # Numerical gradient is too slow. We need analytical or rely on BFGS with finite diff.
        # Given the small dimensions, BFGS with analytical gradient is preferred if possible,
        # but deriving the gradient of a tensor network is complex.
        # Let's use BFGS with finite difference approximation initially, or just 'L-BFGS-B' with numerical grad.
        # Actually, for this specific tensor, we can derive the gradient.
        # L = sum_{i,j,k} ( sum_r U_{ijr} V_{jkr} W_{kir} - 1 )^2
        # dL/dU_{ijr} = 2 * sum_{i,j,k} ( ... ) * V_{jkr} * W_{kir}
        # This is doable.
        
        U = x[:R*n*m].reshape(n, m, R)
        V = x[R*n*m:R*n*m + R*m*p].reshape(m, p, R)
        W = x[R*n*m + R*m*p:].reshape(p, n, R)
        
        computed = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        target = np.ones((n, m, p))
        err = computed - target
        
        # Gradient for U
        # dL/dU_{ijr} = 2 * sum_k err_{ijk} * V_{jkr} * W_{kir}
        # V_{jkr} is V[j, k, r]. W_{kir} is W[k, i, r].
        # Note: W is indexed [k, i, r] in our 3D reshape.
        
        # We need to sum over k.
        # grad_U[i, j, r] = 2 * sum_k err[i, j, k] * V[j, k, r] * W[k, i, r]
        
        # Let's compute this efficiently.
        # err is (n, m, p). V is (m, p, R). W is (p, n, R).
        
        # For a fixed r:
        # grad_U_r[i, j] = 2 * sum_k err[i, j, k] * V[j, k, r] * W[k, i, r]
        
        # This is a bit tricky with einsum directly on the full arrays due to shared indices.
        # Let's iterate over r for clarity, as R is expected to be small (< 30).
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for r in range(R):
            # U term:
            # sum_k err[i,j,k] * V[j,k,r] * W[k,i,r]
            # V_slice = V[:, :, r] (m, p)
            # W_slice = W[:, :, r] (p, n) -> Note: W[k, i, r] means row k, col i of W_slice.
            
            V_r = V[:, :, r] # (m, p)
            W_r = W[:, :, r] # (p, n)
            
            # We want G_U[i, j] = sum_k err[i, j, k] * V_r[j, k] * W_r[k, i]
            # This is like:
            # Temp1[i, j, k] = err[i, j, k] * V_r[j, k]
            # Then sum over k with W_r[k, i].
            
            # Let's use einsum:
            # err: ijk
            # V_r: jk
            # W_r: ki (since W[k, i])
            
            # grad_U_r[i, j] = einsum('ijk,jk,ki->ij', err, V_r, W_r)
            grad_U_r = np.einsum('ijk,jk,ki->ij', err, V_r, W_r)
            grad_U[:, :, r] = 2 * grad_U_r
            
            # V term:
            # dL/dV_{jkr} = 2 * sum_i err[i,j,k] * U[i,j,r] * W[k,i,r]
            # U_r: (n, m) -> U[i, j, r]
            # W_r: (p, n) -> W[k, i, r]
            # err: (n, m, p)
            
            U_r = U[:, :, r] # (n, m)
            
            # grad_V_r[j, k] = einsum('ijk,ij,ki->jk', err, U_r, W_r)
            grad_V_r = np.einsum('ijk,ij,ki->jk', err, U_r, W_r)
            grad_V[:, :, r] = 2 * grad_V_r
            
            # W term:
            # dL/dW_{kir} = 2 * sum_j err[i,j,k] * U[i,j,r] * V[j,k,r]
            # Note: W index is [k, i, r].
            
            # grad_W_r[k, i] = einsum('ijk,ij,jk->ki', err, U_r, V_r)
            grad_W_r = np.einsum('ijk,ij,jk->ki', err, U_r, V_r)
            grad_W[:, :, r] = 2 * grad_W_r
        
        # Flatten gradients
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    best_rank = None
    best_U, best_V, best_W = None, None, None
    
    # Try ranks from 24 upwards
    # We limit the number of attempts based on time.
    ranks_to_try = [24, 25, 26, 28, 30, 32, 35, 40]
    
    for R in ranks_to_try:
        if time.time() > deadline - 20.0:
            break
            
        # Initialize with random values or schoolbook truncated?
        # Random initialization is better for escaping local minima.
        # Scale: values around 1.
        np.random.seed(42) # Fixed seed for reproducibility in this attempt
        
        # Try multiple restarts for each rank
        max_restarts = 5
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10.0:
                break
                
            x0 = np.random.randn(R * (n*m + m*p + p*n)) * 0.5
            
            # Use BFGS
            try:
                res = minimize(objective, x0, jac=gradient, method='BFGS', 
                               options={'maxiter': 1000, 'gtol': 1e-8, 'disp': False})
                
                if res.success or res.fun < 1e-6:
                    # Check if the solution is close to integer/half-integer?
                    # The task requires integer/half-integer entries.
                    # BFGS finds real numbers. We must check if we can round them.
                    # Or, more robustly, check if the exact decomposition holds.
                    # If res.fun is very small, the real decomposition is exact.
                    # Can we find an integer decomposition?
                    # This is a hard problem. 
                    # However, if the rank is minimal, the solution space might be discrete.
                    # Let's just check if the real solution is valid (error ~ 0).
                    # The evaluator checks for "integer/half-integer entries".
                    # If BFGS gives 0.9999, we can round to 1.0.
                    # If it gives 0.5, we keep 0.5.
                    
                    U_sol = res.x[:R*n*m].reshape(n, m, R)
                    V_sol = res.x[R*n*m:R*n*m + R*m*p].reshape(m, p, R)
                    W_sol = res.x[R*n*m + R*m*p:].reshape(p, n, R)
                    
                    # Round to nearest half-integer
                    U_rounded = np.round(U_sol * 2) / 2
                    V_rounded = np.round(V_sol * 2) / 2
                    W_rounded = np.round(W_sol * 2) / 2
                    
                    # Verify the rounded solution
                    if verify(U_rounded.reshape(n*m, R), V_rounded.reshape(m*p, R), W_rounded.reshape(p*n, R)):
                        best_rank = R
                        best_U = U_rounded.reshape(n*m, R)
                        best_V = V_rounded.reshape(m*p, R)
                        best_W = W_rounded.reshape(p*n, R)
                        found = True
                        break
                        
            except Exception as e:
                pass
        
        if found:
            break

    if best_rank is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback to Schoolbook
    U_s, V_s, W_s = get_schoolbook()
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
