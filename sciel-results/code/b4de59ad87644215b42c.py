import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 6, 12, 8
    
    # Build the target tensor constraints
    # T[i*m+j, j*p+k, k*n+i] = 1
    constraints = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                constraints.append((i * m + j, j * p + k, k * n + i))
    
    num_constraints = len(constraints)
    
    # Pre-compute index arrays for vectorized operations
    u_idxs = np.array([c[0] for c in constraints])
    v_idxs = np.array([c[1] for c in constraints])
    w_idxs = np.array([c[2] for c in constraints])
    
    def compute_tensor(U, V, W):
        """Compute the full tensor T[a,b,c] = sum_k U[a,k]*V[b,k]*W[c,k]"""
        return np.einsum('ak,bk,ck->abc', U, V, W)
    
    def check_exact(U, V, W, tol=1e-9):
        """Check if decomposition is exact"""
        T = compute_tensor(U, V, W)
        # Check that all constraint positions are 1 and all others are 0
        for u, v, w in constraints:
            if abs(T[u, v, w] - 1.0) > tol:
                return False
        # Check all non-constraint positions are 0
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if (a, b, c) not in constraints:
                        if abs(T[a, b, c]) > tol:
                            return False
        return True
    
    def error_metric(U, V, W):
        """Compute error: sum of squared differences at constraint positions + penalty for non-zero elsewhere"""
        T = compute_tensor(U, V, W)
        err = 0.0
        for u, v, w in constraints:
            err += (T[u, v, w] - 1.0) ** 2
        # Penalty for non-constraint positions
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if (a, b, c) not in constraints:
                        err += T[a, b, c] ** 2
        return err
    
    def try_optimize_rank(r, num_restarts=20, max_iter=300):
        """Try to find rank-r decomposition using gradient descent with momentum"""
        if time.time() > deadline - 10:
            return None
        
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            np.random.seed(42 + restart * 1000)
            U = np.random.randn(nm, r) * 0.3
            V = np.random.randn(mp, r) * 0.3
            W = np.random.randn(pn, r) * 0.3
            
            # Gradient descent with momentum
            lr = 0.05
            momentum = 0.9
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
            
            prev_err = float('inf')
            stall_count = 0
            
            for step in range(max_iter):
                if time.time() > deadline - 10:
                    break
                
                # Compute current tensor
                T = compute_tensor(U, V, W)
                
                # Compute error and gradients
                err = 0.0
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                # Constraint positions: target = 1
                for idx, (u, v, w) in enumerate(constraints):
                    diff = T[u, v, w] - 1.0
                    err += diff * diff
                    # Gradient: dT/du = V[v,:]*W[w,:], etc.
                    dU[u, :] += 2 * diff * V[v, :] * W[w, :]
                    dV[v, :] += 2 * diff * U[u, :] * W[w, :]
                    dW[w, :] += 2 * diff * U[u, :] * V[v, :]
                
                # Non-constraint positions: target = 0
                for a in range(nm):
                    for b in range(mp):
                        for c in range(pn):
                            if (a, b, c) not in constraints:
                                diff = T[a, b, c]
                                err += diff * diff
                                dU[a, :] += 2 * diff * V[b, :] * W[c, :]
                                dV[b, :] += 2 * diff * U[a, :] * W[c, :]
                                dW[c, :] += 2 * diff * U[a, :] * V[b, :]
                
                # Update with momentum
                vU = momentum * vU - lr * dU
                vV = momentum * vV - lr * dV
                vW = momentum * vW - lr * dW
                
                U += vU
                V += vV
                W += vW
                
                # Check for convergence
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                # Detect stall
                if err > prev_err - 1e-10:
                    stall_count += 1
                else:
                    stall_count = 0
                prev_err = err
                
                if stall_count > 30:
                    break
                
                if err < 1e-10:
                    break
            
            # After optimization, try rounding to half-integers
            if best_err < 0.1:
                # Try rounding to nearest 0.5
                U_r = np.round(best_U * 2) / 2
                V_r = np.round(best_V * 2) / 2
                W_r = np.round(best_W * 2) / 2
                err_r = error_metric(U_r, V_r, W_r)
                if err_r < 1e-8:
                    return U_r, V_r, W_r
                
                # Try rounding to nearest integer
                U_i = np.round(best_U)
                V_i = np.round(best_V)
                W_i = np.round(best_W)
                err_i = error_metric(U_i, V_i, W_i)
                if err_i < 1e-8:
                    return U_i, V_i, W_i
        
        return None
    
    # Try ranks from low to high
    # Known lower bounds: for 2x3x4, rank is at least 18 (by various bounds)
    # Let's try ranks 18-23
    for r in [18, 19, 20, 21, 22, 23]:
        if time.time() > deadline - 20:
            break
        
        sol = try_optimize_rank(r, num_restarts=15, max_iter=400)
        if sol is not None:
            U, V, W = sol
            # Final verification
            if check_exact(U, V, W, tol=1e-8):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If no lower rank found, return the schoolbook rank-24 solution
    rank_seed = nm * mp * pn  # 24
    U_seed = np.zeros((nm, rank_seed))
    V_seed = np.zeros((mp, rank_seed))
    W_seed = np.zeros((pn, rank_seed))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_seed[i * m + j, col] = 1
                V_seed[j * p + k, col] = 1
                W_seed[k * n + i, col] = 1
                col += 1
    
    return {"U": U_seed.tolist(), "V": V_seed.tolist(), "W": W_seed.tolist()}
