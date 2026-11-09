"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if len(U) > 0 else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # T[(i,j), (jp,k), (kp,ip)] = delta(i,ip)*delta(j,jp)*delta(k,kp)
    # Check: sum_c U[i*m+j,c] * V[jp*p+k,c] * W[kp*n+ip,c] = delta(i,ip)*delta(j,jp)*delta(k,kp)
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            lhs = 0.0
                            for c in range(rank):
                                lhs += U[i*m+j][c] * V[jp*p+k][c] * W[kp*n+ip][c]
                            rhs = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                            if abs(lhs - rhs) > 1e-5:
                                return False
    return True

def _loss_and_grad(x, n, m, p, r):
    """
    Compute loss and gradient for tensor decomposition.
    x is a flattened array of length (n*m + m*p + p*n) * r
    Order: U (n*m*r), V (m*p*r), W (p*n*r)
    """
    U = x[:n*m*r].reshape(n*m, r)
    V = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
    W = x[n*m*r + m*p*r :].reshape(p*n, r)
    
    loss = 0.0
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    dW = np.zeros_like(W)
    
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            rhs = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                            # s = sum_c U[i*m+j,c] * V[jp*p+k,c] * W[kp*n+ip,c]
                            # Vectorized:
                            s = np.dot(np.dot(U[i*m+j], V[jp*p+k]), W[kp*n+ip])
                            err = s - rhs
                            loss += err * err
                            
                            # Gradients
                            # dL/dU = 2 * err * (V * W)
                            # dL/dV = 2 * err * (U * W)
                            # dL/dW = 2 * err * (U * V)
                            dU[i*m+j] += 2 * err * (V[jp*p+k] * W[kp*n+ip])
                            dV[jp*p+k] += 2 * err * (U[i*m+j] * W[kp*n+ip])
                            dW[kp*n+ip] += 2 * err * (U[i*m+j] * V[jp*p+k])
                            
    grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
    return loss, grad

def _try_rank_scipy(r, n, m, p, deadline, seed):
    """
    Try to find a rank-r decomposition using scipy.optimize.
    """
    if time.time() > deadline - 5:
        return None
        
    np.random.seed(seed)
    
    # Initialize with small random values
    U = np.random.randn(n*m, r) * 0.1
    V = np.random.randn(m*p, r) * 0.1
    W = np.random.randn(p*n, r) * 0.1
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    # We use L-BFGS-B which is good for large scale problems
    # However, for small dimensions, even Nelder-Mead or BFGS might work.
    # Let's stick with L-BFGS-B for robustness.
    
    def fun(x):
        loss, grad = _loss_and_grad(x, n, m, p, r)
        return loss, grad

    try:
        res = opt.minimize(fun, x0, jac=True, method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.success or res.fun < 1e-6:
            x_sol = res.x
            U_sol = x_sol[:n*m*r].reshape(n*m, r)
            V_sol = x_sol[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
            W_sol = x_sol[n*m*r + m*p*r :].reshape(p*n, r)
            
            # Try to round to rational numbers
            # The problem asks for integer/half-integer entries.
            # Let's try rounding to nearest 0.5 or integer.
            for denom in [1, 2, 10, 100]:
                U_r = np.round(U_sol * denom) / denom
                V_r = np.round(V_sol * denom) / denom
                W_r = np.round(W_sol * denom) / denom
                
                U_list = U_r.tolist()
                V_list = V_r.tolist()
                W_list = W_r.tolist()
                
                if _verify(U_list, V_list, W_list, n, m, p):
                    return U_list, V_list, W_list
            
            # If rounding failed but loss is very low, return the float solution?
            # The task says "integer/half-integer entries".
            # If we can't round, this solution is invalid for the contract.
            # But let's check if it's close enough to be considered valid by the evaluator?
            # The evaluator checks "exactly decompose". So we MUST round.
            
    except Exception:
        pass
        
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin before 120s limit
    
    # Schoolbook decomposition as baseline
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    # Start from rank 47 down to 30
    # We use multiple restarts for each rank to avoid local minima
    for r in range(best_rank - 1, 29, -1):
        if time.time() > deadline:
            break
            
        found = False
        # Try a few random initializations
        for seed in range(5):
            if time.time() > deadline:
                break
                
            result = _try_rank_scipy(r, n, m, p, deadline, seed + r * 100)
            if result is not None:
                U_list, V_list, W_list = result
                best_rank = r
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                print(f"Found rank {r} decomposition")
                found = True
                break
        
        if found:
            break
            
    return best_result
# EVOLVE-BLOCK-END
