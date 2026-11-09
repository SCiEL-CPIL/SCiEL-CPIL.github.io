"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s safety margin for return
    
    # The tensor constraint: for all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # n=3, m=3, p=4: U:(9,rank), V:(12,rank), W:(12,rank)
    
    # Schoolbook as fallback
    rank_school = n * m * p  # 36
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_school
    
    # Precompute index pairs for efficiency
    index_pairs = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    idx_U = [i * m + j for i, j, k in index_pairs]
    idx_V = [j * p + k for i, j, k in index_pairs]
    idx_W = [k * n + i for i, j, k in index_pairs]
    num_constraints = len(index_pairs)
    
    def compute_violations(U, V, W, rank):
        """Compute constraint violations using vectorized operations"""
        # For each constraint (i,j,k): sum_r U[idx_U[i*m+j], r] * V[idx_V[j*p+k], r] * W[idx_W[k*n+i], r]
        # Use einsum for efficiency
        # U rows indexed by idx_U, V rows by idx_V, W rows by idx_W
        U_sub = U[idx_U, :]  # (num_constraints, rank)
        V_sub = V[idx_V, :]  # (num_constraints, rank)
        W_sub = W[idx_W, :]  # (num_constraints, rank)
        
        # Compute element-wise products and sum over rank
        products = U_sub * V_sub * W_sub  # (num_constraints, rank)
        sums = np.sum(products, axis=1)  # (num_constraints,)
        return sums - 1.0
    
    def objective_from_x(x, rank):
        U = x[:n*m*rank].reshape(n*m, rank)
        V = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
        v = compute_violations(U, V, W, rank)
        return np.sum(v**2)
    
    def gradient_from_x(x, rank):
        """Compute gradient of sum of squared violations"""
        U = x[:n*m*rank].reshape(n*m, rank)
        V = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        v = compute_violations(U, V, W, rank)  # (num_constraints,)
        
        # Gradient of sum(v^2) = 2 * sum(v * dv/dparam)
        # dv/dU[idx_U[a], r] = V[idx_V[a], r] * W[idx_W[a], r]
        # dv/dV[idx_V[a], r] = U[idx_U[a], r] * W[idx_W[a], r]
        # dv/dW[idx_W[a], r] = U[idx_U[a], r] * V[idx_V[a], r]
        
        U_sub = U[idx_U, :]  # (num_constraints, rank)
        V_sub = V[idx_V, :]  # (num_constraints, rank)
        W_sub = W[idx_W, :]  # (num_constraints, rank)
        
        # For each constraint a, contribution to gradient
        # dU_grad[idx_U[a], r] += 2 * v[a] * V_sub[a, r] * W_sub[a, r]
        # dV_grad[idx_V[a], r] += 2 * v[a] * U_sub[a, r] * W_sub[a, r]
        # dW_grad[idx_W[a], r] += 2 * v[a] * U_sub[a, r] * V_sub[a, r]
        
        dU = np.zeros((n*m, rank))
        dV = np.zeros((m*p, rank))
        dW = np.zeros((p*n, rank))
        
        # Vectorized accumulation
        # For U: for each a, add 2*v[a] * (V_sub[a] * W_sub[a]) to dU[idx_U[a]]
        contrib_U = 2.0 * v[:, np.newaxis] * (V_sub * W_sub)  # (num_constraints, rank)
        np.add.at(dU, idx_U, contrib_U)
        
        contrib_V = 2.0 * v[:, np.newaxis] * (U_sub * W_sub)  # (num_constraints, rank)
        np.add.at(dV, idx_V, contrib_V)
        
        contrib_W = 2.0 * v[:, np.newaxis] * (U_sub * V_sub)  # (num_constraints, rank)
        np.add.at(dW, idx_W, contrib_W)
        
        grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        return grad
    
    def verify_decomposition(U, V, W, rank, tol=1e-8):
        """Verify the decomposition is valid"""
        max_v = 0
        for i, j, k in index_pairs:
            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
            max_v = max(max_v, abs(val - 1.0))
        return max_v < tol
    
    def try_rank(rank, num_restarts=5, time_per_restart=None):
        """Try to find a valid decomposition for given rank"""
        if time_per_restart is None:
            time_per_restart = 15.0
        
        np.random.seed(42 + rank * 100)
        
        for restart in range(num_restarts):
            elapsed = time.time() - start_time
            if elapsed > deadline - 25:
                return None
            
            # Initialize with different strategies
            if restart == 0:
                # Sparse initialization based on schoolbook structure (fixed index)
                x_init = np.zeros(n*m*rank + m*p*rank + p*n*rank)
                # Distribute the 36 constraints across rank components
                for idx, (i, j, k) in enumerate(index_pairs):
                    r = idx % rank
                    x_init[i*m + j + r] = 1.0
                    x_init[n*m*rank + j*p + k + r] = 1.0
                    x_init[n*m*rank + m*p*rank + k*n + i + r] = 1.0
            elif restart == 1:
                # Random initialization
                x_init = np.random.randn(n*m*rank + m*p*rank + p*n*rank) * 0.5
            elif restart == 2:
                # Larger random initialization
                x_init = np.random.randn(n*m*rank + m*p*rank + p*n*rank) * 1.0
            else:
                # Scaled random
                scale = 0.3 + 0.2 * (restart - 3)
                x_init = np.random.randn(n*m*rank + m*p*rank + p*n*rank) * scale
            
            try:
                # Use L-BFGS-B with gradient and time-limited optimization
                start_opt = time.time()
                
                def callback(xk):
                    if time.time() - start_opt > time_per_restart:
                        raise TimeoutError("Optimization time limit reached")
                    return False
                
                res = minimize(objective_from_x, x_init, args=(rank,),
                              jac=gradient_from_x,
                              method='L-BFGS-B', 
                              callback=callback,
                              options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 3000})
                
                # If we got a good solution, do a final polish pass
                if res.fun < 1e-10:
                    U_opt = res.x[:n*m*rank].reshape(n*m, rank)
                    V_opt = res.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                    W_opt = res.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                    
                    # Final high-precision polish
                    if time.time() - start_time < deadline - 15:
                        try:
                            res2 = minimize(objective_from_x, res.x, args=(rank,),
                                           jac=gradient_from_x,
                                           method='L-BFGS-B', 
                                           options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 1000})
                            if res2.fun < res.fun:
                                res = res2
                        except Exception:
                            pass
                    
                    U_opt = res.x[:n*m*rank].reshape(n*m, rank)
                    V_opt = res.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                    W_opt = res.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                    
                    if verify_decomposition(U_opt, V_opt, rank, tol=1e-10):
                        # Try to round to half-integers
                        U_rounded = np.round(U_opt * 2) / 2
                        V_rounded = np.round(V_opt * 2) / 2
                        W_rounded = np.round(W_opt * 2) / 2
                        
                        if verify_decomposition(U_rounded, V_rounded, rank, tol=1e-8):
                            return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                        else:
                            # Use cleaned floating point values
                            U_final = np.round(U_opt, 10)
                            V_final = np.round(V_opt, 10)
                            W_final = np.round(W_opt, 10)
                            return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
                        
            except TimeoutError:
                continue
            except Exception:
                continue
        return None
    
    # Try ranks from low to high with adaptive restarts
    # Based on known results, minimum rank for 3x3 * 3x4 is likely around 19-25
    for rank in range(12, best_rank):
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            break
        
        # Adjust number of restarts based on remaining time
        remaining = deadline - 30 - elapsed
        if remaining < 20:
            break
        
        # Use more restarts for promising ranks
        if rank <= 20:
            num_restarts = 4
        else:
            num_restarts = 3
        
        # Scale time per restart based on remaining time
        time_per_restart = min(15.0, max(5.0, (remaining - 10) / num_restarts))
        
        result = try_rank(rank, num_restarts=num_restarts, time_per_restart=time_per_restart)
        if result is not None:
            best_result = result
            best_rank = rank
    
    return best_result
# EVOLVE-BLOCK-END
