# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n, m, p = 3, 3, 3
    deadline = time.time() + 115.0  # Safety margin

    def try_find_decomposition(rank, max_restarts=5, seed_base=42):
        """
        Attempts to find a rank-r decomposition with integer/half-integer entries.
        Returns (U, V, W) if found, else None.
        """
        from scipy.optimize import minimize
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
                
            np.random.seed(seed_base + restart)
            
            # Initialize U, V, W randomly
            # Scaling helps with optimization
            scale = 1.0
            U = np.random.randn(n*m, rank) * scale
            V = np.random.randn(m*p, rank) * scale
            W = np.random.randn(p*n, rank) * scale
            
            def objective(params):
                U_flat, V_flat, W_flat = params[:n*m*rank], params[n*m*rank:n*m*rank+m*p*rank], params[n*m*rank+m*p*rank:]
                U = U_flat.reshape(n*m, rank)
                V = V_flat.reshape(m*p, rank)
                W = W_flat.reshape(p*n, rank)
                
                # Efficient computation of the tensor product sum
                # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # We want T[i,j,k] == 1
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :]) # This is wrong, W is also involved
                            # Actually: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            # This is element-wise product of three vectors, then sum
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            error += (val - 1.0)**2
                return error

            def gradient(params):
                U_flat, V_flat, W_flat = params[:n*m*rank], params[n*m*rank:n*m*rank+m*p*rank], params[n*m*rank+m*p*rank:]
                U = U_flat.reshape(n*m, rank)
                V = V_flat.reshape(m*p, rank)
                W = W_flat.reshape(p*n, rank)
                
                gU = np.zeros_like(U)
                gV = np.zeros_like(V)
                gW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # val = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            # d(val)/d(U[i*m+j, r]) = V[j*p+k, r] * W[k*n+i, r]
                            # d(error)/d(U[i*m+j, r]) = 2 * (val - 1) * V[j*p+k, r] * W[k*n+i, r]
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            diff = val - 1.0
                            gU[i*m+j, :] += 2.0 * diff * (V[j*p+k, :] * W[k*n+i, :])
                            gV[j*p+k, :] += 2.0 * diff * (U[i*m+j, :] * W[k*n+i, :])
                            gW[k*n+i, :] += 2.0 * diff * (U[i*m+j, :] * V[j*p+k, :])
                            
                return np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])

            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = minimize(objective, params, jac=gradient, method='L-BFGS-B', 
                               options={'maxiter': 500, 'gtol': 1e-8})
            except Exception:
                continue
                
            if res.fun < 1e-6:
                # Round to nearest half-integer
                U_opt = res.x[:n*m*rank].reshape(n*m, rank)
                V_opt = res.x[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
                W_opt = res.x[n*m*rank+m*p*rank:].reshape(p*n, rank)
                
                U_round = np.round(U_opt * 2) / 2
                V_round = np.round(V_opt * 2) / 2
                W_round = np.round(W_opt * 2) / 2
                
                # Verify exactness
                verified = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_round[i*m+j, :] * V_round[j*p+k, :] * W_round[k*n+i, :])
                            if abs(val - 1.0) > 1e-9:
                                verified = False
                                break
                        if not verified:
                            break
                    if not verified:
                        break
                
                if verified:
                    return U_round, V_round, W_round
                    
        return None

    # Try rank 20 first (known minimal or near-minimal)
    U, V, W = try_find_decomposition(20, max_restarts=10)
    if U is not None:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    if time.time() < deadline - 20:
        # Try rank 23
        U, V, W = try_find_decomposition(23, max_restarts=10)
        if U is not None:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to rank 27
    rank = 27
    U = np.zeros((9, rank), dtype=int)
    V = np.zeros((9, rank), dtype=int)
    W = np.zeros((9, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
