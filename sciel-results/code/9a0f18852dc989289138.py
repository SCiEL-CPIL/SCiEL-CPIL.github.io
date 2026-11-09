import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    # Start with the trivial decomposition (rank = n*m*p = 90)
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
    
    # Try to find a lower rank decomposition using random initialization and optimization
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # The matrix multiplication tensor has entries T[i,j,k] = 1 for all (i,j,k)
    # We need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all (i,j,k)
    
    # Try progressively lower ranks
    for target_rank in range(min(n*m, m*p, p*n), 0, -1):
        if time.time() > deadline - 10:
            break
            
        # Try multiple random initializations
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
                
            # Initialize with random values
            np.random.seed(attempt + target_rank)
            U_t = np.random.randn(n * m, target_rank) * 0.1
            V_t = np.random.randn(m * p, target_rank) * 0.1
            W_t = np.random.randn(p * n, target_rank) * 0.1
            
            # Use gradient-free optimization (Nelder-Mead style) for small problems
            # Since the problem is small, we can try a simple iterative approach
            
            # For each (i,j,k), we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is a system of nonlinear equations
            
            # Simple approach: use least squares to solve for one set of variables at a time
            # Given U and V, solve for W using linear least squares
            # Given U and W, solve for V using linear least squares
            # Given V and W, solve for U using linear least squares
            
            converged = False
            for iteration in range(20):
                if time.time() > deadline - 2:
                    break
                    
                # Update W: for each (k,i), we have sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # This is linear in W[k*n+i, :]
                # For fixed (k,i), the equation is: sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # Let A[k,i,j,r] = U[i*m+j, r] * V[j*p+k, r]
                # Then sum_j sum_r A[k,i,j,r] * W[k*n+i, r] = 1
                # This is: sum_r (sum_j A[k,i,j,r]) * W[k*n+i, r] = 1
                
                W_new = np.zeros_like(W_t)
                for k in range(p):
                    for i in range(n):
                        # Build the linear system for W[k*n+i, :]
                        # sum_r (sum_j U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1
                        coeffs = np.zeros(target_rank)
                        for r in range(target_rank):
                            coeffs[r] = sum(U[i*m+j, r] * V[j*p+k, r] for j in range(m))
                        # Solve coeffs @ W[k*n+i, :] = 1
                        if np.linalg.norm(coeffs) > 1e-10:
                            W_t[k*n+i, :] = np.linalg.lstsq(coeffs.reshape(1, -1), np.array([1.0]), rcond=None)[0]
                        else:
                            W_t[k*n+i, :] = 0
                
                # Update U: for each (i,j), we have sum_k sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # Linear in U[i*m+j, :]
                U_new = np.zeros_like(U_t)
                for i in range(n):
                    for j in range(m):
                        coeffs = np.zeros(target_rank)
                        for r in range(target_rank):
                            coeffs[r] = sum(V[j*p+k, r] * W[k*n+i, r] for k in range(p))
                        if np.linalg.norm(coeffs) > 1e-10:
                            U_t[i*m+j, :] = np.linalg.lstsq(coeffs.reshape(1, -1), np.array([1.0]), rcond=None)[0]
                        else:
                            U_t[i*m+j, :] = 0
                
                # Update V: for each (j,k), we have sum_i sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # Linear in V[j*p+k, :]
                V_new = np.zeros_like(V_t)
                for j in range(m):
                    for k in range(p):
                        coeffs = np.zeros(target_rank)
                        for r in range(target_rank):
                            coeffs[r] = sum(U[i*m+j, r] * W[k*n+i, r] for i in range(n))
                        if np.linalg.norm(coeffs) > 1e-10:
                            V_t[j*p+k, :] = np.linalg.lstsq(coeffs.reshape(1, -1), np.array([1.0]), rcond=None)[0]
                        else:
                            V_t[j*p+k, :] = 0
                
                # Check convergence
                max_diff = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U_t[i*m+j, r] * V_t[j*p+k, r] * W_t[k*n+i, r] for r in range(target_rank))
                            max_diff = max(max_diff, abs(val - 1.0))
                
                if max_diff < 1e-6:
                    converged = True
                    break
            
            if converged:
                best_rank = target_rank
                best_U, best_V, best_W = U_t, V_t, W_t
                break  # Found a valid decomposition at this rank, try lower
    
    # Use the best decomposition found
    if best_rank < rank:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
