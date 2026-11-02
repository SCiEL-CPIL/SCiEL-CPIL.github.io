import numpy as np
import time

def construct():
    _start = time.time()
    _deadline = _start + 120.0
    n, m, p = 2, 4, 6
    
    # Build the tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i*m + k
                b = k*p + j
                c = j*n + i
                T[a, b, c] = 1.0
    
    # Try ALS for different ranks
    best_rank = 42
    best_U, best_V, best_W = None, None, None
    
    for rank in [40, 41]:
        if time.time() > _deadline - 30.0:
            break
        
        np.random.seed(42)
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        converged = False
        for iteration in range(200):
            if time.time() > _deadline - 10.0:
                break
            
            # Update U
            # For each a: (W ⊗ V) @ U[a,:] = vec(T[a,:,:])
            WV = np.kron(W, V)  # shape (p*n * m*p, rank)
            for a in range(n*m):
                rhs = T[a, :, :].reshape(-1)
                # Solve least squares
                U[a, :], residuals, rank_, sv = np.linalg.lstsq(WV, rhs, rcond=None)
            
            # Update V
            # For each b: (W ⊗ U) @ V[b,:] = vec(T[:,b,:])
            WU = np.kron(W, U)  # shape (p*n * n*m, rank)
            for b in range(m*p):
                rhs = T[:, b, :].reshape(-1)
                V[b, :], residuals, rank_, sv = np.linalg.lstsq(WU, rhs, rcond=None)
            
            # Update W
            # For each c: (V ⊗ U) @ W[c,:] = vec(T[:,:,c])
            VU = np.kron(V, U)  # shape (m*p * n*m, rank)
            for c in range(p*n):
                rhs = T[:, :, c].reshape(-1)
                W[c, :], residuals, rank_, sv = np.linalg.lstsq(VU, rhs, rcond=None)
            
            # Check convergence
            error = np.sum((np.einsum('ar,br,cr->abc', U, V, W) - T)**2)
            if error < 1e-10:
                converged = True
                break
        
        if converged:
            # Check if entries are close to integer or half-integer
            all_close = True
            for arr in [U, V, W]:
                for val in arr:
                    # Check if close to integer or half-integer
                    closest_int = round(val)
                    closest_half = round(2*val)/2
                    if abs(val - closest_int) > 1e-6 and abs(val - closest_half) > 1e-6:
                        all_close = False
                        break
                if not all_close:
                    break
            
            if all_close:
                # Round to nearest integer or half-integer
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                
                # Verify
                error = np.sum((np.einsum('ar,br,cr->abc', U, V, W) - T)**2)
                if error < 1e-10:
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W
                    break
    
    # Convert to list of lists
    if best_U is not None:
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fall back to Strassen
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen(a, b, c, d, e, f, g, h, c11, c12, c21, c22, col):
        U[a][col] += 1
        U[d][col] += 1
        V[e][col] += 1
        V[h][col] += 1
        W[c11][col] += 1
        W[c22][col] += 1
        col += 1
        
        U[c][col] += 1
        U[d][col] += 1
        V[e][col] += 1
        W[c21][col] += 1
        W[c22][col] -= 1
        col += 1
        
        U[a][col] += 1
        V[f][col] += 1
        V[h][col] -= 1
        W[c12][col] += 1
        W[c22][col] += 1
        col += 1
        
        U[d][col] += 1
        V[g][col] += 1
        V[e][col] -= 1
        W[c11][col] += 1
        W[c21][col] += 1
        col += 1
        
        U[a][col] += 1
        U[b][col] += 1
        V[h][col] += 1
        W[c11][col] -= 1
        W[c12][col] += 1
        col += 1
        
        U[c][col] += 1
        U[a][col] -= 1
        V[e][col] += 1
        V[f][col] += 1
        W[c22][col] += 1
        col += 1
        
        U[b][col] += 1
        U[d][col] -= 1
        V[g][col] += 1
        V[h][col] += 1
        W[c11][col] += 1
        col += 1
        
        return col
    
    col = 0
    for k in range(3):
        if time.time() > _deadline - 5.0:
            break
        col = add_strassen(
            0, 1, 4, 5,
            2*k, 2*k+1, 6+2*k, 6+2*k+1,
            2*k*2, (2*k+1)*2, 2*k*2+1, (2*k+1)*2+1,
            col
        )
        if time.time() > _deadline - 5.0:
            break
        col = add_strassen(
            2, 3, 6, 7,
            12+2*k, 12+2*k+1, 18+2*k, 18+2*k+1,
            2*k*2, (2*k+1)*2, 2*k*2+1, (2*k+1)*2+1,
            col
        )
    
    return {"U": U, "V": V, "W": W}
