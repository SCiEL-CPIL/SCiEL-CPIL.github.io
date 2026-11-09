import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 30
    best_U, best_V, best_W = None, None, None
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100  # 100s limit, leaving 20s margin
    
    # Try ranks from 15 to 29
    for rank in range(15, 30):
        if time.time() > deadline:
            break
            
        # Try a few random restarts for this rank
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Random initialization
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # ALS Iterations
            for it in range(100):
                if time.time() > deadline:
                    break
                
                # Solve for U
                # U_{i,j,r} * sum_k V_{j,k,r} W_{k,i,r} = 1
                # This is not linear in U if V,W are fixed?
                # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
                # For fixed V, W, the term for U_{i,j,r} is multiplied by (V * W)_{j,k,r}? No.
                # Let's rewrite.
                # We want sum_r U_{i,j,r} M_{r} = 1, where M_r depends on V,W?
                # No, the equation is for each (i,j,k).
                # sum_r U_{i,j,r} (V_{j,k,r} W_{k,i,r}) = 1.
                # This is linear in U_{i,j,r}.
                # For a fixed (i,j), we have p equations (for each k) and rank unknowns (U_{i,j,0..r-1}).
                # So we can solve a least squares problem for each (i,j).
                
                # Construct matrix A_u for this (i,j)
                # A_u[k, r] = V[j, k] * W[k, i]  (Note: V index is j*p+k, W index is k*n+i)
                # We need to map indices correctly.
                
                # Let's build the linear system for U
                # For each i in n, j in m:
                # Unknowns: U[i*m + j, :] (vector of length rank)
                # Equations: for each k in p:
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                
                # We can form a matrix B of shape (p, rank) where B[k, r] = V[j*p+k, r] * W[k*n+i, r]
                # Then B @ U[i*m+j, :] = 1 (vector of ones)
                
                new_U = np.zeros((n * m, rank))
                for i in range(n):
                    for j in range(m):
                        B = np.zeros((p, rank))
                        for k in range(p):
                            for r in range(rank):
                                B[k, r] = V[j * p + k, r] * W[k * n + i, r]
                        # Solve B @ x = 1
                        x, residuals, rank_sol, s = np.linalg.lstsq(B, np.ones(p), rcond=None)
                        new_U[i * m + j, :] = x
                U = new_U
                
                # Solve for V
                new_V = np.zeros((m * p, rank))
                for j in range(m):
                    for k in range(p):
                        # Unknowns: V[j*p+k, :]
                        # Equations: for each i in n:
                        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                        # B[i, r] = U[i*m+j, r] * W[k*n+i, r]
                        B = np.zeros((n, rank))
                        for i in range(n):
                            for r in range(rank):
                                B[i, r] = U[i * m + j, r] * W[k * n + i, r]
                        x, residuals, rank_sol, s = np.linalg.lstsq(B, np.ones(n), rcond=None)
                        new_V[j * p + k, :] = x
                V = new_V
                
                # Solve for W
                new_W = np.zeros((p * n, rank))
                for k in range(p):
                    for i in range(n):
                        # Unknowns: W[k*n+i, :]
                        # Equations: for each j in m:
                        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                        # B[j, r] = U[i*m+j, r] * V[j*p+k, r]
                        B = np.zeros((m, rank))
                        for j in range(m):
                            for r in range(rank):
                                B[j, r] = U[i * m + j, r] * V[j * p + k, r]
                        x, residuals, rank_sol, s = np.linalg.lstsq(B, np.ones(m), rcond=None)
                        new_W[k * n + i, :] = x
                W = new_W
                
                # Check error
                # Compute tensor reconstruction
                error = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for r in range(rank):
                                val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                            error += (val - 1) ** 2
                
                if error < 1e-6:
                    # Check if entries are close to half-integers
                    valid = True
                    for mat in [U, V, W]:
                        for val in mat:
                            if abs(val * 2 - round(val * 2)) > 1e-4:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        # Round to half-integers
                        U = np.round(U * 2) / 2
                        V = np.round(V * 2) / 2
                        W = np.round(W * 2) / 2
                        
                        # Verify exactness
                        exact = True
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    val = 0
                                    for r in range(rank):
                                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                                    if abs(val - 1) > 1e-9:
                                        exact = False
                                        break
                                if not exact:
                                    break
                            if not exact:
                                break
                        if exact:
                            return {
                                "U": U.tolist(),
                                "V": V.tolist(),
                                "W": W.tolist()
                            }
                
    # Fallback to schoolbook
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
