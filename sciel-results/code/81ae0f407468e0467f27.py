"""Tensor decomposition for 3x4x4 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 4, 4
    # Tensor T with indices (i, j, k): T[i,j,k] = 1 always (full tensor for matmul)
    # U: (n*m) x R, V: (m*p) x R, W: (p*n) x R
    # Constraint: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    nm = n * m  # 12
    mp = m * p  # 16
    pn = p * n  # 12
    
    best_result = None
    best_rank = n * m * p  # schoolbook rank = 48
    
    # Try ranks from small to large
    for rank in range(12, 49):
        if time.time() > deadline - 5:
            break
        
        # Multiple random restarts for each rank
        max_restarts = 200 if rank < 20 else 50
        if time.time() > deadline - 10:
            max_restarts = min(max_restarts, 10)
        
        found = False
        for restart in range(max_restarts):
            if time.time() > deadline - 3:
                break
            
            # Random initialization
            np.random.seed(restart * 7919 + rank * 104729)
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            # ALS iterations
            converged = False
            for it in range(500):
                if time.time() > deadline - 2:
                    break
                
                # Fix U, V; solve for W
                # For each (k, i): sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                # Let A[k*n+i, c] = sum_j U[i*m+j, c] * V[j*p+k, c]
                # Then A @ W_col = 1 where W_col is W[k*n+i, :]
                # Actually: sum_c A[k*n+i, c] * W[k*n+i, c] = 1
                # This is underdetermined. Better approach:
                # sum_c (sum_j U[i*m+j,c]*V[j*p+k,c]) * W[k*n+i,c] = 1
                # Define M[k*n+i, c] = sum_j U[i*m+j, c] * V[j*p+k, c]
                # Then M @ W[k*n+i, :] = 1 (scalar)
                # This gives one equation per (k,i) with rank unknowns - underdetermined
                # 
                # Better: treat it as a linear system. The full constraint is:
                # For all (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # 
                # For fixed U,V: the constraint on W is linear.
                # Let's build the system: unknowns are W[k*n+i, c] for all k,i,c
                # Each (i,j,k) gives: sum_c W[k*n+i, c] * (U[i*m+j,c] * V[j*p+k,c]) = 1
                # 
                # Number of equations: n*m*p = 48
                # Number of unknowns: p*n*rank = 12*rank
                # For rank >= 4, we have more unknowns than equations
                # 
                # Let's solve via least squares
                # Build matrix B of size (n*m*p) x (p*n*rank)
                # B[(i,j,k), (k,i,c)] = U[i*m+j,c] * V[j*p+k,c]
                
                # Build B matrix
                B = np.zeros((n*m*p, pn*rank))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            eq_idx = i * m * p + j * p + k
                            for c in range(rank):
                                col_idx = k * n * rank + i * rank + c
                                B[eq_idx, col_idx] = U[i*m+j, c] * V[j*p+k, c]
                
                rhs = np.ones(n*m*p)
                # Solve B @ W_flat = rhs via least squares
                W_flat, residuals, rank_val, sv = np.linalg.lstsq(B, rhs, rcond=None)
                W = W_flat.reshape(pn, rank)
                
                # Check convergence: compute error
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                            error += (val - 1.0)**2
                
                if error < 1e-10:
                    converged = True
                    break
                
                # Fix U, W; solve for V
                B2 = np.zeros((n*m*p, mp*rank))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            eq_idx = i * m * p + j * p + k
                            for c in range(rank):
                                col_idx = j * p * rank + k * rank + c
                                B2[eq_idx, col_idx] = U[i*m+j, c] * W[k*n+i, c]
                
                V_flat, _, _, _ = np.linalg.lstsq(B2, rhs, rcond=None)
                V = V_flat.reshape(mp, rank)
                
                # Fix V, W; solve for U
                B3 = np.zeros((n*m*p, nm*rank))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            eq_idx = i * m * p + j * p + k
                            for c in range(rank):
                                col_idx = i * m * rank + j * rank + c
                                B3[eq_idx, col_idx] = V[j*p+k, c] * W[k*n+i, c]
                
                U_flat, _, _, _ = np.linalg.lstsq(B3, rhs, rcond=None)
                U = U_flat.reshape(nm, rank)
                
                if it % 50 == 49:
                    # Check error periodically
                    error = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                                error += (val - 1.0)**2
                    if error < 1e-10:
                        converged = True
                        break
            
            if converged:
                # Verify and return
                U_list = [[U[i, c] for c in range(rank)] for i in range(nm)]
                V_list = [[V[i, c] for c in range(rank)] for i in range(mp)]
                W_list = [[W[i, c] for c in range(rank)] for i in range(pn)]
                
                # Final verification
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U_list[i*m+j][c] * V_list[j*p+k][c] * W_list[k*n+i][c] for c in range(rank))
                            if abs(val - 1.0) > 1e-8:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    found = True
                    break
        
        if found:
            break
    
    if best_result is None:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][column] = 1
                    V[j*p+k][column] = 1
                    W[k*n+i][column] = 1
                    column += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
