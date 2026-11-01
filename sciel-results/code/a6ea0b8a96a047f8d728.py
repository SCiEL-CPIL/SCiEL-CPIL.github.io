"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    n, m, p = DIMENSIONS  # 2, 2, 8
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    
    # Target: find decomposition with rank < 32
    # The constraint: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # We have n*m*p = 32 constraints, 3*r unknowns (4r + 16r + 16r = 36r)
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try ranks from 31 down to some minimum
    # For each rank, try to find a solution numerically
    for target_rank in range(rank_sb - 1, 15, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        found = False
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Random initialization
            np.random.seed(attempt * 1000 + target_rank)
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # Use iterative refinement: project onto the constraint manifold
            # The constraint is bilinear in pairs, so we can use alternating least squares
            # Fix V and W, solve for U; fix U and W, solve for V; fix U and V, solve for W
            
            success = False
            for iteration in range(200):
                if time.time() > deadline:
                    break
                
                # For each (i,j), the equation is:
                # sum_k' W[k'*n+i, r] * V[j*p+k', r] * U[i*m+j, r] = 1
                # This is linear in U[i*m+j, :] when V and W are fixed
                # Specifically: U[i*m+j, :] dot (V[j*p+k', :] * W[k'*n+i, :]) = 1 for all k'
                # Wait, that's not quite right. Let me re-derive.
                
                # The full constraint: for each (i,j,k):
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # 
                # Fix V and W. For a fixed (i,j), we have p equations:
                # For each k: sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1
                # This is a linear system in U[i*m+j, :]:
                # M_{i,j} @ U[i*m+j, :] = 1_p
                # where M_{i,j}[k, r] = V[j*p+k, r] * W[k*n+i, r]
                
                # Solve for U
                for i in range(n):
                    for j in range(m):
                        # Build the linear system
                        M_ij = np.zeros((p, target_rank))
                        for k in range(p):
                            M_ij[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        # Solve M_ij @ u = ones(p)
                        try:
                            u, residuals, rank_m, sv = np.linalg.lstsq(M_ij, np.ones(p), rcond=None)
                            U[i * m + j, :] = u
                        except:
                            pass
                
                # Fix U and W. For each (j,k), the equation is:
                # sum_r V[j*p+k, r] * (U[i*m+j, r] * W[k*n+i, r]) = 1 for all i
                # For fixed (j,k), we have n equations:
                # For each i: sum_r V[j*p+k, r] * (U[i*m+j, r] * W[k*n+i, r]) = 1
                # This is linear in V[j*p+k, :]:
                # N_{j,k} @ V[j*p+k, :] = 1_n
                # where N_{j,k}[i, r] = U[i*m+j, r] * W[k*n+i, r]
                
                for j in range(m):
                    for k in range(p):
                        N_jk = np.zeros((n, target_rank))
                        for i in range(n):
                            N_jk[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        try:
                            v, residuals, rank_m, sv = np.linalg.lstsq(N_jk, np.ones(n), rcond=None)
                            V[j * p + k, :] = v
                        except:
                            pass
                
                # Fix U and V. For each (k,i), the equation is:
                # sum_r W[k*n+i, r] * (U[i*m+j, r] * V[j*p+k, r]) = 1 for all j
                # For fixed (k,i), we have m equations:
                # For each j: sum_r W[k*n+i, r] * (U[i*m+j, r] * V[j*p+k, r]) = 1
                # This is linear in W[k*n+i, :]:
                # P_{k,i} @ W[k*n+i, :] = 1_m
                # where P_{k,i}[j, r] = U[i*m+j, r] * V[j*p+k, r]
                
                for k in range(p):
                    for i in range(n):
                        P_ki = np.zeros((m, target_rank))
                        for j in range(m):
                            P_ki[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            w, residuals, rank_m, sv = np.linalg.lstsq(P_ki, np.ones(m), rcond=None)
                            W[k * n + i, :] = w
                        except:
                            pass
                
                # Check convergence
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                
                if max_err < 1e-8:
                    success = True
                    break
            
            if success:
                # Try to round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify rounded solution
                max_err_r = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_rounded[i * m + j, :] * V_rounded[j * p + k, :] * W_rounded[k * n + i, :])
                            err = abs(val - 1.0)
                            if err > max_err_r:
                                max_err_r = err
                
                if max_err_r < 1e-9:
                    found = True
                    best_rank = target_rank
                    best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                    break
                else:
                    # Use the unrounded solution but it must have integer/half-integer entries
                    # So this attempt fails for the strict requirement
                    pass
        
        if found:
            break
    
    # Convert to lists for return
    U_list = [[float(U_list_row) for U_list_row in row] for row in best_U.tolist()]
    V_list = [[float(V_list_row) for V_list_row in row] for row in best_V.tolist()]
    W_list = [[float(W_list_row) for W_list_row in row] for row in best_W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
