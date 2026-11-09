import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Initialize schoolbook
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_rank = rank_schoolbook
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()
    
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    # Try to find a lower rank using ALS
    # We will try ranks from rank_schoolbook - 1 down to 50
    for R in range(rank_schoolbook - 1, 49, -1):
        if time.time() > deadline:
            break
            
        # Initialize with random values or schoolbook subset
        # Using schoolbook subset might be better
        U = U_sb[:, :R].copy()
        V = V_sb[:, :R].copy()
        W = W_sb[:, :R].copy()
        
        # Add some noise to break symmetry
        U += np.random.normal(0, 0.1, U.shape)
        V += np.random.normal(0, 0.1, V.shape)
        W += np.random.normal(0, 0.1, W.shape)
        
        # ALS iterations
        for it in range(50):
            if time.time() > deadline:
                break
                
            # Solve for U given V, W
            # Equation: sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} = 1
            # For fixed i,k: sum_j sum_r U_{i,j,r} (V_{j,k,r} W_{k,i,r}) = 1
            # Let M_{j,r} = V_{j,k,r} W_{k,i,r}
            # Then sum_j sum_r U_{i,j,r} M_{j,r} = 1
            # This is a linear system for U_{i,j,r}
            
            # We can solve for all U at once?
            # The equations are coupled by j and r.
            # Actually, for each (i,k), we have a vector equation for the row of U corresponding to i?
            # No, U has indices (i,j,r).
            # Let's vectorize.
            # U is (nm, R). Flatten to (nm*R,).
            # The equation is linear in U.
            
            # Construct the matrix A for the linear system U_flat = b
            # A has size (n*p) x (n*m*R)
            # This is too large for n=4, m=4, p=7, R=100?
            # n*p = 28. n*m*R = 16*100 = 1600.
            # 28x1600 is small.
            
            A_U = np.zeros((n * p, n * m * R))
            b_U = np.ones(n * p)
            
            for i in range(n):
                for k in range(p):
                    row_idx = k * n + i
                    for j in range(m):
                        for r in range(R):
                            # Term: U_{i,j,r} * V_{j,k,r} * W_{k,i,r}
                            col_idx = (i * m + j) * R + r
                            A_U[row_idx, col_idx] = V[j * p + k, r] * W[k * n + i, r]
                            
            # Solve A_U U_flat = b_U
            try:
                U_flat, _, _, _ = np.linalg.lstsq(A_U, b_U, rcond=None)
                U = U_flat.reshape(n * m, R)
            except:
                break
                
            # Similarly for V
            A_V = np.zeros((m * p, m * p * R))
            b_V = np.ones(m * p)
            for j in range(m):
                for k in range(p):
                    row_idx = j * p + k
                    for i in range(n):
                        for r in range(R):
                            col_idx = (j * p + k) * R + r # Wait, V index is (j,k,r)
                            # Term: U_{i,j,r} * V_{j,k,r} * W_{k,i,r}
                            # For fixed j,k, we sum over i, r.
                            # Sum_i,r U_{i,j,r} W_{k,i,r} V_{j,k,r} = 1
                            # V_{j,k,r} is the variable.
                            # But V_{j,k,r} appears with coefficient U_{i,j,r} W_{k,i,r}
                            # So for each (j,k), the equation is:
                            # Sum_r V_{j,k,r} (Sum_i U_{i,j,r} W_{k,i,r}) = 1
                            # This is not linear in V if we consider all r at once?
                            # Yes it is linear in the vector V_{j,k,:}
                            # Let C_r = Sum_i U_{i,j,r} W_{k,i,r}
                            # Then Sum_r V_{j,k,r} C_r = 1
                            # This is a single equation for each (j,k) involving R variables.
                            # We have m*p equations and m*p*R variables.
                            # Under-determined.
                            # We can use least squares.
                            
                            pass # This logic is flawed for direct construction of A_V
            
            # Let's use a simpler ALS:
            # For V:
            # Sum_i,r U_{i,j,r} W_{k,i,r} V_{j,k,r} = 1
            # For each (j,k), we have a linear system for V_{j,k,:}
            # But it's just one equation: V_{j,k,:} . C_{j,k,:} = 1
            # Where C_{j,k,r} = Sum_i U_{i,j,r} W_{k,i,r}
            # We can choose V_{j,k,:} = C_{j,k,:} / ||C_{j,k,:}||^2
            # This minimizes the norm of V while satisfying the equation.
            
            for j in range(m):
                for k in range(p):
                    C = np.zeros(R)
                    for i in range(n):
                        C += U[i * m + j, :] * W[k * n + i, :]
                    norm_sq = np.dot(C, C)
                    if norm_sq > 1e-10:
                        V[j * p + k, :] = C / norm_sq
                    else:
                        V[j * p + k, :] = 0
                        
            # Similarly for W
            for k in range(p):
                for i in range(n):
                    C = np.zeros(R)
                    for j in range(m):
                        C += U[i * m + j, :] * V[j * p + k, :]
                    norm_sq = np.dot(C, C)
                    if norm_sq > 1e-10:
                        W[k * n + i, :] = C / norm_sq
                    else:
                        W[k * n + i, :] = 0
                        
        # Check error
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(R):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    error += (val - 1.0) ** 2
                    
        if error < 1e-6:
            # Try to round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            error_rounded = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(R):
                            val += U_rounded[i * m + j, r] * V_rounded[j * p + k, r] * W_rounded[k * n + i, r]
                        error_rounded += (val - 1.0) ** 2
                        
            if error_rounded < 1e-6:
                best_rank = R
                best_U = U_rounded.tolist()
                best_V = V_rounded.tolist()
                best_W = W_rounded.tolist()
                # We can break if we find a very low rank, but let's try to go lower
                if R < 60:
                    break
                    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
