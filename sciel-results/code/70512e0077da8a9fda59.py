"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # P1 = (A_0,j0 + A_1,j1)(B_j0,k0 + B_j1,k1)
            U[0*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P2 = (A_1,j0 + A_1,j1) B_j0,k0
            U[1*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            W[k0*n + 1][col] += 1
            W[k1*n + 1][col] -= 1
            col += 1
            
            # P3 = A_0,j0 (B_j0,k1 - B_j1,k1)
            U[0*m + j0][col] += 1
            V[j0*p + k1][col] += 1
            V[j1*p + k1][col] -= 1
            W[k1*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P4 = A_1,j1 (B_j1,k0 - B_j0,k0)
            U[1*m + j1][col] += 1
            V[j1*p + k0][col] += 1
            V[j0*p + k0][col] -= 1
            W[k0*n + 0][col] += 1
            W[k0*n + 1][col] += 1
            col += 1
            
            # P5 = (A_0,j0 + A_0,j1) B_j1,k1
            U[0*m + j0][col] += 1
            U[0*m + j1][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] -= 1
            W[k1*n + 0][col] += 1
            col += 1
            
            # P6 = (A_1,j0 - A_0,j0)(B_j0,k0 + B_j0,k1)
            U[1*m + j0][col] += 1
            U[0*m + j0][col] -= 1
            V[j0*p + k0][col] += 1
            V[j0*p + k1][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P7 = (A_0,j1 - A_1,j1)(B_j1,k0 + B_j1,k1)
            U[0*m + j1][col] += 1
            U[1*m + j1][col] -= 1
            V[j1*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            col += 1
            
        if time.time() > deadline:
            break
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Remaining j=4
    for k in range(p):
        if time.time() > deadline:
            break
        U[0*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 0][col] += 1
        col += 1
        
        U[1*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 1][col] += 1
        col += 1
    
    # Convert to numpy for optimization
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Try to reduce rank by checking if we can eliminate a component
    # via a smooth optimization approach
    best_rank = rank
    best_U, best_V, best_W = U_np, V_np, W_np
    
    # Attempt rank reduction using iterative projection
    for target_rank in range(53, 49, -1):
        if time.time() > deadline:
            break
        if target_rank >= rank:
            continue
        
        # Initialize with subset of current components
        U_t = U_np[:, :target_rank].copy()
        V_t = V_np[:, :target_rank].copy()
        W_t = W_np[:, :target_rank].copy()
        
        # Iteratively refine using alternating least squares
        converged = False
        for iteration in range(200):
            if time.time() > deadline:
                break
            
            # Update U: for each row, solve linear system
            for i in range(n * m):
                # U[i,r] * (V[j*p+k,r] * W[k*n+i',r]) summed over r
                # We need: sum_r U[i,r] * (V[:,r] * W[:,r] reshaped appropriately) = T[i,:,:]
                # Build matrix: M[r, (j,k)] = V[j*p+k, r] * W[k*n+i, r] for the i-th row
                # But W depends on i through k*n+i... this is tricky
                # Let's use a different formulation
                
                # Actually, let's use the tensor unfolding approach
                pass
            
            # Simpler: use gradient-free approach - check if current is good enough
            # and try random perturbations
            break
        
        # Verify if target_rank solution is valid by checking the tensor
        # Build the tensor from current decomposition
        T_recon = np.zeros((n, m, p))
        for r in range(target_rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_recon[i, j, k] += U_t[i*m+j, r] * V_t[j*p+k, r] * W_t[k*n+i, r]
        
        # The true tensor for matrix mult: T[i,j,k] = 1 if j==j and k==k (identity in middle dim)
        # Actually T[i,j,k] = delta_{j,k} for matrix mult (A[i,j]*B[j,k])
        T_true = np.zeros((n, m, p))
        for j in range(m):
            for i in range(n):
                for k in range(p):
                    if j == k:  # Wait, this isn't right
                        T_true[i, j, k] = 1
        # Correct: T[i,j,k] = 1 if the middle index matches, i.e., it's just 1 for all valid (i,j,k)
        # No: the multiplication tensor has T[i,j,k] = 1 for all i,j,k where we multiply A[i,j]*B[j,k]
        # So T[i,j,k] = 1 for all (i,j,k) in [0,n)×[0,m)×[0,p)
        # Wait no. The tensor is T[i,j,k] = delta_{j, j'} where j' is the contraction index
        # For matrix mult C[i,k] = sum_j A[i,j] B[j,k], the tensor is T[i,j,k] = 1 for all i,j,k
        # Actually the structure tensor of matrix mult is T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # No! It's T[i,j,k] = delta where the middle index of A matches the first index of B
        # The tensor entry T[i, j, k] corresponds to the coefficient of A[i,j] * B[j,k] in C[i,k]
        # So T[i,j,k] = 1 for all valid (i,j,k)
        T_true = np.ones((n, m, p))
        
        err = np.max(np.abs(T_recon - T_true))
        if err < 1e-10:
            best_rank = target_rank
            best_U, best_V, best_W = U_t, V_t, W_t
            break
    
    if best_rank < rank:
        # Convert back to lists
        U_out = best_U.tolist()
        V_out = best_V.tolist()
        W_out = best_W.tolist()
        return {"U": U_out, "V": V_out, "W": W_out}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
