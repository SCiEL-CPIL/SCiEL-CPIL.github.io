"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

# Use a fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 8 matrix.
    
    The matrix multiplication tensor T has entries T[i,j,k] = 1 if (i,j,k) corresponds to a valid
    (row, col_in_A, col_in_B) triple where A is n×m and B is m×p.
    
    We want to decompose T as sum_{r=1}^{rank} U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    where U is (n*m)×rank, V is (m*p)×rank, W is (p*n)×rank.
    
    The schoolbook rank is n*m*p = 3*4*8 = 96.
    
    We try to find a lower rank decomposition using random initialization and local search.
    """
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the matrix multiplication tensor as a 3D array
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Start with schoolbook rank
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds to leave safety margin
    
    # Try to find a lower rank decomposition
    # Start from rank = best_rank - 1 and go down
    for target_rank in range(best_rank - 1, 10, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            # Random initialization with small values
            U = np.random.randn(nm, target_rank) * 0.1
            V = np.random.randn(mp, target_rank) * 0.1
            W = np.random.randn(pn, target_rank) * 0.1
            
            # Use ALS (Alternating Least Squares) to fit the tensor
            # For each factor, fix the others and solve the linear system
            converged = False
            for iteration in range(50):
                if time.time() > deadline:
                    break
                
                # Update U: fix V and W
                # T[i,j,k] ≈ sum_r U[i,r] V[j,r] W[k,r]
                # For each i, we have: sum_j sum_k T[i,j,k] V[j,r] W[k,r] = sum_r U[i,r] * (V[j,r] W[k,r])
                # This is a linear system in U[i,:]
                
                # Build the linear system for U
                # For each i: sum_{j,k} T[i,j,k] * (V ⊙ W)[j,k,r] = sum_r' U[i,r'] * ...
                # Actually, let's think of this differently.
                
                # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
                # Fix V and W, for each i:
                # sum_{j,k} T[i,j,k] delta_{j,j'} delta_{k,k'} = U[i,:] dot (V[j',:] * W[k',:])
                # So we need to solve: U[i,:] * M_i = b_i where M_i[j',k'] = V[j',r]*W[k',r] summed over r... 
                
                # Let me reformulate. The tensor T can be matricized.
                # T_{(i)(j,k)} = sum_r U[i,r] (V ⊙ W)[(j,k),r]
                # So T_mat_U = U @ (V ⊙ W).T
                # where T_mat_U[i, (j,k)] = T[i,j,k]
                
                # Similarly T_mat_V = V @ (W ⊙ U).T where T_mat_V[j, (k,i)] = T[i,j,k]
                # And T_mat_W = W @ (U ⊙ V).T where T_mat_W[k, (i,j)] = T[i,j,k]
                
                # Update U:
                # T_mat_U = U @ (V ⊙ W).T
                # So U = T_mat_U @ (V ⊙ W) @ pinv((V ⊙ W).T @ (V ⊙ W))
                
                VW = np.einsum('jr,kr->jkr', V, W).reshape(mp * pn, target_rank)
                T_mat_U = T.reshape(nm, mp * pn)
                try:
                    U_new = T_mat_U @ VW @ np.linalg.pinv(VW.T @ VW)
                    U = U_new
                except:
                    pass
                
                # Update V:
                # T_mat_V[j, (k,i)] = T[i,j,k]
                # We need to reshape T to (mp, pn) where the second index is (k,i)
                # T[i,j,k] -> T_mat_V[j, k*n+i]
                T_mat_V = np.zeros((mp, pn))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_mat_V[j*p + k, k*n + i] = T[i*m + j, j*p + k, k*n + i]
                
                WU = np.einsum('kr,ir->kir', W, U).reshape(pn, target_rank)
                # Wait, we need (W ⊙ U) where the first index is (k,i)
                # W[k,r] * U[i,r] for each (k,i)
                WU = np.einsum('kr,ir->kir', W, U).reshape(pn, target_rank)
                try:
                    V_new = T_mat_V @ WU @ np.linalg.pinv(WU.T @ WU)
                    V = V_new
                except:
                    pass
                
                # Update W:
                # T_mat_W[k, (i,j)] = T[i,j,k]
                # T[i,j,k] -> T_mat_W[k, i*m+j]
                T_mat_W = np.zeros((pn, nm))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_mat_W[k*n + i, i*m + j] = T[i*m + j, j*p + k, k*n + i]
                
                UV = np.einsum('ir,jr->ijr', U, V).reshape(nm, target_rank)
                try:
                    W_new = T_mat_W @ UV @ np.linalg.pinv(UV.T @ UV)
                    W = W_new
                except:
                    pass
                
                # Check convergence
                if iteration > 0:
                    # Compute residual
                    T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
                    residual = np.linalg.norm(T - T_approx) / (np.linalg.norm(T) + 1e-10)
                    if residual < 1e-8:
                        converged = True
                        break
            
            # Verify the solution
            T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
            residual = np.linalg.norm(T - T_approx) / (np.linalg.norm(T) + 1e-10)
            
            if residual < 1e-6:
                # We found a valid decomposition of rank target_rank
                best_rank = target_rank
                best_U = U
                best_V = V
                best_W = W
                break  # Break out of attempt loop, try lower rank
    
    # If we didn't find a better rank, use schoolbook
    if best_U is None:
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        best_U, best_V, best_W = U, V, W
    
    # Convert to lists of lists for the return format
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
