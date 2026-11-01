"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # Build the tensor T with T[a,b,c] = 1 for valid (i,j,k)
    # a = i*m+j, b = j*p+k, c = k*n+i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try to find a low-rank decomposition
    # Start with a high rank and try to reduce
    # Use a greedy approach: iteratively find the best rank-1 approximation
    
    def tensor_to_matrix(T, mode):
        """Reshape tensor to matrix along given mode."""
        if mode == 0:
            return T.reshape(n*m, (m*p)*(p*n))
        elif mode == 1:
            return T.transpose(1,0,2).reshape(m*p, (n*m)*(p*n))
        else:
            return T.transpose(2,0,1).reshape(p*n, (n*m)*(m*p))
    
    def matrix_to_tensor(M, mode, shape):
        """Reshape matrix back to tensor."""
        if mode == 0:
            return M.reshape(shape)
        elif mode == 1:
            return M.reshape(shape[1], shape[0], shape[2]).transpose(1,0,2)
        else:
            return M.reshape(shape[2], shape[0], shape[1]).transpose(2,0,1)
    
    shape = (n*m, m*p, p*n)
    
    # Start with schoolbook rank
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Initialize with schoolbook decomposition
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Try to reduce rank by finding linear dependencies
    # Use SVD-based approach: try to merge columns
    
    # First, let's try a randomized approach to find lower rank
    # We'll use the fact that if rank is r, then T = sum_{c=0}^{r-1} u_c ⊗ v_c ⊗ w_c
    
    # Try different ranks from low to high
    # For 2x2x10, let's try ranks from 15 up to 40
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to find a decomposition of various ranks
    # Use iterative refinement
    
    for target_rank in range(15, 41):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            # Initialize randomly
            U_r = np.random.randn(n*m, target_rank)
            V_r = np.random.randn(m*p, target_rank)
            W_r = np.random.randn(p*n, target_rank)
            
            # Iterative refinement
            for iteration in range(100):
                if time.time() > deadline - 5:
                    break
                
                # Update U: for each column c, U[:,c] = argmin ||T - sum u⊗v⊗w||
                # This is equivalent to: U[:,c] = <T, v_c ⊗ w_c> reshaped
                for c in range(target_rank):
                    # Contract T with v_c and w_c
                    temp = np.tensordot(T, V_r[:, c], axes=([1], [0]))
                    temp = np.tensordot(temp, W_r[:, c], axes=([1], [0]))
                    U_r[:, c] = temp.reshape(n*m)
                
                # Update V
                for c in range(target_rank):
                    temp = np.tensordot(T, U_r[:, c], axes=([0], [0]))
                    temp = np.tensordot(temp, W_r[:, c], axes=([1], [0]))
                    V_r[:, c] = temp.reshape(m*p)
                
                # Update W
                for c in range(target_rank):
                    temp = np.tensordot(T, U_r[:, c], axes=([0], [0]))
                    temp = np.tensordot(temp, V_r[:, c], axes=([1], [0]))
                    W_r[:, c] = temp.reshape(p*n)
            
            # Check if decomposition is exact
            T_recon = np.zeros_like(T)
            for c in range(target_rank):
                T_recon += np.outer(np.outer(U_r[:, c], V_r[:, c]), W_r[:, c]).reshape(shape)
            
            error = np.max(np.abs(T - T_recon))
            if error < 1e-6:
                best_rank = target_rank
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                break
        
        if best_rank == target_rank:
            break
    
    # Convert to lists with integer/half-integer entries if possible
    def to_list(mat):
        lst = []
        for row in mat:
            lst.append([float(x) for x in row])
        return lst
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
