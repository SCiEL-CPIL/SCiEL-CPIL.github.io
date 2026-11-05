import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline for safety
    
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (16, rank)
    # W: (p*n, rank) = (12, rank)
    # W[k*n+i, r] corresponds to C[i,k]
    
    # Target tensor: T[i,j,k] = 1 if A[i,j] * B[j,k] = C[i,k]
    # In our indexing:
    # A index: i*m + j = i*4 + j (i=0..2, j=0..3)
    # B index: j*p + k = j*4 + k (j=0..3, k=0..3)
    # C index (W): k*n + i = k*3 + i (k=0..3, i=0..2)
    
    # The tensor is: sum over j of A[i,j] * B[j,k] = C[i,k]
    # So T[a_idx, b_idx, c_idx] = 1 if a_idx = i*4+j, b_idx = j*4+k, c_idx = k*3+i
    
    # Let's build the target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i*m + j
                b_idx = j*p + k
                c_idx = k*n + i
                T[a_idx, b_idx, c_idx] = 1.0
    
    # Method 1: Laderman projection (rank 23)
    # Method 2: ALS optimization
    
    best_rank = n*m*p  # 48 schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook decomposition (rank 48)
    U_school = np.zeros((n*m, n*m*p))
    V_school = np.zeros((m*p, n*m*p))
    W_school = np.zeros((p*n, n*m*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i*m*p + j*p + k
                U_school[i*m+j, r] = 1
                V_school[j*p+k, r] = 1
                W_school[k*n+i, r] = 1
    
    def verify(U, V, W, T_target, tol=1e-10):
        """Verify that U, V, W decompose T_target."""
        rank = U.shape[1]
        # Reconstruct tensor: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # Use einsum for efficiency
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(T_recon, T_target, atol=tol)
    
    # Try Laderman-based approach
    # I'll use a known Laderman 4x4x4 decomposition
    
    # Actually, let me try ALS to find a low-rank decomposition
    def als_decompose(T_target, rank, max_iter=1000, tol=1e-12):
        """ALS decomposition of tensor T into sum of rank-1 terms."""
        n1, n2, n3 = T_target.shape
        np.random.seed(42)
        U = np.random.randn(n1, rank) * 0.1
        V = np.random.randn(n2, rank) * 0.1
        W = np.random.randn(n3, rank) * 0.1
        
        for it in range(max_iter):
            # Update U: for each i, U[i,:] = (V^T W^T T[:,:,i]) / (V^T W^T V W)
            # Actually: T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
            # Fix V, W: T[:,:,i] = U[i,:] (V W^T)^T... 
            # T[i,j,k] = sum_r U[i,r] (V[j,r] W[k,r])
            # Let M[j,k] = sum_r V[j,r] W[k,r] = V @ W.T
            # T[i,:,:] = U[i,:] @ ... no
            # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
            # = sum_r U[i,r] * (V @ W.T)[j,k]... no, that's not right either.
            # 
            # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
            # For fixed i: T[i,j,k] = sum_r U[i,r] * (V[j,r] * W[k,r])
            # Let's define: for each r, the outer product V[:,r] * W[:,r] is a matrix M_r
            # T[i,:,:] = sum_r U[i,r] * M_r
            # So: T[i,:,:] (as a matrix) = sum_r U[i,r] * (V[:,r] W[:,r]^T)
            # 
            # To solve for U[i,:]: 
            # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
            # This is a linear system in U[i,:]:
            # For fixed i, we have n2*n3 equations in rank unknowns.
            # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
            # Let's vectorize: T[i,:,:].flatten() = (V * W) . U[i,:]
            # where (V*W) is a matrix of size (n2*n3, rank) with entries (V*W)[(j,k),r] = V[j,r]*W[k,r]
            
            # Build the design matrix
            # This is expensive for large sizes, but our sizes are small
            
            # Update U
            for i in range(n1):
                # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
                # Vectorize over (j,k):
                rhs = T_target[i].flatten()  # size n2*n3
                # Design matrix: D[(j,k), r] = V[j,r] * W[k,r]
                D = np.zeros((n2*n3, rank))
                for j in range(n2):
                    for k in range(n3):
                        D[j*n3+k, :] = V[j,:] * W[k,:]
                # Solve D @ U[i,:] = rhs
                U[i,:] = np.linalg.lstsq(D, rhs, rcond=None)[0]
            
            # Update V
            for j in range(n2):
                rhs = T_target[:, j, :].flatten()  # size n1*n3
                D = np.zeros((n1*n3, rank))
                for i in range(n1):
                    for k in range(n3):
                        D[i*n3+k, :] = U[i,:] * W[k,:]
                V[j,:] = np.linalg.lstsq(D, rhs, rcond=None)[0]
            
            # Update W
            for k in range(n3):
                rhs = T_target[:, :, k].flatten()  # size n1*n2
                D = np.zeros((n1*n2, rank))
                for i in range(n1):
                    for j in range(n2):
                        D[i*n2+j, :] = U[i,:] * V[j,:]
                W[k,:] = np.linalg.lstsq(D, rhs, rcond=None)[0]
            
            # Check convergence
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.max(np.abs(T_recon - T_target))
            if err < tol:
                break
            
            if time.time() > deadline - 10:
                break
        
        return U, V, W, err
    
    # Try different ranks
    for rank in range(16, 45):
        if time.time() > deadline - 30:
            break
        
        U, V, W, err = als_decompose(T, rank, max_iter=500)
        
        if err < 1e-10:
            # Try to round to half-integers
            U_round = np.round(U * 2) / 2
            V_round = np.round(V * 2) / 2
            W_round = np.round(W * 2) / 2
            
            T_recon = np.einsum('ir,jr,kr->ijk', U_round, V_round, W_round)
            if np.allclose(T_recon, T, atol=1e-10):
                best_rank = rank
                best_U = U_round
                best_V = V_round
                best_W = W_round
                break
        
        # Also try without rounding
        if err < 1e-10 and rank < best_rank:
            # Check if entries are already close to half-integers
            U_close = np.all(np.abs(U * 2 - np.round(U * 2)) < 1e-8)
            V_close = np.all(np.abs(V * 2 - np.round(V * 2)) < 1e-8)
            W_close = np.all(np.abs(W * 2 - np.round(W * 2)) < 1e-8)
            if U_close and V_close and W_close:
                best_rank = rank
                best_U = np.round(U * 2) / 2
                best_V = np.round(V * 2) / 2
                best_W = np.round(W * 2) / 2
    
    # If no better solution found, use schoolbook
    if best_U is None:
        best_rank = n*m*p
        best_U = U_school
        best_V = V_school
        best_W = W_school
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
