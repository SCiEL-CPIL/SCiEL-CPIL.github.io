"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]
import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # safety margin
    
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i, j, k] = 1 if the schoolbook term A[i//m, i%m] * B[i%m, k//n] contributes to C[k%n, i//m]
    # Using the convention: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # Output C[r, s] = sum_t U[r, t] * V[?, t] * W[?, t]
    # The trace convention: W[k*n + i, r] where i is row of A/C, k is col of B/C
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # In tensor form: T[i*m+j, j*p+k, k*n+i] = 1
    
    # Build tensor as a 3D array
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Schoolbook decomposition (rank = n*m*p = 60)
    rank_schoolbook = n * m * p
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i*m + j*p + k  # unique index
                # Actually let's use sequential
                pass
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1.0
                V_school[j*p + k, col] = 1.0
                W_school[k*n + i, col] = 1.0
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    
    # Try to find lower rank decompositions using iterative refinement
    # Strategy: Start with schoolbook, randomly zero out columns and try to fix
    
    # Use a greedy approach: try to merge pairs of rank-1 tensors
    # Two rank-1 tensors (u1,v1,w1) and (u2,v2,w2) can be merged if
    # their sum can be expressed as a single rank-1 tensor, which requires
    # u1+u2, v1+v2, w1+w2 to be rank-1 in the appropriate sense.
    
    # Alternative: Use numerical CP decomposition
    # For small tensors, we can use the "parallel factor" algorithm
    
    def verify(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose T."""
        r = U.shape[1]
        # Compute the reconstructed tensor
        # T_recon[a, b, c] = sum_t U[a,t] * V[b,t] * W[c,t]
        # Use einsum for efficiency
        T_recon = np.einsum('at,bt,ct->abc', U, V, W)
        diff = np.max(np.abs(T_recon - T))
        return diff < tol
    
    # Try various rank values from 55 down to 40
    for target_rank in range(55, 39, -1):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations
        for trial in range(5):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random perturbation of schoolbook
            U_init = np.random.randn(n*m, target_rank) * 0.1
            V_init = np.random.randn(m*p, target_rank) * 0.1
            W_init = np.random.randn(p*n, target_rank) * 0.1
            
            # Add some structure: copy from schoolbook where possible
            for t in range(min(target_rank, rank_schoolbook)):
                # Find which schoolbook column to use
                src = t % rank_schoolbook
                U_init[:, t] = U_school[:, src]
                V_init[:, t] = V_school[:, src]
                W_init[:, t] = W_school[:, src]
            
            # Use ALS (Alternating Least Squares) to refine
            # For CP decomposition, ALS minimizes ||T - sum u_t v_t w_t||_F
            # But we need EXACT decomposition, so ALS is just a heuristic
            
            # Instead, use a different approach: 
            # The tensor equation is linear in each factor given the other two.
            # Fix V and W, solve for U: U * (V^T ⊙ W) = T_unfolded
            # where ⊙ is the Khatri-Rao product
            
            # Khatri-Rao product of V (m*p x r) and W (p*n x r):
            # KRW = kron-like: KRW[a, b, t] = V[a, t] * W[b, t]
            # Shape: (m*p * p*n, r)
            
            # T_unfolded: reshape T to (n*m, m*p * p*n)
            T_unfold = T.reshape(n*m, m*p * p*n)
            
            # For fixed V, W: U * KRW^T = T_unfold
            # where KRW has shape (m*p*p*n, r), KRW[(a * p*n + b), t] = V[a,t] * W[b,t]
            
            # This is a linear system: U = T_unfold * KRW^+ (pseudo-inverse)
            # But we want U such that U @ KRW.T = T_unfold exactly.
            
            # Let's do ALS iterations
            U_curr = U_init.copy()
            V_curr = V_init.copy()
            W_curr = W_init.copy()
            
            for iteration in range(100):
                if time.time() > deadline - 3:
                    break
                
                # Update U: for each row a, U[a,:] such that sum_t U[a,t]*V[:,t]*W[:,t] matches T[a,:,:]
                # T[a, b, c] = sum_t U[a,t] * V[b,t] * W[c,t]
                # For fixed V, W: T_unfold[a, :] = U[a, :] @ KRW[:, :].T
                # where KRW[:, t] = V[:, t] ⊗ W[:, t] (Kronecker product)
                
                # KRW shape: (m*p * p*n, r)
                KRW = np.einsum('at,bt->abt', V_curr, W_curr).reshape(m*p * p*n, target_rank)
                
                # Solve U_curr @ KRW.T = T_unfold
                # U_curr = T_unfold @ pinv(KRW.T)
                try:
                    KRW_pinv = np.linalg.pinv(KRW.T)
                    U_curr = T_unfold @ KRW_pinv
                except:
                    break
                
                # Update V: for each row b, V[b,:] such that sum_t U[:,t]*V[b,t]*W[:,t] matches T[:,b,:]
                # T[a, b, c] = sum_t U[a,t] * V[b,t] * W[c,t]
                # For fixed U, W: T_unfold2[:, b] = U @ (V[b,:] ⊙ W[:, :]).T ... 
                # Actually: for fixed U, W, the equation for V[b, :] is:
                # T[:, b, :] = sum_t U[:, t] * W[:, t] * V[b, t]
                # T[:, b, :] (shape n*m x p*n) = sum_t V[b,t] * (U[:,t] ⊗ W[:,t])
                # Let M[:, t] = U[:, t] ⊗ W[:, t], shape (n*m * p*n, r)
                # T[:, b, :].reshape(-1) = M[:, :].T @ V[b, :]
                
                M = np.einsum('at,bt->abt', U_curr, W_curr).reshape(n*m * p*n, target_rank)
                T_unfold2 = T.reshape(n*m * p*n, m*p).T  # shape (m*p, n*m*p*n)
                try:
                    M_pinv = np.linalg.pinv(M.T)
                    V_curr = T_unfold2 @ M_pinv
                except:
                    break
                
                # Update W: similar
                # T[a, b, c] = sum_t U[a,t] * V[b,t] * W[c,t]
                # For fixed U, V: T[a, b, c] = sum_t W[c,t] * (U[a,t] * V[b,t])
                # Let N[:, t] = U[:, t] ⊗ V[:, t], shape (n*m * m*p, r)
                # T.reshape(n*m * m*p, p*n) = N[:, :].T @ W
                N = np.einsum('at,bt->abt', U_curr, V_curr).reshape(n*m * m*p, target_rank)
                T_unfold3 = T.reshape(n*m * m*p, p*n)
                try:
                    N_pinv = np.linalg.pinv(N.T)
                    W_curr = T_unfold3 @ N_pinv
                except:
                    break
            
            # Check if we found an exact decomposition
            if verify(U_curr, V_curr, W_curr):
                best_rank = target_rank
                best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
                break  # This rank works, try lower
        
        if best_rank <= target_rank:
            break  # Found a valid decomposition at this rank
    
    # Convert to lists with rounded values for integer/half-integer requirement
    def to_list(arr):
        # Round to nearest 0.5 to satisfy integer/half-integer constraint
        arr_rounded = np.round(arr * 2) / 2
        return arr_rounded.tolist()
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
