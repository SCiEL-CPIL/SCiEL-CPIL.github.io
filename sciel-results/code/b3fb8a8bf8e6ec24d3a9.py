"""Matrix multiplication 2x5 by 5x5: search for rank-minimal exact decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 5]
# W[k*n+i, r] convention: output index is k*n + i where k is column of B, i is row of A
# A is (n x m) = (2 x 5), B is (m x p) = (5 x 5), C is (n x p) = (2 x 5)
# C[i,k] = sum_j A[i,j] * B[j,k]
# Tensor T[i,j,k] = delta (coefficient of A[i,j]*B[j,k] in C[i,k]) = 1
# Decomposition: C[i,k] = sum_r U[i*m+j... wait, let me re-read the interface.

# Interface: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
# U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
# C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * ... 
# Wait, the standard bilinear form is:
# C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
# But that's not right either. Let me think about this more carefully.

# Actually for matrix multiplication, the tensor decomposition is:
# C[i,k] = sum_j A[i,j] * B[j,k]
# In bilinear form: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
# This gives: C[i,k] = sum_r sum_j sum_j' U[i*m+j, r] * A[i,j] * V[j'*p+k, r] * B[j',k]
# = sum_j sum_j' (sum_r U[i*m+j, r] * V[j'*p+k, r]) * A[i,j] * B[j',k]
# For this to equal sum_j A[i,j]*B[j,k], we need:
# sum_r U[i*m+j, r] * V[j'*p+k, r] = delta_{j,j'} * W[k*n+i, r]... 

# Hmm, let me re-read. The W matrix has shape (p*n, rank) and the convention is W[k*n+i, r].
# I think the decomposition is:
# C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
# No wait, that would be trilinear. Let me think again.

# Standard: The multiplication tensor T has entries T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k].
# A rank-r decomposition means: T[i,j,k] = sum_{r=1}^R U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
# where U is (n*m x R), V is (m*p x R), W is (p*n x R).
# So: for all i,j,k: delta_{i,k}^{(j)} ... 

# Actually: C[i,k] = sum_j A[i,j] * B[j,k]
# The bilinear form: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
# Expanding: = sum_r sum_j sum_j' U[i*m+j,r] * V[j'*p+k,r] * A[i,j] * B[j',k]
# We need this to equal sum_j A[i,j] * B[j,k] for all A, B.
# So: sum_r U[i*m+j, r] * V[j'*p+k, r] = delta_{j,j'} for all i,k
# Wait, that's not quite right because W is also involved.

# Let me re-read the problem statement: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
# "W[k*n+i,r]" - this is the output factor.
# I think the correct interpretation is:
# C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
# This is a trilinear form, which is unusual for matrix multiplication.

# Actually, I think the standard formulation for the multiplication tensor is:
# The tensor T ∈ R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n} where T[(i,j),(j',k),(k',i')] = 1 if j=j' and k=k' and i=i'
# Wait no. Let me think of it as: T[a,b,c] where a indexes (i,j), b indexes (j',k), c indexes (k',i')
# T[a,b,c] = 1 if the product of the a-th input to first matrix, b-th input to second matrix, 
# contributes to the c-th output.
# Product A[i,j] * B[j',k] contributes to C[i,k] iff j=j' (and i,i and k,k are consistent).
# So T[(i,j), (j',k), (k',i')] = 1 if j=j' and k=k' and i=i'.

# Rank-r decomposition: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
# So: for all i,j,j',k,k',i': 
#   delta_{j,j'} * delta_{k,k'} * delta_{i,i'} = sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r]

# This is the correct formulation. The schoolbook solution has rank n*m*p = 50.
# For each (i,j,k), we set U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1.

# So the task is to find a decomposition of rank < 50.

def _build_tensor(n, m, p):
    """Build the 3-way tensor T of shape (n*m, m*p, p*n) where
    T[i*m+j, j'*p+k, k'*n+i'] = 1 if j==j' and k==k' and i==i' else 0."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_r U[i*m+j,r]*V[j'*p+k,r]*W[k'*n+i',r] = delta_{j,j'}*delta_{k,k'}*delta_{i,i'}"""
    n, m, p = int(n), int(m), int(p)
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R = U.shape[1]
    # For each r, the outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # Sum over r should give T
    T_recon = np.zeros((n*m, m*p, p*n))
    for r in range(R):
        T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    T_true = _build_tensor(n, m, p)
    max_err = np.max(np.abs(T_recon - T_true))
    return max_err < tol

def _try_rank_random(n, m, p, R, n_iter=200, seed=0):
    """Try to find rank-R decomposition via random initialization + iterative refinement."""
    rng = np.random.RandomState(seed)
    T = _build_tensor(n, m, p)
    # Flatten T into a matrix for SVD-based initialization
    # Mode-1 flattening: (n*m) x (m*p * p*n)
    T_flat1 = T.reshape(n*m, m*p * p*n)
    
    best_err = float('inf')
    best_U = best_V = best_W = None
    
    for it in range(n_iter):
        # Random initialization
        U = rng.randn(n*m, R)
        V = rng.randn(m*p, R)
        W = rng.randn(p*n, R)
        
        # Iterative refinement: fix two, solve for third
        for step in range(50):
            # Solve for U: for each (i,j), U[i*m+j, :] = argmin sum_{j',k,k',i'} (T[i*m+j,j'*p+k,k'*n+i'] - sum_r U[i*m+j,r]*V[j'*p+k,r]*W[k'*n+i',r])^2
            # This is: U[i*m+j, :] * (V^T * W_flat)^T = T[i*m+j, :, :].reshape(m*p*p*n)
            # Actually: sum_r U[a,r] * (sum_{b,c} V[b,r]*W[c,r] * T[a,b,c]) ... no.
            # Let me think. We want to minimize ||T - sum_r u_r ⊗ v_r ⊗ w_r||_F
            # Fix V, W. For each a, minimize ||T[a,:,:] - sum_r U[a,r] * (V[:,r] ⊗ W[:,r])||_F
            # = ||T[a].reshape(-1) - U[a,:] @ (V ⊗ W).reshape(m*p*p*n, R)||^2
            # So U[a,:] = T[a].reshape(-1) @ M / (M^T M) where M = (V⊗W).reshape(m*p*p*n, R)
            
            # Build M_U: for each r, the vector is V[:,r] ⊗ W[:,r] flattened
            # M_U has shape (m*p*p*n, R)
            # M_U[b*R + r] ... no, M_U[idx, r] where idx indexes (b,c)
            # V[:,r] ⊗ W[:,r] has shape (m*p, p*n), flatten to (m*p*p*n,)
            
            # This is expensive to build each iteration. Let's use a different approach.
            # Use the fact that we can solve via least squares on the full tensor.
            
            pass
        
        break  # Skip the complex iterative refinement for now, just return random
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s deadline, 10s safety margin
    
    # Start with schoolbook rank 50
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Try to find lower rank decompositions
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(10, 25, 10) = 25
    # But actual lower bound is higher. For 2x5 by 5x5, known results?
    # Let's try ranks from 25 upward and see if we can find a valid decomposition.
    
    T = _build_tensor(n, m, p)
    
    # Use SVD to get a lower bound and initial guess
    # Mode-1 matricization: (n*m) x (m*p * p*n)
    T_mat1 = T.reshape(n*m, m*p * p*n)
    # Mode-2 matricization: (m*p) x (n*m * p*n)
    T_mat2 = T.transpose(1, 0, 2).reshape(m*p, n*m * p*n)
    # Mode-3 matricization: (p*n) x (n*m * m*p)
    T_mat3 = T.transpose(2, 0, 1).reshape(p*n, n*m * m*p)
    
    # SVD of each matricization gives lower bounds on rank
    s1 = np.linalg.svd(T_mat1, compute_uv=False)
    s2 = np.linalg.svd(T_mat2, compute_uv=False)
    s3 = np.linalg.svd(T_mat3, compute_uv=False)
    
    # Numerical rank (count of singular values > tol)
    tol_svd = 1e-10
    r1 = np.sum(s1 > tol_svd)
    r2 = np.sum(s2 > tol_svd)
    r3 = np.sum(s3 > tol_svd)
    lower_bound = max(r1, r2, r3)
    
    # Try to find decomposition at each rank from lower_bound to best_rank-1
    for R in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations with iterative refinement
        found = False
        n_attempts = min(50, max(10, int((deadline - time.time()) / 2)))
        
        for attempt in range(n_attempts):
            if time.time() > deadline:
                break
            
            rng = np.random.RandomState(attempt * 12345 + 67890)
            
            # Initialize with SVD-based approach
            # Take top R singular vectors from mode-1 matricization
            if R <= T_mat1.shape[0] and R <= T_mat1.shape[1]:
                U_init, s_init, Vt_init = np.linalg.svd(T_mat1, full_matrices=False)
                U = U_init[:, :R]
                # Reshape Vt_init[:R] back
                Vt_R = Vt_init[:R, :].reshape(R, m*p, p*n)
                # For each r, we need V[:,r] and W[:,r]
                # Vt_R[r] is a (m*p, p*n) matrix = V[:,r] ⊗ W[:,r] (outer product)
                # We can get V[:,r] and W[:,r] from SVD of Vt_R[r]
                V = np.zeros((m*p, R))
                W = np.zeros((p*n, R))
                for r in range(R):
                    mat = Vt_R[r]
                    # SVD of mat: mat = U_mat @ diag(s_mat) @ V_mat.T
                    # mat = V[:,r] * W[:,r]^T (outer product)
                    # So rank 1: V[:,r] = U_mat[:,0] * s_mat[0], W[:,r] = V_mat[0,:]
                    # But it might not be exactly rank 1
                    try:
                        u_mat, s_mat, vt_mat = np.linalg.svd(mat, full_matrices=False)
                        if len(s_mat) > 0 and s_mat[0] > 1e-10:
                            V[:, r] = u_mat[:, 0] * np.sqrt(s_mat[0])
                            W[:, r] = vt_mat[0, :] * np.sqrt(s_mat[0])
                        else:
                            V[:, r] = rng.randn(m*p)
                            W[:, r] = rng.randn(p*n)
                    except:
                        V[:, r] = rng.randn(m*p)
                        W[:, r] = rng.randn(p*n)
            else:
                U = rng.randn(n*m, R)
                V = rng.randn(m*p, R)
                W = rng.randn(p*n, R)
            
            # Iterative refinement (ALS-style)
            for step in range(100):
                if time.time() > deadline:
                    break
                
                # Update U: for each a = i*m+j
                # Minimize ||T[a,:,:] - sum_r U[a,r] * (V[:,r] ⊗ W[:,r])||_F
                # Let M_U[r] = (V[:,r] ⊗ W[:,r]).flatten()
                # Then U[a,:] = argmin ||T[a].flatten() - U[a,:] @ M_U||^2
                # U[a,:] = T[a].flatten() @ M_U @ (M_U.T @ M_U)^{-1}
                
                # Build M_U: shape (m*p*p*n, R)
                M_U = np.zeros((m*p*p*n, R))
                for r in range(R):
                    M_U[:, r] = np.outer(V[:, r], W[:, r]).flatten()
                
                # Solve least squares: U = T_mat1 @ M_U @ (M_U.T @ M_U)^{-1}
                # T_mat1 has shape (n*m, m*p*p*n)
                try:
                    # U = T_mat1 @ M_U @ inv(M_U.T @ M_U)
                    M_UM = M_U.T @ M_U  # (R, R)
                    if np.linalg.cond(M_UM) < 1e10:
                        U = T_mat1 @ M_U @ np.linalg.inv(M_UM)
                    else:
                        U = T_mat1 @ M_U @ np.linalg.pinv(M_UM)
                except:
                    pass
                
                # Update V: for each b = j*p+k
                # Minimize ||T[:,b,:] - sum_r V[b,r] * (U[:,r] ⊗ W[:,r])||_F
                M_V = np.zeros((n*m*p*n, R))
                for r in range(R):
                    M_V[:, r] = np.outer(U[:, r], W[:, r]).flatten()
                
                T_mat2_curr = T.transpose(1, 0, 2).reshape(m*p, n*m * p*n)
                try:
                    M_VM = M_V.T @ M_V
                    if np.linalg.cond(M_VM) < 1e10:
                        V = T_mat2_curr @ M_V @ np.linalg.inv(M_VM)
                    else:
                        V = T_mat2_curr @ M_V @ np.linalg.pinv(M_VM)
                except:
                    pass
                
                # Update W: for each c = k*n+i
                # Minimize ||T[:,:,c] - sum_r W[c,r] * (U[:,r] ⊗ V[:,r])||_F
                M_W = np.zeros((n*m*m*p, R))
                for r in range(R):
                    M_W[:, r] = np.outer(U[:, r], V[:, r]).flatten()
                
                T_mat3_curr = T.transpose(2, 0, 1).reshape(p*n, n*m * m*p)
                try:
                    M_WM = M_W.T @ M_W
                    if np.linalg.cond(M_WM) < 1e10:
                        W = T_mat3_curr @ M_W @ np.linalg.inv(M_WM)
                    else:
                        W = T_mat3_curr @ M_W @ np.linalg.pinv(M_WM)
                except:
                    pass
                
                # Check convergence
                if step % 10 == 9:
                    T_recon = np.zeros((n*m, m*p, p*n))
                    for r in range(R):
                        T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                    err = np.max(np.abs(T_recon - T))
                    if err < 1e-8:
                        found = True
                        break
            
            if found:
                # Verify
                if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-7):
                    best_rank = R
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    break
        
        if found:
            break
    
    return best_result
