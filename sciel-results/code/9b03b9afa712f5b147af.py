import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Schoolbook rank = n*m*p = 45
    rank_school = n * m * p
    
    def schoolbook():
        U = np.zeros((n * m, rank_school))
        V = np.zeros((m * p, rank_school))
        W = np.zeros((p * n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    def to_list(mat):
        return [[int(x) if x == int(x) else float(x) for x in row] for row in mat]
    
    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        rank = U.shape[1]
        # The tensor T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
        # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        # But this isn't quite right. Let me think about the convention.
        
        # The multiplication is: C[i,k] = sum_j A[i,j] * B[j,k]
        # U has rows indexed by (i,j) pairs, V by (j,k) pairs, W by (k,i) pairs
        # The decomposition should satisfy:
        # For each output (k,i): C[i,k] = sum_j A[i,j]*B[j,k]
        # = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Actually, the standard tensor decomposition for matrix mult is:
        # T_{i,j,k} = sum_r U_{i,j}^r * V_{j,k}^r * W_{i,k}^r
        # where T_{i,j,k} = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
        
        # Let me check: for each (i,j,k), we need:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # Wait, that's not right either. The tensor for matrix multiplication
        # is defined such that:
        # C[i,k] = sum_{j,r} A[i,j] * B[j,k] * (some coefficient)
        
        # Actually, the correct formulation is:
        # The bilinear map is: (A,B) -> C where C[i,k] = sum_j A[i,j]*B[j,k]
        # This can be written as: C = sum_r (U_r^T A) * (V_r^T B) * W_r^T
        # where U_r, V_r are vectors and W_r gives the output coefficients
        
        # More precisely, if we define:
        # For each rank term r:
        #   term_r = (A * U_r) * (B * V_r) * W_r^T
        # where U_r is n*m x 1, V_r is m*p x 1, W_r is p*n x 1
        
        # Then C = sum_r term_r
        
        # Let me verify with a random test:
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        # Compute using the decomposition
        C_computed = np.zeros((n, p))
        for r in range(rank):
            # U[:, r] is n*m x 1, reshape to n x m
            U_r = U[:, r].reshape(n, m)
            V_r = V[:, r].reshape(m, p)
            W_r = W[:, r].reshape(p, n)
            
            # The term is: sum_{j,k} A[i,j] * V_r[j,k] * W_r[k,i] * ... 
            # Actually, I think the convention is different.
            
            # Let me try: C[i,k] += sum_j A[i,j] * V_r[j*p+k] * W_r[k*n+i]
            # But that doesn't use U_r.
        
        # I think I'm overcomplicating this. Let me look at the schoolbook solution:
        # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
        # So for a given (i,j,k), only one term contributes: the one where col corresponds to (i,j,k)
        # And the contribution is 1*1*1 = 1
        
        # So the decomposition is: C[i,k] = sum_{j,r} A[i,j] * B[j,k] * delta_{r,(i,j,k)}
        # where delta is 1 if r corresponds to (i,j,k)
        
        # This means: for each r, we have:
        # U_r has a 1 at position i*m+j, V_r has a 1 at position j*p+k, W_r has a 1 at position k*n+i
        # where (i,j,k) is determined by r
        
        # So the bilinear form is:
        # C[i,k] = sum_r (sum_j A[i,j] * U_r[i*m+j]) * (sum_k B[j,k] * V_r[j*p+k]) * W_r[k*n+i]
        # Wait, that's not right because j and k are summed separately.
        
        # Let me just test empirically:
        C_computed = np.zeros((n, p))
        for r in range(rank):
            U_r = U[:, r]  # n*m x 1
            V_r = V[:, r]  # m*p x 1
            W_r = W[:, r]  # p*n x 1
            
            # Reshape
            U_mat = U_r.reshape(n, m)
            V_mat = V_r.reshape(m, p)
            W_mat = W_r.reshape(p, n)
            
            # The term should be: sum_{j,k} A[i,j] * V_mat[j,k] * W_mat[k,i] * ... 
            # But we also need U. 
            # I think the correct formula is:
            # C[i,k] += (A @ U_mat[:, k]) * (V_mat @ B[:, i]) ... no
            
            # Let me try a different approach. The tensor T is defined by:
            # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
            # Otherwise T[a,b,c] = 0
            
            # The decomposition is: T = sum_r U_r * V_r * W_r (outer product)
            # So T[a,b,c] = sum_r U_r[a] * V_r[b] * W_r[c]
            
            # For the schoolbook: T[i*m+j, j*p+k, k*n+i] = 1
            # And the decomposition has one term for each (i,j,k)
            
            # So the verification is: for all a,b,c, sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
            
            pass
        
        # Let me do the direct tensor verification:
        # Build T
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        
        # Compute sum_r U[:,r] * V[:,r] * W[:,r]
        T_computed = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T_computed += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        return np.allclose(T, T_computed, atol=tol)
    
    # Try to find lower rank decompositions
    best_U, best_V, best_W = schoolbook()
    best_rank = rank_school
    
    # Strategy: Use the fact that for 3x3x5, we can try to find a rank < 45 decomposition
    # by exploiting the structure. The key insight is that the tensor has a special form.
    
    # For now, let's try a simple approach: random initialization + ALS
    def als_decompose(target_T, rank, max_iter=100, tol=1e-8, seed=42):
        """ALS decomposition for tensor T."""
        rng = np.random.RandomState(seed)
        n_dim = target_T.shape
        
        # Initialize
        U = rng.randn(n_dim[0], rank)
        V = rng.randn(n_dim[1], rank)
        W = rng.randn(n_dim[2], rank)
        
        prev_err = np.inf
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Update U
            for a in range(n_dim[0]):
                # T[a,:,:] = sum_r U[a,r] * V[:,r] * W[:,r] (broadcast)
                # For fixed a, we have: T[a,:,:] = sum_r U[a,r] * (V[:,r] * W[:,r])
                # This is a linear system in U[a,:]
                M = np.einsum('br,cr->bc', V, W)  # m*p x p*n
                # We want: U[a,:] @ M ≈ T[a,:,:].ravel()
                # So: M.T @ U[a,:].T ≈ T[a,:,:].ravel()
                rhs = target_T[a, :, :].ravel()
                M_flat = M.ravel()
                # This is underdetermined if rank > something, so use least squares
                # Actually, we can write: sum_r U[a,r] * M[r,:] = T[a,:,:].ravel()
                # where M is rank x (m*p*p*n)
                M_full = np.einsum('br,cr->bcr', V, W).reshape(rank, -1)
                U[a, :] = np.linalg.lstsq(M_full.T, rhs, rcond=None)[0]
            
            # Update V
            for b in range(n_dim[1]):
                M_full = np.einsum('ar,cr->acr', U, W).reshape(rank, -1)
                rhs = target_T[:, b, :].ravel()
                V[b, :] = np.linalg.lstsq(M_full.T, rhs, rcond=None)[0]
            
            # Update W
            for c in range(n_dim[2]):
                M_full = np.einsum('ar,br->abr', U, V).reshape(rank, -1)
                rhs = target_T[:, :, c].ravel()
                W[c, :] = np.linalg.lstsq(M_full.T, rhs, rcond=None)[0]
            
            # Compute error
            T_computed = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T_computed - target_T)
            
            if abs(prev_err - err) < tol:
                break
            prev_err = err
        
        return U, V, W, err
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try ranks from 40 down to 21
    for target_rank in range(40, 20, -1):
        if time.time() > deadline - 10:
            break
        
        # Try a few seeds
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            
            U, V, W, err = als_decompose(T, target_rank, max_iter=50, tol=1e-6, seed=seed)
            
            if err < 1e-6:
                # Verify exactly
                if verify(U, V, W, tol=1e-9):
                    best_rank = target_rank
                    best_U, best_V, best_W = U, V, W
                    break
        
        if best_rank < rank_school:
            break
    
    # If we found a better rank, use it; otherwise use schoolbook
    if best_rank < rank_school:
        return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
    else:
        U_school, V_school, W_school = schoolbook()
        return {"U": to_list(U_school), "V": to_list(V_school), "W": to_list(W_school)}
