import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin of 5 seconds
    
    n, m, p = 3, 4, 8
    
    # Create the matrix multiplication tensor T
    # T is a tensor such that T[i, j, k] is the coefficient of A[i,j] * B[j,k] in C[i,k]
    # In the standard tensor decomposition: T = sum_r U_r \otimes V_r \otimes W_r
    # Where U_r[i,j] = u_{ij,r}, V_r[j,k] = v_{jk,r}, W_r[i,k] = w_{ik,r}
    # Actually, the indices for the third mode in the task are (k, i) for W[k*n+i, r].
    # Let's align the tensor definition carefully.
    
    # The task requires:
    # U shape: (n*m, rank) -> indices (i*m+j, r)
    # V shape: (m*p, rank) -> indices (j*p+k, r)
    # W shape: (p*n, rank) -> indices (k*n+i, r)
    
    # The multiplication is C[i,k] = sum_j A[i,j] B[j,k]
    # We want sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] to represent the identity tensor?
    # No, the decomposition of the *multiplication tensor* T means:
    # T_{i, j, k, i'} ... wait, the tensor is T_{i, j, k} such that 
    # <T, A \otimes B> = Tr(A B)? No, C = A B.
    # The tensor T defines the bilinear map.
    # T_{(i,j), (j,k), (k,i)} = delta_{i,i'} delta_{k,k'} ? No.
    # Let's use the standard definition:
    # The multiplication tensor T has entries T[i, j, k] = 1 if we are computing C[i,k] from A[i,j] and B[j,k].
    # But the dimensions of U, V, W are coupled to the indices.
    # U[r] is a vector in R^{n*m}. V[r] is a vector in R^{m*p}. W[r] is a vector in R^{p*n}.
    # The condition for exact decomposition is:
    # For all i, j, k, the term A[i,j]B[j,k] contributes to C[i,k] with coefficient 1.
    # So, sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 if the indices match the multiplication pattern?
    # Actually, the tensor T is of size (n*m) x (m*p) x (p*n).
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) such that this triple corresponds to a valid multiplication step.
    # Specifically, the only non-zero entries in the "identity" of this tensor space are those where 
    # the j index in a matches the j index in b, and the k index in b matches the k index in c, and 
    # the i index in a matches the i index in c.
    # Let's construct T as a 3D array of shape (n*m, m*p, p*n).
    
    T = np.zeros((n*m, m*p, p*n))
    
    # Indices:
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
                
    # Now we search for rank R such that T = sum_{r=0}^{R-1} u_r \otimes v_r \otimes w_r
    # We use ALS with a smooth penalty to encourage sparsity, which helps in finding exact low-rank decompositions.
    
    # Initial guess: Schoolbook is rank n*m*p = 96.
    # We try to reduce rank. Known bounds for 3x4x8?
    # Lower bound: max(n*m, m*p, p*n) = max(12, 32, 24) = 32.
    # Also lower bound by matrix multiplication complexity.
    # Let's try ranks starting from 32 upwards.
    
    best_rank = n * m * p
    best_U = np.zeros((n*m, best_rank))
    best_V = np.zeros((m*p, best_rank))
    best_W = np.zeros((p*n, best_rank))
    
    # Initialize with schoolbook to have a valid starting point
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                best_U[a, col] = 1.0
                best_V[b, col] = 1.0
                best_W[c, col] = 1.0
                col += 1
                
    # Search for lower ranks
    for R in range(32, best_rank + 1):
        if time.time() > deadline - 10.0:
            break
            
        # Try multiple random restarts for each R
        success = False
        for restart in range(5):
            if time.time() > deadline - 5.0:
                break
                
            # Random initialization
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            # ALS Iteration
            for it in range(100):
                if time.time() > deadline - 5.0:
                    break
                    
                # Update U: For each r, U[:, r] = argmin || T[:, b, c] V[b, r] W[c, r] - U[:, r] ||^2 + penalty(U[:, r])
                # Since T is sparse (only 96 non-zeros), we can compute the target efficiently.
                # Target for U[:, r] is: sum_{b,c} T[:, b, c] * (V[b, r] * W[c, r])
                # Let's compute the weighted sum of T slices.
                
                # Construct the target tensor slice for U
                # U_target[:, r] = sum_{b,c} T[:, b, c] * V[b, r] * W[c, r]
                
                # This is a tensor contraction.
                # T is (n*m, m*p, p*n).
                # We need T_contracted[:, r] = np.einsum('abc,bc->a', T, V[:,r,None]*W[:,r,None])? No.
                # V[b, r] is a scalar for fixed r. W[c, r] is a scalar for fixed r.
                # So for each r, we compute:
                # u_target = sum_{b,c} T[:, b, c] * V[b, r] * W[c, r]
                
                # To speed this up, we can precompute the non-zero indices.
                # But N=12, M=32, P=24 is small. 12*32*24 = 9216 elements.
                # einsum is fine.
                
                V_r = V[:, r]
                W_r = W[:, r]
                
                # Compute target for U column r
                # T shape: (n*m, m*p, p*n)
                # We want result shape (n*m)
                # u_target = np.tensordot(T, V_r * W_r, axes=([1, 2], [0, 0])) -> No, indices don't match directly for tensordot like this.
                
                # Let's use einsum: 'abc,b,c->a'
                # T is (a, b, c). V_r is (b). W_r is (c).
                # u_target = np.einsum('abc,b,c->a', T, V_r, W_r)
                
                # Similarly for V: v_target = np.einsum('abc,a,c->b', T, U[:, r], W_r)
                # And for W: w_target = np.einsum('abc,a,b->c', T, U[:, r], V_r)
                
                # Update U[:, r] to minimize ||u_target - U[:, r]||^2 + lambda * ||U[:, r]||_1 (smoothed)
                # Using Soft-thresholding or Huber loss derivative?
                # For simplicity and robustness, let's use a simple projection or gradient step if exact solution is not found.
                # However, ALS typically sets U[:, r] = u_target if no penalty.
                # With penalty, we solve the convex problem.
                # Let's start with exact fit (no penalty) to see if rank R is sufficient.
                # If it converges to low error, we check for sparsity/exactness.
                
                U[:, r] = u_target
                V[:, r] = np.einsum('abc,a,c->b', T, U[:, r], W_r)
                W[:, r] = np.einsum('abc,a,b->c', T, U[:, r], V_r)
                
            # Check error after ALS
            # Compute tensor reconstruction error
            # Err = || T - sum_r U_r \otimes V_r \otimes W_r ||
            # We can compute the Frobenius norm of the difference.
            # Reconstructed T: Rec[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # Rec = np.einsum('ar,br,cr->abc', U, V, W)
            # This might be expensive if R is large, but R <= 96.
            
            if time.time() > deadline - 2.0:
                continue
                
            Rec = np.einsum('ar,br,cr->abc', U, V, W)
            error = np.linalg.norm(T - Rec)
            
            # If error is very small, we have a valid decomposition.
            # We then attempt to clean up the values to integers/half-integers.
            if error < 1e-6:
                # Round to nearest 0.5
                U_clean = np.round(U * 2) / 2
                V_clean = np.round(V * 2) / 2
                W_clean = np.round(W * 2) / 2
                
                # Verify cleaned version
                Rec_clean = np.einsum('ar,br,cr->abc', U_clean, V_clean, W_clean)
                error_clean = np.linalg.norm(T - Rec_clean)
                
                if error_clean < 1e-6:
                    best_rank = R
                    best_U = U_clean
                    best_V = V_clean
                    best_W = W_clean
                    success = True
                    break
        
        if success:
            break
            
    # Return the best found
    # Convert to lists as per interface
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
