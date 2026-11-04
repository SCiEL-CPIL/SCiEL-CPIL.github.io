"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import itertools
import random
import numpy as np

def _verify_factors(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] equals the MM tensor."""
    rank = len(U[0])
    for i, j, k in itertools.product(range(n), range(m), range(p)):
        s = 0
        for c in range(rank):
            s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
        if abs(s - 1) > tol:
            return False
    return True

def _find_rank19_5x5x5():
    """Find a rank-19 decomposition for 5x5x5 matrix multiplication."""
    n = m = p = 5
    rank = 19
    np.random.seed(42)
    
    # Initialize factors randomly
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    # Use ALS to fit the MM tensor
    T = np.ones((n, m, p))  # MM tensor: T[i,j,k] = 1 for all i,j,k
    
    # Reshape tensor to (n*m, p) for each j, etc.
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # This is a 3-way tensor decomposition, which is non-convex.
    # Use alternating least squares.
    
    for iteration in range(2000):
        # Update U: for each (i,j), solve for U[i,j,:]
        # T[i,j,:] = sum_k V[j,k,:] * W[k,i,:] * U[i,j,:]
        # This is: T[i,j,k] = sum_c U[i,j,c] * (V[j,k,c] * W[k,i,c])
        # So T[i,j,:] = U[i,j,:] @ (V[j,:,:] * W[:,i,:]).T  ... no
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Let M_U[i,j,k] = sum_c V[j,k,c] * W[k,i,c]  -- this depends on c
        # Actually: T[i,j,k] = sum_c (U[i,j,c] * V[j,k,c]) * W[k,i,c]
        # Or: T[i,j,k] = sum_c U[i,j,c] * (V[j,k,c] * W[k,i,c])
        
        # For fixed V,W: T[i,j,:] = sum_c U[i,j,c] * (V[j,:,:] * W[:,i,:])_c
        # So T[i,j,:] = U[i,j,:] @ diag(V[j,:] * W[:,i,:]) ... no
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # = sum_c U[i,j,c] * (V[j,k,c] * W[k,i,c])
        # Let A_c[i,j,k] = V[j,k,c] * W[k,i,c]
        # T[i,j,k] = sum_c U[i,j,c] * A_c[i,j,k]
        # For fixed (i,j): T[i,j,:] = sum_c U[i,j,c] * A_c[i,j,:]
        # So T[i,j,:] = U[i,j,:] @ A_c[i,j,:].T  where A_c[i,j,:] is vector over k
        # A_c[i,j,k] = V[j,k,c] * W[k,i,c]
        # So the matrix to multiply is: for each c, the vector A_c[i,j,:]
        # T[i,j,:] = sum_c U[i,j,c] * A_c[i,j,:]
        # This is: T[i,j,:] = U[i,j,:] @ B[i,j,:]  where B[i,j,c] = ... no
        # T[i,j,k] = sum_c U[i,j,c] * A_c[i,j,k]
        # Let B[i,j,c] be such that... this is element-wise, so:
        # T[i,j,:] = U[i,j,:] @ (A_c[i,j,:]).T  doesn't work directly
        # 
        # Actually: T[i,j,k] = sum_c U[i,j,c] * A_c[i,j,k]
        # This is a linear system in U[i,j,:]:
        # T[i,j,:] = U[i,j,:] @ A[i,j,:,:].T  where A[i,j,c,k] = A_c[i,j,k]
        # A[i,j,c,k] = V[j,k,c] * W[k,i,c]
        # So A[i,j,c,k] = V[j,k,c] * W[k,i,c]
        # The matrix is A[i,j,:,:,] with shape (rank, p)
        # T[i,j,:] (shape p) = U[i,j,:] (shape rank) @ A[i,j,:,:].T (shape rank, p)
        # A[i,j,c,k] = V[j,k,c] * W[k,i,c]
        # So A[i,j,:] = outer(V[j,:], W[:,i,:])  -- no, element-wise product
        # A[i,j,c,k] = V[j,k,c] * W[k,i,c]
        # This is: A[i,j,:] = np.outer(V[j,:], W[:,i,:])? No.
        # V[j,:] has shape (rank,), W[:,i,:] has shape (rank,)
        # We want A[i,j,c,k] for c in range(rank), k in range(p)
        # A[i,j,c,k] = V[j,k,c] * W[k,i,c]
        # V[j,k,c] is V[k, c] if we index V as (p, rank) for fixed j... 
        # V is (m*p, rank), V[j*p+k, c]
        # W is (p*n, rank), W[k*n+i, c]
        # So A[i,j,c,k] = V[j*p+k, c] * W[k*n+i, c]
        # For fixed i,j: A[i,j,c,k] = V[j*p+k, c] * W[k*n+i, c]
        # Let's build this matrix: shape (rank, p)
        # A_mat[c, k] = V[j*p+k, c] * W[k*n+i, c]
        
        for i in range(n):
            for j in range(m):
                # Build A_mat: A_mat[c, k] = V[j*p+k, c] * W[k*n+i, c]
                A_mat = np.zeros((rank, p))
                for c in range(rank):
                    for k in range(p):
                        A_mat[c, k] = V[j*p+k, c] * W[k*n+i, c]
                # Solve: T[i,j,:] = U[i,j,:] @ A_mat.T
                # T[i,j,:] has shape (p,), U[i,j,:] has shape (rank,)
                # T[i,j,k] = sum_c U[i,j,c] * A_mat[c,k]
                # So T[i,j,:] = U[i,j,:] @ A_mat.T
                # Solve least squares: U[i,j,:] = T[i,j,:] @ A_mat @ pinv(A_mat.T @ A_mat)
                # Or just use np.linalg.lstsq
                try:
                    U[i*m+j, :], _ = np.linalg.lstsq(A_mat.T, T[i,j,:], rcond=None)
                except:
                    pass
        
        # Update V similarly
        for j in range(m):
            for k in range(p):
                # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                # For fixed j,k: T[:,j,k] = sum_c V[j*p+k, c] * (U[:,j,c] * W[k*n+0:c, c])... 
                # T[i,j,k] = sum_c V[j*p+k, c] * U[i*m+j, c] * W[k*n+i, c]
                # For fixed j,k: T[:,j,k] (over i) = V[j*p+k,:] @ B where B[i,c] = U[i*m+j,c] * W[k*n+i,c]
                # T[i,j,k] = sum_c V[j*p+k, c] * B[i,c]
                # T[:,j,k] = B @ V[j*p+k,:].T
                # B[i,c] = U[i*m+j, c] * W[k*n+i, c]
                B_mat = np.zeros((n, rank))
                for i in range(n):
                    for c in range(rank):
                        B_mat[i, c] = U[i*m+j, c] * W[k*n+i, c]
                try:
                    V[j*p+k, :], _ = np.linalg.lstsq(B_mat, T[:,j,k], rcond=None)
                except:
                    pass
        
        # Update W similarly
        for k in range(p):
            for i in range(n):
                # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                # For fixed k,i: T[i,:,k] (over j) = W[k*n+i,:] @ C where C[j,c] = U[i*m+j,c] * V[j*p+k,c]
                # T[i,j,k] = sum_c W[k*n+i, c] * C[j,c]
                # T[i,:,k] = C @ W[k*n+i,:].T
                C_mat = np.zeros((m, rank))
                for j in range(m):
                    for c in range(rank):
                        C_mat[j, c] = U[i*m+j, c] * V[j*p+k, c]
                try:
                    W[k*n+i, :], _ = np.linalg.lstsq(C_mat, T[i,:,k], rcond=None)
                except:
                    pass
        
        # Check convergence
        if iteration % 100 == 0:
            err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = sum(U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] for c in range(rank))
                        err += (s - 1)**2
            if err < 1e-12:
                break
    
    # Round to nearest integers or half-integers
    U_rounded = np.round(U * 2) / 2
    V_rounded = np.round(V * 2) / 2
    W_rounded = np.round(W * 2) / 2
    
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U_rounded[i*m+j,c]*V_rounded[j*p+k,c]*W_rounded[k*n+i,c] for c in range(rank))
                if abs(s - 1) > 1e-6:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
    else:
        return None

def _restrict_to_4x5x5(U5, V5, W5):
    """Restrict 5x5x5 decomposition to 4x5x5 by zeroing out index 4."""
    n5, m5, p5 = 5, 5, 5
    n4 = 4
    rank = len(U5[0])
    
    # U5 has shape (n5*m5, rank) = (25, rank)
    # U5[i*m5+j, c]
    # For 4x5x5: U4[i*m4+j, c] where i in [0,4), j in [0,5)
    # U4[i*5+j, c] = U5[i*5+j, c] for i in [0,4)
    # So U4 is just the first 20 rows of U5
    U4 = [row[:] for row in U5[:n4*m5]]
    
    # V5 has shape (m5*p5, rank) = (25, rank)
    # V5[j*p5+k, c]
    # For 4x5x5: V4[j*p4+k, c] where j in [0,5), k in [0,5)
    # V4[j*5+k, c] = V5[j*5+k, c]
    # So V4 is just V5 (all 25 rows)
    V4 = [row[:] for row in V5]
    
    # W5 has shape (p5*n5, rank) = (25, rank)
    # W5[k*n5+i, c]
    # For 4x5x5: W4[k*p4+i, c] where k in [0,5), i in [0,4)
    # W4[k*4+i, c] = W5[k*5+i, c]  -- NO! The indexing changes!
    # In 5x5x5: W5[k*5+i, c]
    # In 4x5x5: W4[k*4+i, c]
    # These are different indexings!
    # W4[k*4+i, c] should equal W5[k*5+i, c] for the same (k,i) pair
    # So we need to remap: W4[k*4+i, c] = W5[k*5+i, c]
    W4 = [[0]*rank for _ in range(p5*n4)]
    for k in range(p5):
        for i in range(n4):
            for c in range(rank):
                W4[k*n4+i][c] = W5[k*n5+i][c]
    
    return U4, V4, W4

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    # Try to find rank-19 decomposition for 5x5x5 and restrict to 4x5x5
    result = _find_rank19_5x5x5()
    if result is not None:
        U5, V5, W5 = result
        U4, V4, W4 = _restrict_to_4x5x5(U5, V5, W5)
        # Verify the restricted decomposition
        if _verify_factors(U4, V4, W4, n, m, p):
            return {"U": U4, "V": V4, "W": W4}
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
