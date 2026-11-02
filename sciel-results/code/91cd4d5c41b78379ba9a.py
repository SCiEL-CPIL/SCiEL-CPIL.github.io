"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # Build the matrix multiplication tensor T[i,j,k] = delta for C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor has indices (i, j, k) with shape (n, m, p)
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, let me re-read the convention.
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The tensor entry T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Build tensor T of shape (n*m, m*p, p*n) where T[i*m+j, j*p+k, k*n+i] = 1
    # Actually the tensor is: for each (i,j,k), T[i,j,k] = 1 if we consider the standard matmul tensor
    # The decomposition requires: for all i,j,k: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Let's build the tensor as a 3D array T[i, j, k] of shape (n, m, p)
    # T[i,j,k] = 1 for all valid i,j,k (it's the identity-like tensor for matmul)
    # Actually no. The matmul tensor T has T[i,j,k] = 1 if we think of it as the structure tensor.
    # The condition is: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # And the decomposition must satisfy: for all i,j,k, T[i,j,k] = sum_c U[i*m+j, c]*V[j*p+k, c]*W[k*n+i, c]
    
    # Wait, that's not right either. Let me think again.
    # The matrix multiplication tensor M has entries M[i,j,k] where i indexes row of A (and C),
    # j indexes column of A / row of B, k indexes column of B (and C).
    # M[i,j,k] = 1 for all valid indices.
    # A rank-r decomposition gives: M[i,j,k] = sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # So we need: for all i in range(n), j in range(m), k in range(p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of n*m*p = 80 equations.
    
    # Strategy: Use ALS to find low-rank decompositions.
    # Start from schoolbook (rank 80) or Strassen (rank 64) and try to reduce.
    
    # First, let's build the Strassen decomposition for 4x4 * 4x5.
    # Split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]] where B11, B21 are 2x2, B12, B22 are 2x3
    # C = [[C11, C12], [C21, C22]] where C11, C21 are 2x2, C12, C22 are 2x3
    
    # Strassen for rectangular: we need to handle the fact that B has rectangular blocks.
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Each "multiplication" here is a 2x2 * 2x2 or 2x2 * 2x3 matrix product.
    # A 2x2 * 2x2 product has tensor rank at most 4 (schoolbook) or 7 (Strassen, but that's for the recursion).
    # Actually, for the base case, a 2x2*2x2 multiplication has rank 7 (Strassen) or 4 (schoolbook).
    # A 2x2*2x3 multiplication has rank at most 6 (schoolbook).
    
    # This is getting complex. Let me use a different approach.
    
    # Approach: Use ALS with random initialization, trying to find a decomposition of rank r
    # that exactly satisfies the tensor equations.
    
    def build_tensor():
        """Build the matmul tensor as a 3D array T[i,j,k] of shape (n,m,p), all ones."""
        T = np.ones((n, m, p))
        return T
    
    def verify_decomposition(U, V, W):
        """Check if U, V, W exactly decompose the tensor. Returns True if valid."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        # For each (i,j,k), check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def als_decompose(T, r, U0=None, V0=None, W0=None, max_iter=50, tol=1e-12):
        """
        ALS to find rank-r decomposition of tensor T.
        T has shape (n, m, p).
        U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r).
        Constraint: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = T[i,j,k]
        """
        if U0 is None:
            U = np.random.randn(n*m, r) * 0.1
        else:
            U = np.array(U0, dtype=float).copy()
            if U.shape[1] < r:
                U = np.hstack([U, np.random.randn(n*m, r - U.shape[1]) * 0.1])
            elif U.shape[1] > r:
                U = U[:, :r]
        
        if V0 is None:
            V = np.random.randn(m*p, r) * 0.1
        else:
            V = np.array(V0, dtype=float).copy()
            if V.shape[1] < r:
                V = np.hstack([V, np.random.randn(m*p, r - V.shape[1]) * 0.1])
            elif V.shape[1] > r:
                V = V[:, :r]
        
        if W0 is None:
            W = np.random.randn(p*n, r) * 0.1
        else:
            W = np.array(W0, dtype=float).copy()
            if W.shape[1] < r:
                W = np.hstack([W, np.random.randn(p*n, r - W.shape[1]) * 0.1])
            elif W.shape[1] > r:
                W = W[:, :r]
        
        T = np.array(T, dtype=float)
        
        for it in range(max_iter):
            # Update U: for each row i*m+j, we have equations for each k:
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
            # This is not linear in U alone because V and W also depend on c.
            # ALS approach: fix V and W, solve for U.
            # For fixed V, W: U[i*m+j, :] needs to satisfy for all k:
            #   sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = T[i,j,k]
            # Let A[i,j,k,c] = V[j*p+k, c] * W[k*n+i, c]
            # Then: sum_c U[i*m+j, c] * A[i,j,k,c] = T[i,j,k] for all k
            # This is a linear system for U[i*m+j, :]:
            #   For fixed (i,j): sum_c U[i*m+j, c] * A[i,j,k,c] = T[i,j,k] for k=0..p-1
            # This is p equations in r unknowns. If p >= r, we can solve.
            # Build matrix M[i,j] of shape (p, r): M[i,j][k, c] = V[j*p+k, c] * W[k*n+i, c]
            # Solve M[i,j] @ U[i*m+j, :] = T[i,j,:]
            
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, r))
                    for k in range(p):
                        for c in range(r):
                            M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    # Solve M @ u = T[i,j,:]
                    try:
                        u, residuals, rank_m, sv = np.linalg.lstsq(M, T[i, j, :], rcond=None)
                        U[i*m+j, :] = u
                    except:
                        pass
            
            # Update V: for each row j*p+k, we have equations for each i:
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
            # For fixed (j,k): sum_c V[j*p+k, c] * (U[i*m+j, c] * W[k*n+i, c]) = T[i,j,k] for i=0..n-1
            # Matrix N[j,k] of shape (n, r): N[j,k][i, c] = U[i*m+j, c] * W[k*n+i, c]
            # Solve N[j,k] @ V[j*p+k, :] = T[:, j, k]
            
            for j in range(m):
                for k in range(p):
                    N = np.zeros((n, r))
                    for i in range(n):
                        for c in range(r):
                            N[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    try:
                        v, residuals, rank_n, sv = np.linalg.lstsq(N, T[:, j, k], rcond=None)
                        V[j*p+k, :] = v
                    except:
                        pass
            
            # Update W: for each row k*n+i, we have equations for each j:
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
            # For fixed (k,i): sum_c W[k*n+i, c] * (U[i*m+j, c] * V[j*p+k, c]) = T[i,j,k] for j=0..m-1
            # Matrix P[k,i] of shape (m, r): P[k,i][j, c] = U[i*m+j, c] * V[j*p+k, c]
            # Solve P[k,i] @ W[k*n+i, :] = T[i, :, k]
            
            for k in range(p):
                for i in range(n):
                    P = np.zeros((m, r))
                    for j in range(m):
                        for c in range(r):
                            P[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    try:
                        w, residuals, rank_p, sv = np.linalg.lstsq(P, T[i, :, k], rcond=None)
                        W[k*n+i, :] = w
                    except:
                        pass
            
            # Check convergence
            if it % 5 == 0:
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(r):
                                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            err += (val - T[i,j,k])**2
                if err < tol:
                    break
        
        return U, V, W
    
    def round_to_half_inters(U, V, W):
        """Round entries to nearest half-integer."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        return U_r, V_r, W_r
    
    def to_lists(U, V, W):
        """Convert numpy arrays to lists of lists."""
        return (U.tolist(), V.tolist(), W.tolist())
    
    # Build tensor
    T = np.ones((n, m, p))
    
    # Try ranks from low to high
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # First, try to find a decomposition starting from rank 64 (Strassen-like) down
    # The theoretical minimum rank for 4x4*4x5 is unknown, but we know it's <= 64 (Strassen)
    # Let's try ranks from 40 to 80
    
    for r in range(40, min(81, n*m*p + 1)):
        if time.time() > deadline - 10:
            break
        
        # Multiple random restarts for each rank
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            
            U, V, W = als_decompose(T, r, max_iter=30, tol=1e-10)
            
            # Round to half-integers
            U_r, V_r, W_r = round_to_half_inters(U, V, W)
            
            # Verify
            if verify_decomposition(U_r, V_r, W_r):
                best_result = to_lists(U_r, V_r, W_r)
                best_rank = r
                break
        
        if best_result is not None:
            break
    
    if best_result is None:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        best_result = (U, V, W)
        best_rank = rank
    
    U_list, V_list, W_list = best_result
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
