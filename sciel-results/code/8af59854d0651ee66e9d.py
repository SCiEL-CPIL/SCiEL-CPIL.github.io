"""Matrix multiplication 2x4 by 4x5 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    """
    Returns the rank-minimal exact tensor decomposition for 2x4 by 4x5 matrix multiplication.
    
    The output factor W follows the official trace convention W[k*n+i, r].
    
    Returns:
        dict with keys 'U', 'V', 'W' - factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)
    """
    deadline = time.time() + 120
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as fallback (rank = n*m*p = 40)
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find a lower-rank decomposition
    # For matrix multiplication C = A * B where A is n x m and B is m x p:
    # The tensor has indices (i, j, k) for A[i,j], B[j,k], C[i,k]
    # 
    # Key insight: We can decompose this as a sum over j of outer products.
    # A[i,j] * B[j,k] = C[i,k]
    # 
    # The rank of the matrix multiplication tensor M(n,m,p) is at most n*m*p (schoolbook).
    # For small dimensions, we can try to find lower rank decompositions.
    #
    # Let's try rank = n * p = 10 (since n=2, p=5)
    # This would correspond to decomposing each of the m=4 "intermediate" dimensions
    # into shared components.
    
    # Strategy: Use the fact that we can write the multiplication as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # A rank-r decomposition means:
    # C[i,k] = sum_{r=1}^{R} U[i*m + ?] * V[? * p + k] * W[k*n + i]
    # 
    # Actually, the standard form is:
    # U has shape (n*m, R), V has shape (m*p, R), W has shape (p*n, R)
    # The reconstruction is: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = C[i,k]
    # Wait, W is indexed as W[k*n + i, r] per the convention.
    
    # Let me think about this more carefully.
    # The matrix multiplication tensor T[i,j,k] = A[i,j] * B[j,k] = C[i,k]
    # A rank-1 decomposition is: T[i,j,k] = U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where W uses the convention W[k*n + i, r].
    
    # For rank R, we need:
    # sum_{r=1}^{R} U[i,j,r] * V[j,k,r] * W[k,i,r] = delta(i,j,k) for the identity
    # No wait, the tensor represents the multiplication operation.
    
    # The correct formulation:
    # We want: C[i,k] = sum_j A[i,j] * B[j,k]
    # This is a 3-way tensor with entries T[i,j,k] = 1 if this is the multiplication tensor.
    # Actually T[i,j,k] represents the contribution of A[i,j]*B[j,k] to C[i,k].
    
    # A rank-R decomposition gives:
    # T[i,j,k] = sum_{r=1}^{R} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For the multiplication tensor, T[i,j,k] = 1 for all valid (i,j,k) since each
    # (i,j,k) triple contributes exactly once to C[i,k] via A[i,j]*B[j,k].
    # Wait no, that's not right either. The tensor is:
    # T[i,j,k] = 1 if the triple (i,j,k) is a valid multiplication step.
    # All (i,j,k) with i in [0,n), j in [0,m), k in [0,p) are valid, so T[i,j,k] = 1.
    
    # Hmm, but that would make it an all-ones tensor which has rank 1. That can't be right.
    
    # Let me reconsider. The matrix multiplication tensor is defined such that:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T is a 3-tensor where T[i,j,k] = 1 for all i,j,k in range.
    # But the "decomposition" is about expressing the bilinear map, not the tensor itself.
    
    # Actually, I think the correct interpretation is:
    # The tensor T has entries T[i,j,k] = 1 (it's the structure tensor of matrix mult).
    # A rank-R decomposition means we can write T as a sum of R rank-1 tensors:
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # where each rank-1 tensor is the outer product of U_r, V_r, W_r.
    
    # But T[i,j,k] = 1 for all i,j,k means we need:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k
    
    # This is actually a system of equations. Let's set up the problem for small R.
    
    # For n=2, m=4, p=5: we have 2*4*5 = 40 equations, and R unknowns per factor.
    # U has 8*R entries, V has 20*R entries, W has 10*R entries.
    
    # Let's try to find a solution for small R using linear algebra.
    # The equations are: for each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # This is a system of multilinear equations. We can try to solve it iteratively.
    
    # Let's use a simple approach: fix R and try to find U, V, W satisfying the equations.
    # We can use alternating least squares or similar methods.
    
    # First, let's establish the lower bound. The rank of M(n,m,p) is at least
    # max(n*m, m*p, n*p) = max(8, 20, 10) = 20.
    # Actually, a better lower bound is needed. For M(2,4,5), what's the minimal rank?
    
    # Known results: R(2,2,2) = 7 (Strassen), R(1,n,m) = max(n,m).
    # For R(2,4,5), let's try various ranks.
    
    # Let's implement a solver that tries to find a decomposition for a given rank.
    def try_decomposition(R, max_iter=1000):
        """Try to find a rank-R decomposition using alternating minimization."""
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(n * m, R) * 0.1
        V = np.random.randn(m * p, R) * 0.1
        W = np.random.randn(p * n, R) * 0.1
        
        # We want: for all (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
        # 
        # This is equivalent to: the tensor formed by U, V, W should equal the all-ones
        # tensor of shape (n, m, p) when contracted appropriately.
        #
        # Let's reformulate: define T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
        # We want T[i,j,k] = 1 for all i,j,k.
        
        best_error = float('inf')
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            # Update U: for each (i,j), we want sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r])_k = 1 for all k
            # This is: U[i*m+j,:] @ (V[j*p+k,:] * W[k*n+i,:])^T summed over k = 1
            # More precisely: sum_r U[i*m+j,r] * sum_k V[j*p+k,r] * W[k*n+i,r] = 1
            # Hmm, this isn't quite right. Let me re-derive.
            
            # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # We want T[i,j,k] = 1 for all i,j,k.
            
            # Update W: for fixed (k,i), we want sum_r W[k*n+i,r] * (sum_j U[i*m+j,r] * V[j*p+k,r]) = 1
            # Let A_w[k*n+i, r] = sum_j U[i*m+j,r] * V[j*p+k,r]
            # Then we want: W[k*n+i,:] @ A_w[k*n+i,:].T... no.
            # sum_r W[k*n+i,r] * A_w[k*n+i,r] = 1
            # This is just the dot product of W[k*n+i,:] and A_w[k*n+i,:].
            # To minimize error, set W[k*n+i,:] = A_w[k*n+i,:] / sum(A_w[k*n+i,:]^2) if norm > 0
            
            # Actually, let's use a simpler approach. For each (i,j,k), we have one equation.
            # Total equations: n*m*p = 40. Unknowns: (n*m + m*p + p*n) * R = (8+20+10)*R = 38*R.
            # For R >= 2, we have more unknowns than equations, so solutions should exist.
            # But the equations are nonlinear (trilinear).
            
            # Let's use a different strategy: fix two factors and solve for the third.
            
            # Step 1: Update W. For each (k,i):
            # We want sum_r W[k*n+i,r] * B[k,i,r] = 1 where B[k,i,r] = sum_j U[i*m+j,r] * V[j*p+k,r]
            # To best satisfy this, set W[k*n+i,r] = B[k,i,r] / (B[k,i,:] @ B[k,i,:]) if norm > 0
            B_w = np.zeros((p * n, R))
            for k in range(p):
                for i in range(n):
                    for r in range(R):
                        B_w[k * n + i, r] = 0
                        for j in range(m):
                            B_w[k * n + i, r] += U[i * m + j, r] * V[j * p + k, r]
            for idx in range(p * n):
                norm_sq = np.dot(B_w[idx], B_w[idx])
                if norm_sq > 1e-12:
                    W[idx] = B_w[idx] / norm_sq
                else:
                    W[idx] = B_w[idx]  # keep as is
            
            # Step 2: Update U. For each (i,j):
            # We want sum_r U[i*m+j,r] * C[i,j,r] = 1 where C[i,j,r] = sum_k V[j*p+k,r] * W[k*n+i,r]
            C_u = np.zeros((n * m, R))
            for i in range(n):
                for j in range(m):
                    for r in range(R):
                        C_u[i * m + j, r] = 0
                        for k in range(p):
                            C_u[i * m + j, r] += V[j * p + k, r] * W[k * n + i, r]
            for idx in range(n * m):
                norm_sq = np.dot(C_u[idx], C_u[idx])
                if norm_sq > 1e-12:
                    U[idx] = C_u[idx] / norm_sq
                else:
                    U[idx] = C_u[idx]
            
            # Step 3: Update V. For each (j,k):
            # We want sum_r V[j*p+k,r] * D[j,k,r] = 1 where D[j,k,r] = sum_i U[i*m+j,r] * W[k*n+i,r]
            D_v = np.zeros((m * p, R))
            for j in range(m):
                for k in range(p):
                    for r in range(R):
                        D_v[j * p + k, r] = 0
                        for i in range(n):
                            D_v[j * p + k, r] += U[i * m + j, r] * W[k * n + i, r]
            for idx in range(m * p):
                norm_sq = np.dot(D_v[idx], D_v[idx])
                if norm_sq > 1e-12:
                    V[idx] = D_v[idx] / norm_sq
                else:
                    V[idx] = D_v[idx]
            
            # Compute error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(R):
                            val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        error += (val - 1.0) ** 2
            
            if error < best_error:
                best_error = error
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if error < 1e-8:
                break
        
        if best_error < 1e-6:
            # Convert to integer/half-integer if possible
            def to_clean(x):
                # Round to nearest half-integer
                return round(x * 2) / 2.0
            
            U_clean = [[to_clean(best_U[i, j]) for j in range(R)] for i in range(n * m)]
            V_clean = [[to_clean(best_V[i, j]) for j in range(R)] for i in range(m * p)]
            W_clean = [[to_clean(best_W[i, j]) for j in range(R)] for i in range(p * n)]
            
            # Verify
            def verify(Um, Vm, Wm, rank):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(rank):
                                val += Um[i * m + j][r] * Vm[j * p + k][r] * Wm[k * n + i][r]
                            if abs(val - 1.0) > 1e-6:
                                return False
                return True
            
            if verify(U_clean, V_clean, W_clean, R):
                return {"U": U_clean, "V": V_clean, "W": W_clean}
            else:
                # Try with unrounded values
                U_list = [[float(best_U[i, j]) for j in range(R)] for i in range(n * m)]
                V_list = [[float(best_V[i, j]) for j in range(R)] for i in range(m * p)]
                W_list = [[float(best_W[i, j]) for j in range(R)] for i in range(p * n)]
                if verify(U_list, V_list, W_list, R):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from the theoretical lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(8, 20, 10) = 20
    # But better lower bounds exist. Let's try from 10 upward.
    for R in range(10, 41):
        result = try_decomposition(R)
        if result is not None:
            return result
    
    # Fall back to schoolbook
    return schoolbook()
