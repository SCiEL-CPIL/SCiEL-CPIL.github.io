import numpy as np
import time
import random
import sys

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Schoolbook decomposition as fallback
    def schoolbook():
        r = n * m * p
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    U_s, V_s, W_s = schoolbook()
    best_rank = U_s.shape[1]
    best_U, best_V, best_W = U_s, V_s, W_s

    # Deadline guard: 140s to leave 10s for overhead/serialization
    deadline = time.time() + 140

    def verify(U, V, W, tol=1e-10):
        """Check if the decomposition is exact."""
        S = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(S, T, atol=tol)

    def round_and_verify(U, V, W):
        """
        Try to round U, V, W to integers or half-integers and verify.
        Returns (U_int, V_int, W_int) if successful, else None.
        """
        # Try integers first
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        if verify(U_int, V_int, W_int):
            return U_int, V_int, W_int
        
        # Try half-integers
        U_half = np.round(U * 2) / 2.0
        V_half = np.round(V * 2) / 2.0
        W_half = np.round(W * 2) / 2.0
        if verify(U_half, V_half, W_half):
            return U_half, V_half, W_half
            
        # Try quarters (0.25) as a broader net
        U_q = np.round(U * 4) / 4.0
        V_q = np.round(V * 4) / 4.0
        W_q = np.round(W * 4) / 4.0
        if verify(U_q, V_q, W_q):
            return U_q, V_q, W_q

        return None

    def laser_method(R):
        """
        Laser Method: Fix the input subspace (U matrix) and solve for V, W
        using linear algebra. This is more efficient than generic optimization.
        
        For each rank-1 term r: T = sum_r U[:,r] outer V[:,r] outer W[:,r]
        
        If we fix U, then for each output index (a,c), we need:
        T[a,:,c] = sum_r U[a,r] * V[:,r] * W[c,r]
        
        This is a system of bilinear equations. We can use the following approach:
        1. Randomly initialize U with R columns
        2. For each column r of U, the corresponding V[:,r] and W[:,r] must satisfy
           the tensor equation
        3. Use iterative refinement or direct linear algebra
        """
        if time.time() > deadline:
            return None
        
        # Strategy: Use the structure of the tensor
        # T[i*m+j, j*p+k, k*n+i] = 1
        
        # For a rank-1 decomposition: T = sum_r u_r outer v_r outer w_r
        # where u_r is in R^(nm), v_r in R^(mp), w_r in R^(pn)
        
        # Key insight: The tensor has a specific sparsity pattern.
        # T[a,b,c] = 1 if and only if:
        #   a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
        #   i.e., j = a % m, i = a // m, k = b % p, and c = k*n + i
        
        # Let's try a structured approach:
        # Group the rank-1 terms by their "support pattern"
        
        # Alternative: Use the fact that we can permute the decomposition
        # to get a canonical form
        
        # Let's try a direct linear algebra approach:
        # Fix U to have a specific structure, then solve for V and W
        
        # Initialize U with random values
        U_init = np.random.randn(nm, R)
        
        # For each rank-1 term, we need to find v_r and w_r such that:
        # sum_r U[a,r] * v_r[b] * w_r[c] = T[a,b,c]
        
        # This is equivalent to: For each a, the matrix sum_r U[a,r] * v_r * w_r^T = T[a,:,:]
        
        # Let's use an iterative approach:
        # 1. Initialize V and W randomly
        # 2. For each r, update v_r and w_r to best fit the residual
        
        V = np.random.randn(mp, R) * 0.1
        W = np.random.randn(pn, R) * 0.1
        
        for iteration in range(50):
            if time.time() > deadline:
                return None
            
            # Compute current tensor estimate
            S = np.einsum('ar,br,cr->abc', U_init, V, W)
            
            # Check convergence
            if np.allclose(S, T, atol=1e-8):
                break
            
            # Update V: For each b, we need sum_r V[b,r] * W[:,r] * U[:,r] = T[:,:,b]
            # This is a linear system in V[:,r] for fixed W
            # Actually, let's update one factor at a time
            
            # Update V: For each b, T[:,:,b] = sum_r V[b,r] * (U[:,r] * W[:,r])
            # So for each b, we need to solve: sum_r V[b,r] * (U[:,r] * W[:,r]) = T[:,:,b]
            # This is a linear system: M_b * V[:,b] = T[:,:,b] where M_b[a,c] = U[a,r]*W[c,r] summed over r
            # Wait, that's not quite right. Let me think again.
            
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            
            # For fixed U and W, V[b,r] appears in T[a,b,c] with coefficient U[a,r]*W[c,r]
            # So for each b, we have a linear system in V[:,b]:
            # sum_r V[b,r] * (U[:,r] * W[:,r]) = T[:,b,:]
            
            # Let's form the matrix for each b:
            # A_b is a matrix where A_b[(a,c), r] = U[a,r] * W[c,r]
            # Then A_b * V[:,b] = T[:,b,:].flatten()
            
            for b in range(mp):
                # Form A_b: shape (nm*pn, R)
                A_b = np.zeros((nm * pn, R))
                for a in range(nm):
                    for c in range(pn):
                        A_b[a*pn+c, :] = U_init[a, :] * W[c, :]
                
                rhs = T[:, b, :].flatten()
                
                # Solve least squares
                try:
                    V[:, b], residuals, rank, sv = np.linalg.lstsq(A_b, rhs, rcond=None)
                except:
                    pass
            
            # Update W similarly
            for c in range(pn):
                A_c = np.zeros((nm * mp, R))
                for a in range(nm):
                    for b in range(mp):
                        A_c[a*mp+b, :] = U_init[a, :] * V[b, :]
                
                rhs = T[:, :, c].flatten()
                
                try:
                    W[:, c], residuals, rank, sv = np.linalg.lstsq(A_c, rhs, rcond=None)
                except:
                    pass
            
            # Update U
            for a in range(nm):
                A_a = np.zeros((mp * pn, R))
                for b in range(mp):
                    for c in range(pn):
                        A_a[b*pn+c, :] = V[b, :] * W[c, :]
                
                rhs = T[a, :, :].flatten()
                
                try:
                    U_init[:, a], residuals, rank, sv = np.linalg.lstsq(A_a, rhs, rcond=None)
                except:
                    pass
        
        # Verify the result
        S = np.einsum('ar,br,cr->abc', U_init, V, W)
        if np.allclose(S, T, atol=1e-8):
            result = round_and_verify(U_init, V, W)
            if result:
                return result
        
        return None

    def try_rank(R, max_attempts=3):
        """
        Attempt to find a decomposition of rank R.
        Uses Laser Method and random initialization.
        """
        if R >= best_rank:
            return None
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Try Laser Method
            res = laser_method(R)
            if res is not None:
                return res
        
        return None

    # Try to reduce rank
    # Start from best_rank - 1 and go down
    # Limit the search to a reasonable range to fit in time
    # Schoolbook is 84. Try down to 70.
    for R in range(best_rank - 1, 69, -1):
        if time.time() > deadline:
            break
            
        # Try a few attempts for each rank
        res = try_rank(R, max_attempts=3)
        
        if res is not None:
            Ur, Vr, Wr = res
            best_rank = R
            best_U, best_V, best_W = Ur, Vr, Wr
            # We found a lower rank, can we go lower?
            # Yes, continue the loop.
            
    def to_list(M):
        L = []
        for row in M:
            L.append([float(x) for x in row])
        return L

    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
