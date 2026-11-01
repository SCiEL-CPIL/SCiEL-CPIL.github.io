"""Matrix multiplication 2,2,11 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill
    
    # Build the full multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # We want to find rank-r decomposition: T = sum_c U[:,c] * V[:,c] * W[:,c] (outer product)
    # i.e., T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # Strategy: 
    # 1. Try small ranks, check if a decomposition exists
    # 2. Use the structure: since n=2 is small, exploit it
    
    # Key insight: C = A*B where A is 2x2, B is 2x11, C is 2x11
    # C[0,:] = A[0,0]*B[0,:] + A[0,1]*B[1,:]
    # C[1,:] = A[1,0]*B[0,:] + A[1,1]*B[1,:]
    
    # This is: for each of the 11 columns, we do a 2x2 * 2x1 multiplication
    # The 2x2 * 2x1 multiplication has rank 3 (known result)
    # But the 11 columns share structure
    
    # Actually, let's think of it differently:
    # C = A * B, where A is 2x2, B is 2x11
    # We can write B as [B0; B1] where B0, B1 are 1x11 vectors
    # C0 = A[0,0]*B0 + A[0,1]*B1  (1x11)
    # C1 = A[1,0]*B0 + A[1,1]*B1  (1x11)
    
    # Each output row is a linear combination of B0 and B1 with coefficients from A
    # This is a bilinear map: (A[0,0], A[0,1], A[1,0], A[1,1]) x (B0, B1) -> (C0, C1)
    # where B0, B1 are 11-dimensional
    
    # The tensor is: T[a, b, c] where:
    # a indexes A (4 entries), b indexes B (22 entries), c indexes C (22 entries)
    # T = sum over j of A[i,j] * B[j,k] for C[i,k]
    
    # Let's use a direct search approach with SVD
    
    best_result = None
    best_rank = n * m * p  # schoolbook rank 44
    
    # Try to find decompositions of decreasing rank
    # Start from a reasonable upper bound
    
    # First, let's try a structured approach based on the small n=2
    
    # Approach: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # = (A[i,0] + A[i,1]) * ((B[0,k] + B[1,k])/2) + (A[i,0] - A[i,1]) * ((B[0,k] - B[1,k])/2)
    # This uses 2 "products" per (i,k) but the factors are shared across k
    
    # Let's define:
    # S_i = A[i,0] + A[i,1]
    # D_i = A[i,0] - A[i,1]  
    # E_k = (B[0,k] + B[1,k])/2
    # F_k = (B[0,k] - B[1,k])/2
    # Then C[i,k] = S_i * E_k + D_i * F_k
    
    # But S_i and D_i depend on i, and E_k and F_k depend on k
    # So we need: for each i, compute S_i and D_i (2 values)
    # For each k, compute E_k and F_k (2 values)
    # Then C[i,k] = S_i*E_k + D_i*F_k
    
    # The multiplications needed: S_i*E_k for all i,k (2*11=22) and D_i*F_k (2*11=22)
    # That's 44 multiplications - same as schoolbook!
    
    # We need to do better. Let's use the fact that the A-side has only 4 entries.
    
    # Alternative: Use a rank decomposition of the 4x22x22 tensor directly
    # with numerical methods, then round.
    
    # Let's try rank values from low to high
    # Lower bound: The tensor has "slice rank" considerations
    # For [2,2,11], lower bound is at least max(n*m, m*p, p*n) / something... 
    # Actually, a simple lower bound: the tensor restricted to any 2x2x2 sub-tensor has rank >= 3
    # (since R(2,2,2) >= 7 for Strassen, but for our specific structure...)
    
    # Let's just do a numerical search
    
    def try_rank(r, max_time=20.0):
        """Try to find a rank-r decomposition."""
        t0 = time.time()
        # Use a greedy/iterative approach
        # Initialize with random or structured guess
        np.random.seed(42)
        
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        # Iterative refinement: alternate least squares
        # Minimize ||T - sum_c U[:,c] o V[:,c] o W[:,c]||_F
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for iteration in range(50):
            if time.time() - t0 > max_time:
                break
            
            # Update U: for each column c, U[:,c] should approximate T * V[:,c] * W[:,c]
            # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
            # Fix V,W: T[:, :, :] ≈ sum_r U[:,r] o V[:,r] o W[:,r]
            # For updating U[:,r]: 
            #   T[a,b,c] = U[a,r]*V[b,r]*W[c,r] + sum_{r'!=r} U[a,r']*V[b,r']*W[c,r']
            #   => U[a,r] = (T[a,b,c] - sum_{r'!=r} U[a,r']*V[b,r']*W[c,r']) / (V[b,r]*W[c,r])
            #   In matrix form: reshape T to (n*m, m*p*p*n) and solve
            
            # More efficiently: 
            # T_reshaped = T.reshape(n*m, m*p, p*n)
            # For fixed V,W: T = sum_r U[:,r] o (V[:,r] * W[:,r])  (broadcast)
            # Let M_r = V[:,r] * W[:,r]  (shape m*p*p*n... no)
            # V[:,r] has shape (m*p,), W[:,r] has shape (p*n,)
            # V[:,r] o W[:,r] has shape (m*p, p*n)
            # T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # = sum_r U[a,r] * (V[:,r] o W[:,r])[b,c]
            # So T (as n*m x (m*p*p*n)) = U @ (V o W reshaped)
            # Let P = zeros(m*p*p*n, r) where P[b*p*n+c, r] = V[b,r]*W[c,r]
            # Then T.reshape(n*m, -1) = U @ P
            # Solve for U: U = T_2d @ P^T @ (P @ P^T)^{-1}
            
            T_2d = T.reshape(n * m, m * p * p * n)
            P = np.zeros((m * p * p * n, r))
            for b in range(m * p):
                for c in range(p * n):
                    for rr in range(r):
                        P[b * p * n + c, rr] = V[b, rr] * W[c, rr]
            
            try:
                PTP = P.T @ P + 1e-8 * np.eye(r)
                U_new = T_2d @ P.T @ np.linalg.inv(PTP)
            except:
                U_new = U
            
            # Update V similarly
            # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
            # Fix U,W: T = sum_r (U[:,r]*W[:,r]) o V[:,r]
            # T.reshape(m*p, n*m*p*n) = V @ Q where Q[b*n*m*p*n... 
            # T[b, a, c] = sum_r V[b,r] * U[a,r] * W[c,r]
            # Let Q[a*p*n+c, r] = U[a,r]*W[c,r]
            # T_reshaped2 = T.transpose(1,0,2).reshape(m*p, n*m*p*n)
            # T_reshaped2 = V @ Q
            T_2d_v = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
            Q = np.zeros((n * m * p * n, r))
            for a in range(n * m):
                for c in range(p * n):
                    for rr in range(r):
                        Q[a * p * n + c, rr] = U[a, rr] * W[c, rr]
            try:
                QTP = Q.T @ Q + 1e-8 * np.eye(r)
                V_new = T_2d_v @ Q.T @ np.linalg.inv(QTP)
            except:
                V_new = V
            
            # Update W
            T_2d_w = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)
            R_mat = np.zeros((n * m * m * p, r))
            for a in range(n * m):
                for b in range(m * p):
                    for rr in range(r):
                        R_mat[a * m * p + b, rr] = U[a, rr] * V[b, rr]
            try:
                RTP = R_mat.T @ R_mat + 1e-8 * np.eye(r)
                W_new = T_2d_w @ R_mat.T @ np.linalg.inv(RTP)
            except:
                W_new = W
            
            U, V, W = U_new, V_new, W_new
            
            # Compute error
            err = 0.0
            for rr in range(r):
                err += np.sum((T - U[:, rr][:, None, None] * V[:, rr][None, :, None] * W[:, rr][None, None, :])**2)
            err = np.sqrt(err)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-6:
                break
        
        return best_err, best_U, best_V, best_W
    
    # Try ranks from a reasonable lower bound upward
    # Lower bound: at least max(n*m, m*p, p*n) = max(4, 22, 22) = 22
    # But actually the tensor rank is at least the max of the slice ranks
    # For now, try from 22 upward
    
    for r in range(22, 45):
        if time.time() > deadline - 10:
            break
        err, U, V, W = try_rank(r, max_time=15.0)
        if err < 1e-4:
            # Found a decomposition! Now try to round to integers/half-integers
            # and verify
            best_rank = r
            best_result = (U, V, W)
            break
    
    if best_result is None:
        # Fall back to schoolbook
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
        return {"U": U, "V": V, "W": W}
    
    U, V, W = best_result
    r = best_rank
    
    # Try to round to integers or half-integers
    # Scale each column to make entries nice
    def round_to_half(x):
        return np.round(x * 2) / 2
    
    # Try rounding
    U_r = round_to_half(U)
    V_r = round_to_half(V)
    W_r = round_to_half(W)
    
    # Verify
    def verify(U, V, W, T, tol=1e-6):
        for c in range(U.shape[1]):
            T_check = T - U[:, c][:, None, None] * V[:, c][None, :, None] * W[:, c][None, None, :]
            # Check if remaining can be filled by other terms
            pass
        # Full check
        T_recon = np.zeros_like(T)
        for c in range(U.shape[1]):
            T_recon += U[:, c][:, None, None] * V[:, c][None, :, None] * W[:, c][None, None, :]
        return np.max(np.abs(T - T_recon)) < tol
    
    if verify(U_r, V_r, W_r, T):
        # Convert to lists
        U_list = U_r.tolist()
        V_list = V_r.tolist()
        W_list = W_r.tolist()
        # Clean up -0.0
        for mat in [U_list, V_list, W_list]:
            for i in range(len(mat)):
                for j in range(len(mat[i])):
                    if mat[i][j] == -0.0:
                        mat[i][j] = 0.0
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Try original (unrounded) if it's already close to half-integers
        if verify(U, V, W, T, tol=1e-3):
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            for mat in [U_list, V_list, W_list]:
                for i in range(len(mat)):
                    for j in range(len(mat[i])):
                        if abs(mat[i][j] - round(mat[i][j] * 2) / 2) < 1e-6:
                            mat[i][j] = round(mat[i][j] * 2) / 2
            return {"U": U_list, "V": V_list, "W": W_list}
        else:
            # Fall back to schoolbook
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
            return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
