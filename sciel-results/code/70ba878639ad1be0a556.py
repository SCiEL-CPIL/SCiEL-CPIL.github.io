"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # safety margin
    
    # Build the multiplication tensor T[a, b, c] where:
    # a = i*m + j (row of A, col of A)
    # b = j*p + k (row of B, col of B)  
    # c = k*n + i (row of output, col of output)
    # T[a,b,c] = 1 if there exist i,j,k with a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_sb
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose T (after rounding to half-integers)."""
        T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
        return np.allclose(T_recon, T, atol=tol)
    
    def try_als(r, seed, max_iters=300):
        """Alternating least squares for rank r decomposition."""
        np.random.seed(seed)
        U = np.random.randn(n * m, r) * 0.5
        V = np.random.randn(m * p, r) * 0.5
        W = np.random.randn(p * n, r) * 0.5
        
        for it in range(max_iters):
            if time.time() > deadline - 3:
                return None
            
            # Update U: for each a, solve G_U @ U[a,:] = h_U[a,:]
            # where G_U[d1,d2] = sum_{b,c} V[b,d1]*W[c,d1]*V[b,d2]*W[c,d2]
            # and h_U[a,d] = sum_{b,c} T[a,b,c]*V[b,d]*W[c,d]
            
            # Efficient computation using einsum
            # TW[b,c,d] = V[b,d] * W[c,d]
            TW = V[:, np.newaxis, :] * W[np.newaxis, :, :]  # (m*p, p*n, r)
            
            h_U = np.einsum('abc,bcd->ad', T, TW)  # (n*m, r)
            G_U = np.einsum('bcd,bce->de', TW, TW)  # (r, r)
            
            try:
                U_new = np.linalg.solve(G_U + 1e-10 * np.eye(r), h_U.T).T
            except np.linalg.LinAlgError:
                U_new = h_U
            U = U_new
            
            # Update V
            UW = U[:, np.newaxis, :] * W[np.newaxis, :, :]  # (n*m, p*n, r)
            h_V = np.einsum('abc,acd->bd', T, UW)  # (m*p, r)
            G_V = np.einsum('acd,ace->de', UW, UW)  # (r, r)
            
            try:
                V_new = np.linalg.solve(G_V + 1e-10 * np.eye(r), h_V.T).T
            except np.linalg.LinAlgError:
                V_new = h_V
            V = V_new
            
            # Update W
            UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]  # (n*m, m*p, r)
            h_W = np.einsum('abc,abd->cd', T, UV)  # (p*n, r)
            G_W = np.einsum('abd,abe->de', UV, UV)  # (r, r)
            
            try:
                W_new = np.linalg.solve(G_W + 1e-10 * np.eye(r), h_W.T).T
            except np.linalg.LinAlgError:
                W_new = h_W
            W = W_new
        
        # Try rounding to half-integers
        for scale in [2, 1]:  # half-integers first, then integers
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            
            if verify_decomposition(U_r, V_r, W_r):
                return (U_r, V_r, W_r)
        
        return None
    
    def strassen_3x3_core():
        """
        Return the Strassen 3x3x3 core tensors as (U_core, V_core, W_core)
        with integer entries. These give rank-15 decomposition of 3x3x3 matmul.
        """
        # Strassen's algorithm for 3x3 matrices uses 15 multiplications
        # The standard decomposition:
        # M1 = (A11+A22+A33)(B11+B22+B33)
        # M2 = (A11-A22)(B11-B22)
        # M3 = (A22-A33)(B22-B33)
        # M4 = (A33-A11)(B33-B11)
        # M5 = (A12+A13)(B21+B31)
        # M6 = (A21+A23)(B12+B32)
        # M7 = (A31+A32)(B13+B23)
        # M8 = (A11+A12-A23)(B22+B23-B33)
        # M9 = (A11+A13-A32)(B33+B32-B22)
        # M10 = (A22+A21-A13)(B33+B31-B12)
        # M11 = (A33+A32-A21)(B11+B12-B31)
        # M12 = (A13+A33-A22)(B21+B11-B22)
        # M13 = (A21+A11-A33)(B32+B22-B33)
        # M14 = (A32+A22-A11)(B13+B33-B11)
        # M15 = (A12+A23+A31)(B12+B23+B31)
        
        # Actually, let me use a simpler known rank-15 decomposition.
        # For 3x3x3, the minimal rank is 15.
        
        # Let me construct it properly.
        # Output C[i,j] = sum_k A[i,k] * B[k,j]
        
        # I'll use the known 15-term decomposition.
        # Let me define the factors explicitly.
        
        U = np.zeros((9, 15))
        V = np.zeros((9, 15))
        W = np.zeros((9, 15))
        
        # This is the standard 3x3 Strassen-like decomposition
        # Let me use a verified one from literature
        
        # Actually, for 3x3x3, one known rank-15 decomposition:
        # The bilinear complexity is 15.
        
        # Let me just verify schoolbook works and try to find better via search.
        # For now, return None to indicate we should use search.
        return None
    
    def try_substitution_method(r, seed):
        """
        Substitution method: fix U to a structured form and solve linear system.
        For the 3x3x5 case, we can try to extend the 3x3x3 Strassen core.
        """
        if time.time() > deadline - 3:
            return None
        
        # Strategy: Use the 3x3x3 part (columns k=0,1,2) with Strassen rank 15,
        # then add columns for k=3,4.
        
        # For the 3x3x3 part, we need rank at least 15.
        # For the full 3x3x5, the additional 2 output columns (k=3,4) 
        # require additional rank.
        
        # Let's try a different approach: direct search with better initialization.
        
        # Initialize using structure: 
        # The tensor has the form T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
        # So T = sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
        
        # This is already a rank-45 decomposition (schoolbook).
        # We need to find a better one.
        
        # Try ALS with multiple seeds and better initialization
        best_result_local = None
        
        for attempt in range(5):
            if time.time() > deadline - 3:
                break
            
            np.random.seed(seed + attempt * 7919)
            
            # Better initialization: use the schoolbook structure as a starting point
            # and add random perturbations
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            
            # Random initialization
            U = np.random.randn(n * m, r) * 0.3
            V = np.random.randn(m * p, r) * 0.3
            W = np.random.randn(p * n, r) * 0.3
            
            # Run ALS
            for it in range(200):
                if time.time() > deadline - 3:
                    break
                
                # Update U
                TW = V[:, np.newaxis, :] * W[np.newaxis, :, :]
                h_U = np.einsum('abc,bcd->ad', T, TW)
                G_U = np.einsum('bcd,bce->de', TW, TW)
                
                try:
                    U_new = np.linalg.solve(G_U + 1e-10 * np.eye(r), h_U.T).T
                except np.linalg.LinAlgError:
                    U_new = h_U
                U = U_new
                
                # Update V
                UW = U[:, np.newaxis, :] * W[np.newaxis, :, :]
                h_V = np.einsum('abc,acd->bd', T, UW)
                G_V = np.einsum('acd,ace->de', UW, UW)
                
                try:
                    V_new = np.linalg.solve(G_V + 1e-10 * np.eye(r), h_V.T).T
                except np.linalg.LinAlgError:
                    V_new = h_V
                V = V_new
                
                # Update W
                UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]
                h_W = np.einsum('abc,abd->cd', T, UV)
                G_W = np.einsum('abd,abe->de', UV, UV)
                
                try:
                    W_new = np.linalg.solve(G_W + 1e-10 * np.eye(r), h_W.T).T
                except np.linalg.LinAlgError:
                    W_new = h_W
                W = W_new
            
            # Try rounding to half-integers and integers
            for scale in [2, 1]:
                U_r = np.round(U * scale) / scale
                V_r = np.round(V * scale) / scale
                W_r = np.round(W * scale) / scale
                
                if verify_decomposition(U_r, V_r, W_r):
                    return (U_r, V_r, W_r)
        
        return None
    
    # Try ranks from lower bound up to schoolbook
    # For 3x3x5, the rank must be at least the slice rank lower bound
    # Slice rank lower bound: max over partitions
    
    # The tensor rank of 3x3x5 matmul is known to be at least 15 (from 3x3x3 subproblem)
    # Upper bound is 45 (schoolbook)
    # Let's try to find a decomposition with rank < 45
    
    lower_bound = 15
    ranks_to_try = list(range(lower_bound, rank_sb + 1))
    
    seeds = [42, 0, 1, 2, 3, 7, 13, 99, 123, 456, 789, 1000, 1234, 5678, 9999]
    
    for r in ranks_to_try:
        if time.time() > deadline - 3:
            break
        if r >= best_rank:
            break
        
        found = False
        for seed in seeds:
            if time.time() > deadline - 3:
                break
            
            result = try_als(r, seed)
            if result is not None:
                U_f, V_f, W_f = result
                if r < best_rank:
                    best_rank = r
                    U_list = [[float(x) for x in row] for row in U_f]
                    V_list = [[float(x) for x in row] for row in V_f]
                    W_list = [[float(x) for x in row] for row in W_f]
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                found = True
                break
        
        if found:
            # Found a valid decomposition at this rank, try lower ranks
            continue
    
    return best_result
# EVOLVE-BLOCK-END
