"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    _start = time.time()
    _deadline = _start + 140.0  # 140s deadline, 10s safety margin
    
    n, m, p = DIMENSIONS
    
    def schoolbook():
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
    
    def verify(U, V, W, tol=1e-9):
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        rank = U_arr.shape[1]
        
        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
                    if abs(s - 1.0) > max_err:
                        max_err = abs(s - 1.0)
        
        return max_err < tol
    
    def strassen_decomposition():
        """
        Partition the 4-dimensional inner product into two 2-dim halves.
        A (2x4) = [A0 | A1] where A0, A1 are 2x2
        B (4x5) = [B0; B1] where B0, B1 are 2x5
        C = A0*B0 + A1*B1
        
        For each 2x2 * 2x5 product, use a Strassen-like approach.
        
        For A0*B0 where A0=[[a,b],[c,d]], B0 is 2x5:
        Row 0 of product: [a*B0[0,0]+b*B0[1,0], ..., a*B0[0,4]+b*B0[1,4]]
        Row 1 of product: [c*B0[0,0]+d*B0[1,0], ..., c*B0[0,4]+d*B0[1,4]]
        
        Using Strassen on the 2x2 matrix part:
        We can write the 2x2 * 2x5 product using 7 "multiplications" where each
        multiplication is a scalar times a 1x5 vector (or 2x5).
        
        Actually, for 2x2 * 2xp, the standard Strassen doesn't directly apply
        because the second matrix is 2xp not 2x2. Let me think differently.
        
        For C = A0*B0 + A1*B1, we can use the following approach:
        Let M1 = (A0 + A1) * (B0 + B1)
        Then A0*B0 + A1*B1 = M1 - A0*B1 - A1*B0
        
        This doesn't help directly. Let me try a different partition.
        
        Actually, let me just use the direct approach:
        C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
        
        Split j into {0,1} and {2,3}:
        C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
        
        First sum: A[i,0]*B[0,k] + A[i,1]*B[1,k]
        Second sum: A[i,2]*B[2,k] + A[i,3]*B[3,k]
        
        For the first sum (2x2 * 2x5 sub-problem):
        Using Strassen-like:
        Let P1 = A[i,0] + A[i,1], P2 = A[i,0] - A[i,1]
        Q1 = B[0,k] + B[1,k], Q2 = B[0,k] - B[1,k]
        
        A[i,0]*B[0,k] + A[i,1]*B[1,k] = (P1*Q1 + P2*Q2) / 2
        
        But this still requires 4 products (P1*Q1, P2*Q2) per (i,k), same as before.
        
        Hmm, Strassen works for full 2x2 matrix multiplication where we get 7 instead of 8.
        For 2x2 * 2x5, each of the 2 rows of the result involves 2*5=10 scalar products
        in schoolbook (2 per row element, 5 elements per row, 2 rows).
        
        Actually wait: 2x2 * 2x5 gives a 2x5 result. Each element C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k].
        That's 2 multiplications per element, 10 elements total = 20 multiplications.
        
        With Strassen for the 2x2 part: we can compute the 2x2 matrix product M = A0 * B0' 
        where B0' is 2x2, using 7 multiplications. But B0 is 2x5, not 2x2.
        
        Let me try a completely different approach. Just verify if rank 40 is truly minimal
        or if we can do slightly better.
        
        Actually, for the tensor M_{n,m,p}, the minimum rank is known to be at least
        the "slice rank" which for matrix multiplication is max(nm, mp, np) = max(8, 20, 10) = 20.
        But the actual minimum rank could be higher.
        
        Let me just try to find any decomposition with rank < 40 via a more systematic search.
        """
        return None
    
    # Try a systematic approach: use the fact that we can sometimes reduce rank
    # by combining terms.
    
    def try_combined_decomposition():
        """
        Try to find a decomposition with rank < 40 by looking for linear dependencies.
        The schoolbook has 40 terms. We look for ways to combine them.
        
        Key insight: if we have two terms that share structure, we might combine them.
        For example, A[i,j]*B[j,k] and A[i,j']*B[j',k] for different j,j' can sometimes
        be combined using (A[i,j]+A[i,j'])*(B[j,k]+B[j',k]) - A[i,j]*B[j',k] - A[i,j']*B[j,k]
        but that adds terms.
        
        Let me try a gradient-based approach to find a lower-rank decomposition.
        """
        # Start from schoolbook and try to reduce rank by finding dependencies
        # This is complex, so let's try a simpler approach:
        # Use the known result that M_{2,4,5} might have rank less than 40.
        
        # For now, return None and fall back to schoolbook
        return None
    
    # Attempt random search for lower rank
    def random_search(max_rank, num_trials=2000):
        """Try to find a decomposition of rank max_rank."""
        np.random.seed(42)
        
        for trial in range(num_trials):
            if time.time() > _deadline - 15:
                break
            
            # Random initialization with small values
            U = np.random.randn(n * m, max_rank) * 0.3
            V = np.random.randn(m * p, max_rank) * 0.3
            W = np.random.randn(p * n, max_rank) * 0.3
            
            best_err = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for step in range(30):
                if time.time() > _deadline - 15:
                    break
                
                # Compute error using vectorized operations
                U_3d = U.reshape(n, m, max_rank)
                V_3d = V.reshape(m, p, max_rank)
                W_3d = W.reshape(p, n, max_rank)
                
                # Compute T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
                # Use einsum for speed
                # First compute U*V: (i,j,r)*(j,k,r) -> need to sum over r
                # UV[i,j,k,r] = U[i,j,r] * V[j,k,r]
                # Then T[i,j,k] = sum_r UV[i,j,k,r] * W[k,i,r]
                
                # Vectorized computation
                # UV[i,j,k,r] = U_3d[i,j,r] * V_3d[j,k,r]
                # This is tricky with einsum. Let's do:
                # temp[i,j,k,r] = U_3d[i,j,r] * V_3d[j,k,r]
                # We can compute this as: for each r, U_3d[:,:,r] * V_3d[:,:,r] gives (n,m) * (m,p)
                # but we need element-wise, not matrix mult.
                
                # Let's just use loops for correctness
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(max_rank):
                                s += U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
                            err += (s - 1.0) ** 2
                
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if best_err < 1e-9:
                    return best_U.tolist(), best_V.tolist(), best_W.tolist()
                
                # Adaptive step size
                step_size = max(0.01, 0.1 * (1.0 - step/30.0))
                U += np.random.randn(n * m, max_rank) * step_size
                V += np.random.randn(m * p, max_rank) * step_size
                W += np.random.randn(p * n, max_rank) * step_size
            
            if best_err < 1e-9:
                return best_U.tolist(), best_V.tolist(), best_W.tolist()
        
        return None
    
    # Try ranks from 39 down to 25
    for rank in range(39, 24, -1):
        if time.time() > _deadline - 40:
            break
        
        result = random_search(rank, num_trials=1000)
        if result is not None:
            U, V, W = result
            if verify(U, V, W):
                return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
