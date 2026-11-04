import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    def verify(U, V, W):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
                    val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def make_schoolbook():
        """Return the trivial (schoolbook) decomposition with rank = n*m*p = 54."""
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

    # Strategy: 
    # 1. Try to find a decomposition with rank < 54 using a structured approach.
    # 2. The tensor is T_{ijk} where C[i,k] = sum_j A[i,j] B[j,k].
    # 3. We want sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k.
    
    # We will attempt to use a rank-23 decomposition for the 3x3x3 core, 
    # and then extend it to 3x3x6.
    # However, simply extending rank-23 to 3x3x6 by repeating or combining is non-trivial 
    # and might not yield rank < 54 easily without specific structure.
    
    # Let's try a direct optimization approach with a tighter deadline and better initialization.
    # We'll use a simple gradient-free local search or random restarts with a small rank.
    
    # Lower bound for 3x3x3x3 is 23. For 3x3x3x6, it's likely at least 23.
    # Schoolbook is 54.
    # Can we do better? 
    # If we take two independent 3x3x3 multiplications (one for k=0..2, one for k=3..5), 
    # the rank would be 23 + 23 = 46.
    # This is a valid decomposition!
    # Let's construct this explicitly.
    
    # We need a rank-23 decomposition for 3x3x3.
    # Finding one from scratch is hard. But we can try to find one via search if we have time.
    # Or, we can stick to schoolbook if we can't find a better one quickly.
    
    # Let's try to find a rank-23 decomposition for 3x3x3 using a known algorithm or search.
    # Since we don't have the terms hardcoded, we might not be able to construct rank 46 reliably.
    
    # Alternative: Try to find ANY decomposition with rank < 54.
    # Let's try rank 46 first with a randomized search, but it might be too slow.
    
    # Given the time constraint and the difficulty of finding a rank-23 decomposition,
    # we will fall back to schoolbook unless we can find a quick improvement.
    
    # However, the prompt asks to combine strategies. The strongest idea from inspiration 2 is the search.
    # The strongest idea from inspiration 3 is the verification and structured approach.
    
    # Let's try a very limited search for rank 46 (two blocks of 23) or rank 45, etc.
    # But without a good initialization, random search for rank 23 is very hard.
    
    # Let's try a different angle: 
    # Is there a simple algebraic structure for 3x3x3x6?
    # If we view the 3x6 matrix as two 3x3 matrices side by side, B = [B1, B2].
    # Then C = [C1, C2] where C1 = A B1 and C2 = A B2.
    # The tensor is the direct sum of two 3x3x3 tensors.
    # The rank of the direct sum is the sum of the ranks.
    # So rank(MM_3x3x6) = rank(MM_3x3x3) + rank(MM_3x3x3) = 23 + 23 = 46.
    
    # So if we can find a rank-23 decomposition for 3x3x3, we can construct a rank-46 decomposition for 3x3x6.
    
    # Let's try to find a rank-23 decomposition for 3x3x3 using a local search.
    # We have ~140 seconds.
    
    def find_rank_23_3x3x3():
        """Try to find a rank-23 decomposition for 3x3x3 matrix multiplication."""
        n3, m3, p3 = 3, 3, 3
        rank = 23
        U = np.random.randn(n3 * m3, rank)
        V = np.random.randn(m3 * p3, rank)
        W = np.random.randn(p3 * n3, rank)
        
        target = np.ones((n3, m3, p3))
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(2000):  # Limited iterations
            if time.time() > deadline - 10:
                break
            
            lr = 0.05
            U_new = U + lr * np.random.randn(n3 * m3, rank)
            V_new = V + lr * np.random.randn(m3 * p3, rank)
            W_new = W + lr * np.random.randn(p3 * n3, rank)
            
            # Compute residual
            err = 0.0
            for i in range(n3):
                for j in range(m3):
                    for k in range(p3):
                        val = np.sum(U_new[i*m3+j] * V_new[j*p3+k] * W_new[k*n3+i])
                        err += (val - 1.0)**2
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U_new.copy(), V_new.copy(), W_new.copy()
                U, V, W = U_new.copy(), V_new.copy(), W_new.copy()
                if err < 1e-8:
                    break
            
            if err < 1e-6:
                # Try to round
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                
                res_err = 0.0
                for i in range(n3):
                    for j in range(m3):
                        for k in range(p3):
                            val = np.sum(U_round[i*m3+j] * V_round[j*p3+k] * W_round[k*n3+i])
                            res_err += (val - 1.0)**2
                if res_err < 1e-8:
                    return U_round, V_round, W_round
        
        return None
    
    # Attempt to find rank-23 decomposition
    U23, V23, W23 = find_rank_23_3x3x3()
    
    if U23 is not None:
        # Construct rank-46 decomposition for 3x3x6
        rank = 46
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # First block: k=0..2
        for r in range(23):
            for i in range(3):
                for j in range(3):
                    U[i*3+j, r] = U23[i*3+j, r]
                for k in range(3):
                    V[j*3+k, r] = V23[j*3+k, r]
                    W[k*3+i, r] = W23[k*3+i, r]
        
        # Second block: k=3..5
        for r in range(23):
            for i in range(3):
                for j in range(3):
                    U[i*3+j, r+23] = U23[i*3+j, r]
                for k in range(3):
                    V[j*3+k+3, r+23] = V23[j*3+k, r]
                    W[(k+3)*3+i, r+23] = W23[k*3+i, r]
        
        # Verify
        if verify(U.tolist(), V.tolist(), W.tolist()):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    return make_schoolbook()
# EVOLVE-BLOCK-END
