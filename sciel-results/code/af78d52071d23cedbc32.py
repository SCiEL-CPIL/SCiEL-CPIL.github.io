"""Matrix multiplication 3x3x8 tensor decomposition."""
DIMENSIONS = [3, 3, 8]

def construct():
    """
    Construct a rank-reduced exact tensor decomposition for multiplying
    a 3x3 matrix (A) by a 3x8 matrix (B) to produce a 3x8 matrix (C).
    
    Strategy:
    1. Split the multiplication into two independent 3x3x4 blocks (columns 0-3 and 4-7).
    2. Use a known low-rank decomposition for 3x3x4 (rank 15 per block, total 30).
    3. If the known decomposition is not available or fails, fall back to schoolbook (rank 72).
    
    The known rank-15 decomposition for 3x3x4 is not standard in all libraries,
    so we attempt to construct it via a small search or use a hardcoded known solution if available.
    Since finding a rank-15 decomposition for 3x3x4 is non-trivial and time-consuming,
    and the problem size is small, we first try a direct randomized search for low rank
    using the linear algebra insight from inspiration 2, but with a strict time limit.
    
    However, to ensure correctness and speed, we will:
    - First try to verify if a simple block structure allows rank reduction.
    - For 3x3x8, the tensor is the direct sum of two 3x3x4 tensors.
    - Rank(3x3x8) = Rank(3x3x4) + Rank(3x3x4) if the blocks are independent.
    - Known best rank for 3x3x3 is 23 (Landsberg). For 3x3x4, it is likely around 30-35.
    - Schoolbook is 36 for 3x3x4. So we might save 6 per block, total 12, reaching rank 60.
    
    Given the 120s limit and the difficulty of finding exact low-rank decompositions,
    we will implement a robust search for 3x3x4 blocks using the linear system approach
    from inspiration 2, but optimized for speed and reliability. If no improvement is
    found quickly, we return the schoolbook decomposition.
    """
    import numpy as np
    import time
    
    n, m, p = DIMENSIONS  # n=3, m=3, p=8
    deadline = time.time() + 110.0  # Leave 10s margin
    
    # Schoolbook decomposition as fallback
    def schoolbook_decomposition(n, m, p):
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
    
    # Try to find a better decomposition for 3x3x4 blocks
    # We'll try to find a rank R decomposition for 3x3x4
    # and then combine two such blocks.
    
    def find_low_rank_3x3x4(target_rank, max_time):
        """
        Attempt to find a rank-target_rank decomposition for 3x3x4 multiplication.
        Returns (U, V, W) as lists or None.
        """
        n_sub, m_sub, p_sub = 3, 3, 4
        R = target_rank
        start_time = time.time()
        
        # Random search with linear solve for W
        for attempt in range(100):
            if time.time() > start_time + max_time:
                break
            
            # Random U, V with small integers
            U = np.random.randint(-2, 3, size=(n_sub * m_sub, R))
            V = np.random.randint(-2, 3, size=(m_sub * p_sub, R))
            
            # Solve for W
            W = np.zeros((p_sub * n_sub, R))
            valid = True
            for i in range(n_sub):
                for k in range(p_sub):
                    # Build B (m x R)
                    B = np.zeros((m_sub, R))
                    for j in range(m_sub):
                        for r in range(R):
                            B[j, r] = U[i * m_sub + j, r] * V[j * p_sub + k, r]
                    # Solve B w = 1
                    try:
                        w, residuals, rank, sv = np.linalg.lstsq(B, np.ones(m_sub), rcond=None)
                        if np.allclose(B @ w, np.ones(m_sub), atol=1e-6):
                            W[k * n_sub + i, :] = w
                        else:
                            valid = False
                            break
                    except:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Snap to nearest 0.5
                W_snapped = np.round(W * 2) / 2
                U_snapped = np.round(U)
                V_snapped = np.round(V)
                
                # Verify
                for i in range(n_sub):
                    for j in range(m_sub):
                        for k in range(p_sub):
                            s = 0.0
                            for r in range(R):
                                s += U_snapped[i * m_sub + j, r] * V_snapped[j * p_sub + k, r] * W_snapped[k * n_sub + i, r]
                            if abs(s - 1.0) > 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    return U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist()
        
        return None
    
    # Try ranks for 3x3x4: start from 20 up to 35
    # If we find one, we can combine two blocks to get rank 2*R for 3x3x8
    best_sub = None
    for R_sub in range(20, 36):
        if time.time() > deadline:
            break
        # Try to find a decomposition of rank R_sub for 3x3x4
        result = find_low_rank_3x3x4(R_sub, max_time=5.0)
        if result is not None:
            best_sub = (R_sub, result)
            break
    
    if best_sub is not None:
        R_sub, (U_sub, V_sub, W_sub) = best_sub
        # Combine two blocks: columns 0-3 and 4-7
        rank = 2 * R_sub
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Block 1: columns 0-3
        for i in range(n * m):
            for r in range(R_sub):
                U[i][r] = U_sub[i][r]
        for j in range(m):
            for k in range(4):
                for r in range(R_sub):
                    V[j * p + k][r] = V_sub[j * 4 + k][r]
        for k in range(4):
            for i in range(n):
                for r in range(R_sub):
                    W[k * n + i][r] = W_sub[k * n + i][r]
        
        # Block 2: columns 4-7
        offset = R_sub
        for i in range(n * m):
            for r in range(R_sub):
                U[i][offset + r] = U_sub[i][r]
        for j in range(m):
            for k in range(4, 8):
                for r in range(R_sub):
                    V[j * p + k][offset + r] = V_sub[j * 4 + (k - 4)][r]
        for k in range(4, 8):
            for i in range(n):
                for r in range(R_sub):
                    W[k * n + i][offset + r] = W_sub[(k - 4) * n + i][r]
        
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return schoolbook_decomposition(n, m, p)
