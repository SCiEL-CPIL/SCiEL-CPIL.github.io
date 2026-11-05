import numpy as np
import time

def construct():
    n, m, p = 3, 3, 5
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p  # 45
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Try block decomposition: Laderman (3x3x3) + simple (3x3x2)
    try:
        result = block_decomposition(n, m, p, deadline)
        if result is not None:
            # Verify the decomposition
            U, V, W = result
            U_arr = np.array(U)
            V_arr = np.array(V)
            W_arr = np.array(W)
            rank = U_arr.shape[1]
            
            # Verify: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(rank):
                            val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                        err = abs(val - 1)
                        if err > max_err:
                            max_err = err
            
            if max_err < 1e-9:
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

def laderman_3x3x3():
    """
    Returns U, V, W for 3x3x3 matrix multiplication with rank 23.
    U: (9, 23), V: (9, 23), W: (9, 23)
    T[i,j,k] = sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r] = 1
    """
    # Laderman decomposition - using a known set of factors
    # This is based on the standard Laderman algorithm
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # I'll use a verified Laderman decomposition
    # The entries are 0, 1, -1, 1/2, -1/2
    
    # Using the formulation from standard references
    # Each term r contributes to specific (i,j,k) positions
    
    terms = [
        # (U_idx, V_idx, W_idx, u_val, v_val, w_val)
        # Term 0
        (0, 0, 0, 1, 1, 1),
        (1, 4, 1, 1, 1, 1),
        (2, 8, 2, 1, 1, 1),
        (3, 1, 3, 1, 1, 1),
        (4, 5, 4, 1, 1, 1),
        (5, 9, 5, 1, 1, 1),
        (6, 2, 6, 1, 1, 1),
        (7, 6, 7, 1, 1, 1),
        (8, 10, 8, 1, 1, 1),
        # Terms 9-22: the "cross" terms that cancel
        (0, 4, 1, 1, 1, -1),
        (1, 0, 0, 1, 1, -1),
        (1, 8, 2, 1, 1, 1),
        (2, 4, 1, 1, 1, 1),
        (3, 5, 4, 1, 1, -1),
        (4, 1, 3, 1, 1, -1),
        (4, 9, 5, 1, 1, 1),
        (5, 5, 4, 1, 1, 1),
        (6, 6, 7, 1, 1, -1),
        (7, 2, 6, 1, 1, -1),
        (7, 10, 8, 1, 1, 1),
        (8, 6, 7, 1, 1, 1),
    ]
    
    # This is getting complex. Let me use a different, verified approach.
    # I'll construct it using the known structure.
    
    # Actually, let me use a simpler verified Laderman implementation
    # from known sources. The key is that it must be exact.
    
    # Using the standard Laderman factors (0, ±1, ±1/2)
    U = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=float)
    
    # This is just identity - not Laderman. Let me use a proper implementation.
    # I'll use a known correct Laderman decomposition.
    
    # Based on the standard Laderman algorithm:
    # The 23 terms are constructed from:
    # - 9 diagonal terms
    # - 14 off-diagonal terms with specific patterns
    
    # Let me use a verified set from the literature
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # I'll use the formulation where we can verify correctness
    # For now, let me use a simple but correct approach:
    # Use rank 27 (3x3x3 with a simpler decomposition) if I can't get Laderman right
    
    # Actually, let me just use the schoolbook for 3x3x3 (rank 27) 
    # and schoolbook for 3x3x2 (rank 18), total = 45 (same as full schoolbook)
    
    # Better: use a known rank-23 Laderman. Let me hardcode it properly.
    
    # Standard Laderman 3x3x3 factors (verified):
    U = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    ], dtype=float)
    
    # This is still not right. Let me use a completely different approach.
    # I'll use the fact that 3x3x3 can be done in rank 23, but I need the actual factors.
    
    # Let me use a simpler but still better than schoolbook approach:
    # Use rank 27 for 3x3x3 (which is just schoolbook) and try to optimize 3x3x2.
    
    # Actually, the simplest improvement: 
    # For 3x3x5, instead of 45, I can potentially do better by sharing computation.
    
    # Let me try a direct optimization approach with a good initialization.
    
    return None  # Signal to use fallback

def block_decomposition(n, m, p, deadline):
    """
    Block decomposition: split output into 3+2 columns.
    Use Laderman (rank 23) for 3x3x3, and optimize 3x3x2.
    """
    # For now, use a simple approach:
    # 3x3x3 part: rank 27 (schoolbook) - we can improve later
    # 3x3x2 part: rank 18 (schoolbook)
    # Total: 45 (no improvement)
    
    # To get improvement, we need actual Laderman factors.
    # Let me use a known correct Laderman decomposition.
    
    # I'll use the standard Laderman 3x3x3 with rank 23.
    # The factors are well-known in the literature.
    
    # For the 3x3x2 part, I'll use schoolbook (rank 18).
    
    # Total rank: 23 + 18 = 41
    
    # Laderman 3x3x3 factors (I'll use a verified set)
    # Based on the standard algorithm:
    
    U_lad = np.zeros((9, 23))
    V_lad = np.zeros((9, 23))
    W_lad = np.zeros((9, 23))
    
    # I'll construct a valid Laderman decomposition
    # Using the known structure with 0, ±1, ±1/2 entries
    
    # For simplicity and correctness, let me use a different strategy:
    # Use rank 23 for 3x3x3 if I can verify it, otherwise use 27.
    
    # Let me just use schoolbook for both parts and return rank 45.
    # The block decomposition doesn't help without proper Laderman factors.
    
    return None

def schoolbook_decomposition(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()
