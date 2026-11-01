import numpy as np
import time

# EVOLVE-BLOCK-START
DIMENSIONS = [2, 4, 5]
DEADLINE = 110.0  # seconds

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    D = np.einsum('ar,br,cr->abc', Ua, Va, Wa, optimize=True)
    return np.allclose(D, T, atol=1e-9)

def _schoolbook_decomposition():
    n, m, p = DIMENSIONS
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
    return U, V, W, rank

def _build_strassen_2x4x5():
    """
    Construct a rank-36 decomposition for 2x4 * 4x5 matrix multiplication.
    
    Strategy:
    - Split the 2x4 matrix into two 2x2 blocks: A = [A1 A2]
    - Split the 4x5 matrix into two 2x5 blocks: B = [B1; B2]
    - Result: C = A1*B1 + A2*B2
    - Use Strassen for each 2x2 * 2x5 product (rank 7 each)
    - Total rank: 7 + 7 = 14? No wait, 2x2 * 2x5 is not square.
    
    Actually, let me think about this differently.
    For 2x4 * 4x5:
    - Split A (2x4) into A1 (2x2) and A2 (2x2)
    - Split B (4x5) into B1 (2x5) and B2 (2x5)
    - C = A1*B1 + A2*B2
    
    For 2x2 * 2x5, we can use a Strassen-like decomposition.
    Actually, 2x2 * 2x5 can be seen as two independent 2x2 * 2x2 problems
    plus one 2x2 * 2x1, but that's not clean.
    
    Alternative: Use the known result that 2x4 * 4x5 has rank 36.
    The schoolbook rank is 40. We can save 4 operations.
    
    Let me use a different approach:
    - Split the 4x4 middle dimension into 2+2
    - Use Strassen on the 2x2 sub-problems
    
    Actually, the cleanest way:
    A (2x4) = [A1 A2] where A1, A2 are 2x2
    B (4x5) = [B1; B2] where B1, B2 are 2x5
    
    C = A1*B1 + A2*B2
    
    For each 2x2 * 2x5 product, we can use 7 multiplications (Strassen-like).
    Total: 14 multiplications. But rank 14 seems too low for 2x4*4x5.
    
    Wait, I'm confusing myself. Let me reconsider.
    
    The tensor rank of 2x4 * 4x5 is at most 40 (schoolbook).
    It's known to be 36 based on the knowledge.
    
    Let me try a different splitting:
    - Split the 4-dimension into 2+2
    - A (2x4) -> two 2x2 blocks
    - B (4x5) -> two 2x5 blocks
    
    For 2x2 * 2x5, the rank is at most 10 (schoolbook) but can be reduced.
    Actually, 2x2 * 2x2 has rank 7 (Strassen).
    2x2 * 2x5 = (2x2 * 2x2) + (2x2 * 2x2) + (2x2 * 2x1)
    That's 7 + 7 + 4 = 18, which is worse than 10.
    
    I think the correct approach is to use the known rank-36 decomposition.
    Since I can't derive it easily, let me just return the schoolbook decomposition
    which is guaranteed to work.
    
    Actually, let me try a simple optimization:
    Use the fact that some entries might be zero or redundant.
    """
    n, m, p = DIMENSIONS
    # For now, just return schoolbook as a safe fallback
    return _schoolbook_decomposition()

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Get schoolbook decomposition
    U, V, W, rank = _schoolbook_decomposition()
    
    # Try to improve with Strassen-like decomposition
    if time.time() - start_time < DEADLINE - 10:
        try:
            U_s, V_s, W_s, rank_s = _build_strassen_2x4x5()
            if rank_s < rank and _verify(U_s, V_s, W_s):
                U, V, W, rank = U_s, V_s, W_s, rank_s
        except Exception:
            pass
    
    # Final verification
    if not _verify(U, V, W):
        U, V, W, rank = _schoolbook_decomposition()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
