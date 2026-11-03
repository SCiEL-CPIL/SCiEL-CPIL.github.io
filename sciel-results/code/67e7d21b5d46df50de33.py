import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # We'll construct a decomposition with rank 32:
    # - 28 terms for 3x4x4 (two 3x2x2 Strassen blocks)
    # - 4 terms for the 5th column
    
    terms = []  # list of (u, v, w) where u is 12-vector, v is 20-vector, w is 15-vector
    
    # Strassen 3x2x2: A (3x2) * B (2x2) -> C (3x2)
    # Using standard Strassen for 2x2x2, applied to each row of A
    # Actually, let's use a different approach: decompose 3x2x2 directly
    
    # For 3x2x2, we can use:
    # M1 = (A1+A2) * (B1+B2)  -- but this is for 2x2
    # Let me use the fact that 3x2x2 has rank at most 3*2*2 = 12 (trivial)
    # But we can do better.
    
    # Actually, let me use a simpler approach:
    # 3x4x5 = 3x4x4 + 3x4x1
    # 3x4x4: split into two 3x2x2 blocks
    # 3x2x2: use the trivial 12-term decomposition for now, but try to optimize
    
    # For 3x2x2, the trivial rank is 12. Can we do better?
    # rank(3x2x2) <= rank(2x2x2) + rank(1x2x2) = 7 + 2 = 9
    # Or: rank(3x2x2) <= 3 * rank(1x2x2) = 3*2 = 6? No, that's not right.
    # Actually, rank(3x2x2) is at most 12, and I believe it's exactly 12 or close.
    
    # Let me just use the trivial decomposition for 3x2x2 (rank 12)
    # Two blocks: 24 terms + 4 terms = 28 total
    
    # Wait, that's still better than 60!
    
    # Block 1: rows 0,1 of B (columns 0-3 of A)
    # Block 2: rows 2,3 of B (columns 0-3 of A)
    
    col = 0
    for block in range(2):
        for i in range(3):
            for j in range(2):
                for k in range(2):
                    u = [0.0] * 12
                    v = [0.0] * 20
                    w = [0.0] * 15
                    u[i * 4 + j] = 1.0
                    v[(block * 2 + j) * 5 + k] = 1.0
                    w[k * 3 + i] = 1.0
                    terms.append((u, v, w))
                    col += 1
    
    # 5th column: A (3x4) * B_col4 (4x1) -> C (3x1)
    for i in range(3):
        for j in range(4):
            u = [0.0] * 12
            v = [0.0] * 20
            w = [0.0] * 15
            u[i * 4 + j] = 1.0
            v[j * 5 + 4] = 1.0
            w[4 * 3 + i] = 1.0
            terms.append((u, v, w))
            col += 1
    
    rank = len(terms)
    
    # Build matrices
    U = np.zeros((12, rank))
    V = np.zeros((20, rank))
    W = np.zeros((15, rank))
    
    for r, (u, v, w) in enumerate(terms):
        U[:, r] = u
        V[:, r] = v
        W[:, r] = w
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

