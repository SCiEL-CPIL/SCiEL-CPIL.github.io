"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # Try rank 23 (Coppersmith-Winograd) first, then 24, 25, 26
    for target_rank in [23, 24, 25, 26]:
        if time.time() > deadline:
            break
        result = _try_rank(n, m, p, target_rank, deadline)
        if result is not None:
            return result
    
    # Fallback: return a known rank-27 decomposition (simple)
    return _rank_27_fallback()

def _try_rank(n, m, p, target_rank, deadline):
    """
    Attempt to find a rank-`target_rank` decomposition with verification.
    """
    nm, mp, pn = n*m, m*p, p*n
    
    if target_rank == 23:
        result = _rank_23_cw(n, m, p)
        if result is not None:
            return result
    elif target_rank == 24:
        result = _rank_24_attempt(n, m, p)
        if result is not None:
            return result
    elif target_rank == 25:
        result = _rank_25_attempt(n, m, p)
        if result is not None:
            return result
    elif target_rank == 26:
        result = _rank_26_seed()
        if result is not None:
            return result
    
    return None

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    nm, mp, pn = n*m, m*p, p*n
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build target tensor
    T_target = np.zeros((nm, mp, pn), dtype=int)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = i * p + j
                T_target[a, b, c] = 1
    
    # Reconstruct
    T_recon = np.zeros((nm, mp, pn), dtype=float)
    for r in range(rank):
        T_recon += np.einsum('a,b,c->abc', U_arr[:, r], V_arr[:, r], W_arr[:, r])
    
    return np.allclose(T_recon, T_target, atol=1e-9)

def _rank_23_cw(n, m, p):
    """
    Attempt a rank-23 decomposition based on Coppersmith-Winograd.
    Uses a structured construction and verifies before returning.
    """
    nm, mp, pn = n*m, m*p, p*n
    rank = 23
    
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    # Coppersmith-Winograd rank-23 construction for 3x3 matrix multiplication
    # This is based on the algebraic structure of the tensor decomposition.
    # The construction uses the fact that 3x3 matrix multiplication can be
    # decomposed using the tensor rank 23.
    
    # Term 0: (A00+A11)(B00+B11) -> C00, C11
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[4][0] = 1
    
    # Term 1: (A10+A11)B00 -> C10, C11
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[4][1] = -1
    
    # Term 2: A00(B01-B11) -> C01, C11
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    W[3][2] = 1; W[4][2] = 1
    
    # Term 3: A11(B10-B00) -> C00, C10
    U[4][3] = 1
    V[3][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # Term 4: (A00+A01)B11 -> C00, C01
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[0][4] = -1; W[3][4] = 1
    
    # Term 5: (A10-A00)(B00+B01) -> C11
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[4][5] = 1
    
    # Term 6: (A01-A11)(B10+B11) -> C00
    U[1][6] = 1; U[4][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[0][6] = 1
    
    # Terms 7-12: Cross terms involving the third row/column
    # Using the CW structure to handle the boundary terms with rank 16
    
    # Term 7: A02*B20 -> C00
    U[2][7] = 1; V[6][7] = 1; W[0][7] = 1
    
    # Term 8: A02*B21 -> C01
    U[2][8] = 1; V[7][8] = 1; W[3][8] = 1
    
    # Term 9: A12*B20 -> C10
    U[5][9] = 1; V[6][9] = 1; W[1][9] = 1
    
    # Term 10: A12*B21 -> C11
    U[5][10] = 1; V[7][10] = 1; W[4][10] = 1
    
    # Term 11: A20*B00 -> C20
    U[6][11] = 1; V[0][11] = 1; W[2][11] = 1
    
    # Term 12: A20*B01 -> C21
    U[6][12] = 1; V[1][12] = 1; W[5][12] = 1
    
    # Term 13: A21*B00 -> C20
    U[7][13] = 1; V[0][13] = 1; W[2][13] = 1
    
    # Term 14: A21*B01 -> C21
    U[7][14] = 1; V[1][14] = 1; W[5][14] = 1
    
    # Term 15: A02*B22 -> C02
    U[2][15] = 1; V[8][15] = 1; W[6][15] = 1
    
    # Term 16: A12*B22 -> C12
    U[5][16] = 1; V[8][16] = 1; W[7][16] = 1
    
    # Term 17: A22*B00 -> C20
    U[8][17] = 1; V[0][17] = 1; W[2][17] = 1
    
    # Term 18: A22*B01 -> C21
    U[8][18] = 1; V[1][18] = 1; W[5][18] = 1
    
    # Term 19: A22*B02 -> C22
    U[8][19] = 1; V[2][19] = 1; W[8][19] = 1
    
    # Term 20: A20*B10 -> C21
    U[6][20] = 1; V[3][20] = 1; W[5][20] = 1
    
    # Term 21: A21*B10 -> C21
    U[7][21] = 1; V[3][21] = 1; W[5][21] = 1
    
    # Term 22: A22*B11 -> C22
    U[8][22] = 1; V[4][22] = 1; W[8][22] = 1
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    return None

def _rank_24_attempt(n, m, p):
    """Attempt rank-24 decomposition."""
    nm, mp, pn = n*m, m*p, p*n
    rank = 24
    
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    # Use the rank-26 structure but try to merge two terms
    # Start with the rank-26 seed and see if we can eliminate one term
    
    # For now, this is a placeholder - the rank-23 attempt is the primary goal
    return None

def _rank_25_attempt(n, m, p):
    """Attempt rank-25 decomposition."""
    nm, mp, pn = n*m, m*p, p*n
    rank = 25
    
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    # Placeholder - rank-23 is the primary goal
    return None

def _rank_27_fallback():
    """Return a simple rank-27 decomposition."""
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    rank = 27
    
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    for i in range(n):
        for k in range(m):
            for j in range(p):
                r = i*(m*p) + k*p + j
                U[i*m+k][r] = 1
                V[k*p+j][r] = 1
                W[i*p+j][r] = 1
    
    return {"U": U, "V": V, "W": W}

def _rank_26_seed():
    """Replicate the seed's rank-26 construction."""
    n, m, p = DIMENSIONS
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 1. A2 B2 (Top-left 2x2 * 2x2) -> Rank 7 (Strassen)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    
    U[0][col] = 1
    V[1][col] = 1; V[4][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    U[4][col] = 1
    V[3][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1
    col += 1
    
    # 2. a b^T (Top-right 2x1 * Bottom-left 1x2) -> Rank 4
    U[2][col] = 1; V[6][col] = 1; W[0][col] = 1; col += 1
    U[2][col] = 1; V[7][col] = 1; W[3][col] = 1; col += 1
    U[5][col] = 1; V[6][col] = 1; W[1][col] = 1; col += 1
    U[5][col] = 1; V[7][col] = 1; W[4][col] = 1; col += 1
    
    # 3. A2 b (Top-left 2x2 * Top-right 2x1) -> Rank 4
    U[0][col] = 1; V[2][col] = 1; W[6][col] = 1; col += 1
    U[1][col] = 1; V[5][col] = 1; W[6][col] = 1; col += 1
    U[3][col] = 1; V[2][col] = 1; W[7][col] = 1; col += 1
    U[4][col] = 1; V[5][col] = 1; W[7][col] = 1; col += 1
    
    # 4. a b3 (Top-right 2x1 * Bottom-right 1x1) -> Rank 2
    U[2][col] = 1; V[8][col] = 1; W[6][col] = 1; col += 1
    U[5][col] = 1; V[8][col] = 1; W[7][col] = 1; col += 1
    
    # 5. a^T B2 (Bottom-left 1x2 * Top-left 2x2) -> Rank 4
    U[6][col] = 1; V[0][col] = 1; W[2][col] = 1; col += 1
    U[7][col] = 1; V[3][col] = 1; W[2][col] = 1; col += 1
    U[6][col] = 1; V[1][col] = 1; W[5][col] = 1; col += 1
    U[7][col] = 1; V[4][col] = 1; W[5][col] = 1; col += 1
    
    # 6. a3 b^T (Bottom-right 1x1 * Bottom-left 1x2) -> Rank 2
    U[8][col] = 1; V[6][col] = 1; W[2][col] = 1; col += 1
    U[8][col] = 1; V[7][col] = 1; W[5][col] = 1; col += 1
    
    # 7. a^T b (Bottom-left 1x2 * Top-right 2x1) -> Rank 2
    U[6][col] = 1; V[2][col] = 1; W[8][col] = 1; col += 1
    U[7][col] = 1; V[5][col] = 1; W[8][col] = 1; col += 1
    
    # 8. a3 b3 (Bottom-right 1x1 * Bottom-right 1x1) -> Rank 1
    U[8][col] = 1; V[8][col] = 1; W[8][col] = 1; col += 1
    
    # Verify before returning
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        return None
# EVOLVE-BLOCK-END
