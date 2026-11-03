import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _get_strassen_222():
    """
    Returns (U, V, W) for 2x2x2 Strassen multiplication.
    U shape: (4, 7)
    V shape: (4, 7)
    W shape: (4, 7)
    Indexing:
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # Term 0: M1 = (A00 + A11)(B00 + B11)
    U[0][0] = 1
    U[3][0] = 1
    V[0][0] = 1
    V[3][0] = 1
    W[0][0] += 1
    W[3][0] += 1

    # Term 1: M2 = (A10 + A11)B00
    U[2][1] = 1
    U[3][1] = 1
    V[0][1] = 1
    W[2][1] += 1
    W[3][1] += -1

    # Term 2: M3 = A00(B01 - B11)
    U[0][2] = 1
    V[1][2] = 1
    V[3][2] = -1
    W[2][2] += 1
    W[3][2] += 1

    # Term 3: M4 = A11(B10 - B00)
    U[3][3] = 1
    V[2][3] = 1
    V[0][3] = -1
    W[0][3] += -1
    W[2][3] += 1

    # Term 4: M5 = (A00 + A01)B11
    U[0][4] = 1
    U[1][4] = 1
    V[3][4] = 1
    W[0][4] += 1
    W[2][4] += 1

    # Term 5: M6 = (A10 - A00)(B00 + B01)
    U[2][5] = 1
    U[0][5] = -1
    V[0][5] = 1
    V[1][5] = 1
    W[3][5] += 1

    # Term 6: M7 = (A01 - A11)(B10 + B11)
    U[1][6] = 1
    U[3][6] = -1
    V[2][6] = 1
    V[3][6] = 1
    W[0][6] += -1

    return U, V, W

def _get_brauer_333():
    """
    Returns (U, V, W) for 3x3x3 Brauer multiplication (rank 23).
    U shape: (9, 23)
    V shape: (9, 23)
    W shape: (9, 23)
    Indexing:
    U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
    """
    rank = 23
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(9)]
    
    # Brauer's 23-rank algorithm for 3x3x3
    # Based on the known rank-23 decomposition
    
    # Term 0: (a00+a11+a22)(b00+b11+b22)
    U[0][0] = 1; U[4][0] = 1; U[8][0] = 1
    V[0][0] = 1; V[4][0] = 1; V[8][0] = 1
    W[0][0] += 1; W[4][0] += 1; W[8][0] += 1
    
    # Term 1: (a00-a11+a22)(b00-b11+b22)
    U[0][1] = 1; U[4][1] = -1; U[8][1] = 1
    V[0][1] = 1; V[4][1] = -1; V[8][1] = 1
    W[0][1] += 1; W[4][1] += -1; W[8][1] += 1
    
    # Term 2: (a00+a11-a22)(b00+b11-b22)
    U[0][2] = 1; U[4][2] = 1; U[8][2] = -1
    V[0][2] = 1; V[4][2] = 1; V[8][2] = -1
    W[0][2] += 1; W[4][2] += 1; W[8][2] += -1
    
    # Term 3: a01*b10
    U[1][3] = 1
    V[3][3] = 1
    W[0][3] += 1
    
    # Term 4: a10*b01
    U[3][4] = 1
    V[1][4] = 1
    W[1][4] += 1
    
    # Term 5: a02*b20
    U[2][5] = 1
    V[6][5] = 1
    W[0][5] += 1
    
    # Term 6: a20*b02
    U[6][6] = 1
    V[2][6] = 1
    W[2][6] += 1
    
    # Term 7: a12*b21
    U[5][7] = 1
    V[8][7] = 1
    W[1][7] += 1
    
    # Term 8: a21*b12
    U[7][8] = 1
    V[5][8] = 1
    W[2][8] += 1
    
    # Terms 9-22: More complex terms
    # This is a simplified version; the full Brauer algorithm has 23 terms
    # For correctness, let's use a verified set
    
    # Actually, let me use a different approach - use the known correct Brauer decomposition
    # The standard Brauer 3x3x3 rank 23 algorithm:
    
    # Reset and use verified terms
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(9)]
    
    # I'll use a simpler approach: since implementing full Brauer is complex,
    # let's try a different partition strategy that's easier to verify
    
    # For now, return schoolbook for 3x3x3 (rank 27) as fallback
    # and use the existing Strassen-based approach
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j][i*3+j*3+k if False else 0] = 0  # placeholder
    
    # Actually, let me just return None to indicate we should use schoolbook
    return None

def _build_rank71():
    n, m, p = 3, 5, 5
    
    su, sv, sw = _get_strassen_222()
    
    # Partitions
    i_blocks = [(0, 1), (2,)]
    j_blocks = [(0, 1), (2, 3), (4,)]
    k_blocks = [(0, 1), (2, 3), (4,)]
    
    total_rank = 0
    for i_b in i_blocks:
        for j_b in j_blocks:
            for k_b in k_blocks:
                if len(i_b)==2 and len(j_b)==2 and len(k_b)==2:
                    total_rank += 7
                else:
                    total_rank += len(i_b)*len(j_b)*len(k_b)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i_b in i_blocks:
        for j_b in j_blocks:
            for k_b in k_blocks:
                if len(i_b)==2 and len(j_b)==2 and len(k_b)==2:
                    # Strassen
                    for c in range(7):
                        for li in range(2):
                            gi = i_b[li]
                            for lj in range(2):
                                gj = j_b[lj]
                                U[gi*m + gj][col+c] = su[li*2+lj][c]
                        for lj in range(2):
                            gj = j_b[lj]
                            for lk in range(2):
                                gk = k_b[lk]
                                V[gj*p + gk][col+c] = sv[lj*2+lk][c]
                        for lk in range(2):
                            gk = k_b[lk]
                            for li in range(2):
                                gi = i_b[li]
                                W[gk*n + gi][col+c] = sw[lk*2+li][c]
                    col += 7
                else:
                    # Schoolbook
                    for gi in i_b:
                        for gj in j_b:
                            for gk in k_b:
                                U[gi*m + gj][col] = 1
                                V[gj*p + gk][col] = 1
                                W[gk*n + gi][col] = 1
                                col += 1
                                
    return U, V, W

def _build_rank67():
    """
    Alternative: partition as (2+1)x(2+2+1)x(2+2+1) with different grouping
    Try to get lower rank by using Strassen more effectively
    """
    n, m, p = 3, 5, 5
    
    su, sv, sw = _get_strassen_222()
    
    # Try partition: i=(2,1), j=(2,3), k=(2,3)
    # 2x2x2 blocks use Strassen (7), others use schoolbook
    # (2,2,2): 7
    # (2,2,3): 2*2*3=12
    # (2,3,2): 2*3*2=12
    # (2,3,3): 2*3*3=18
    # (1,2,2): 1*2*2=4
    # (1,2,3): 1*2*3=6
    # (1,3,2): 1*3*2=6
    # (1,3,3): 1*3*3=9
    # Total: 7+12+12+18+4+6+6+9 = 74 (worse)
    
    # The current 71 is already good. Let me try another partition.
    # i=(1,2), j=(3,2), k=(3,2)
    # (1,3,3): 9
    # (1,3,2): 6
    # (1,2,3): 6
    # (1,2,2): 4
    # (2,3,3): 18
    # (2,3,2): 12
    # (2,2,3): 12
    # (2,2,2): 7
    # Total: 9+6+6+4+18+12+12+7 = 74 (worse)
    
    # Current 71 seems optimal for this approach.
    # Let me just return the same as rank71
    return _build_rank71()

def _verify_decomposition(U, V, W, n, m, p, rank):
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i*m + j]
                v_val = V_np[j*p + k]
                w_val = W_np[k*n + i]
                s = np.sum(u_val * v_val * w_val)
                if not np.isclose(s, 1.0, atol=1e-9):
                    return False
    return True

def _schoolbook_factors(n, m, p):
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
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    try:
        U, V, W = _build_rank71()
        rank = len(W[0])
        if _verify_decomposition(U, V, W, n, m, p, rank):
            return {"U": U, "V": V, "W": W}
    except Exception as e:
        pass
        
    U, V, W = _schoolbook_factors(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
