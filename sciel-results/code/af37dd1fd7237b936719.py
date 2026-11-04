"""Matrix multiplication 5x5x5 with reduced rank decomposition."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition.
    U, V, W are lists of 7 vectors, each of length 4.
    Indexing: A[i,j] -> i*2+j, B[j,k] -> j*2+k, C[k,i] -> k*2+i
    """
    U = [
        [1, 0, 0, 1],   # P1: A00+A11
        [0, 0, 1, 1],   # P2: A10+A11
        [1, 0, 0, 0],   # P3: A00
        [0, 0, 0, 1],   # P4: A11
        [1, 0, 1, 0],   # P5: A00+A10
        [-1, 1, 0, 0],  # P6: A01-A00
        [0, 0, 1, -1],  # P7: A10-A11
    ]
    V = [
        [1, 0, 0, 1],   # P1: B00+B11
        [1, 0, 0, 0],   # P2: B00
        [0, 0, 1, -1],  # P3: B10-B11
        [-1, 1, 0, 0],  # P4: B01-B00
        [0, 0, 0, 1],   # P5: B11
        [1, 0, 1, 0],   # P6: B00+B10
        [0, 1, 0, 1],   # P7: B01+B11
    ]
    W = [
        [1, 0, 0, 1],   # P1: C00+C11
        [0, 0, 1, -1],  # P2: C10-C11
        [0, 1, 0, 1],   # P3: C01+C11
        [1, 0, 1, 0],   # P4: C00+C10
        [-1, 1, 0, 0],  # P5: -C00+C01
        [0, 0, 0, 1],   # P6: C11
        [1, 0, 0, 0],   # P7: C00
    ]
    return U, V, W


def _build_4x4_strassen():
    """Build 4x4x4 Strassen decomposition (rank 49).
    Returns U, V, W as lists of 49 vectors, each of length 16.
    """
    U2, V2, W2 = _strassen_2x2()
    
    U4 = []
    V4 = []
    W4 = []
    
    for r in range(7):
        for s in range(7):
            u = [0] * 16
            v = [0] * 16
            w = [0] * 16
            
            # For each block position (bi, bj) in A (2x2 blocks)
            for bi in range(2):
                for bj in range(2):
                    # Top-level coefficient for A block (bi,bj)
                    a_top = U2[r][bi * 2 + bj]
                    # Top-level coefficient for B block (bj,bi) -- B's row block is bj, col block is bi
                    b_top = V2[r][bj * 2 + bi]
                    # Top-level coefficient for C block (bi,bi) -- C's row block is bi, col block is bi? No.
                    # C block (ci, ck) where ci=bi, ck=bi? No, that's not right either.
                    
                    # Let me reconsider. In Strassen for 2x2 blocks:
                    # P1 = (A00+A11)*(B00+B11)
                    # This means: for the A part, we use blocks 00 and 11 with coeff 1
                    # For the B part, we use blocks 00 and 11 with coeff 1
                    # The result contributes to C00 and C11.
                    
                    # So for product r:
                    # A uses blocks determined by U2[r]
                    # B uses blocks determined by V2[r]
                    # C uses blocks determined by W2[r]
                    
                    # But the blocks for A and B are different!
                    # A block (bi,bj) and B block (bj,bk) where bk is determined by V2[r].
                    
                    # I think the correct way is:
                    # For product r, the A part is sum_{bi,bj} U2[r][bi*2+bj] * A_block(bi,bj)
                    # The B part is sum_{bj,bk} V2[r][bj*2+bk] * B_block(bj,bk)
                    # The C part is sum_{bi,bk} W2[r][bi*2+bk] * C_block(bi,bk)
                    
                    # For the tensor decomposition, each term (r,s) has:
                    # U vector: for each A[i,j], coefficient = sum over block positions
                    # This is getting too complex. Let me just compute it numerically.
                    
                    pass
            
            # Let me use a different approach: directly compute the rank-1 terms
            # by considering how Strassen works at each level.
            
            # For the 4x4x4 tensor, I'll use the fact that it can be decomposed
            # using the 49-term Strassen algorithm.
            
            # Each term corresponds to a scalar multiplication in the algorithm.
            
            # Term (r,s) where r is top-level product, s is bottom-level product:
            # The A entries involved are determined by U2[r] and U2[s]
            # The B entries involved are determined by V2[r] and V2[s]
            # The C entries involved are determined by W2[r] and W2[s]
            
            # For A[i,j] where i = i1*2+i2, j = j1*2+j2:
            #   coeff = U2[r][i1*2+j1] * U2[s][i2*2+j2]
            
            # For B[j,k] where j = j1*2+j2, k = k1*2+k2:
            #   coeff = V2[r][j1*2+k1] * V2[s][j2*2+k2]
            
            # For C[i,k] where i = i1*2+i2, k = k1*2+k2:
            #   coeff = W2[r][i1*2+k1] * W2[s][i2*2+k2]
            
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            u[i * 4 + j] = U2[r][i1 * 2 + j1] * U2[s][i2 * 2 + j2]
            
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            v[j * 4 + k] = V2[r][j1 * 2 + k1] * V2[s][j2 * 2 + k2]
            
            for i1 in range(2):
                for k1 in range(2):
                    for i2 in range(2):
                        for k2 in range(2):
                            i = i1 * 2 + i2
                            k = k1 * 2 + k2
                            # W is indexed by k*4+i
                            w[k * 4 + i] = W2[r][i1 * 2 + k1] * W2[s][i2 * 2 + k2]
            
            U4.append(u)
            V4.append(v)
            W4.append(w)
    
    return U4, V4, W4


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False, (i, j, k, total)
    return True, None


def construct():
    deadline = time.time() + 110  # Leave 10s margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Build 4x4x4 Strassen decomposition
    U4, V4, W4 = _build_4x4_strassen()
    
    # Verify 4x4x4 decomposition
    ok, info = _verify_decomposition(U4, V4, W4, 4, 4, 4)
    if not ok:
        # If Strassen 4x4 is broken, fallback to schoolbook immediately
        rank = 125
        U5 = [[0] * rank for _ in range(25)]
        V5 = [[0] * rank for _ in range(25)]
        W5 = [[0] * rank for _ in range(25)]
        col = 0
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    U5[i * 5 + j][col] = 1
                    V5[j * 5 + k][col] = 1
                    W5[k * 5 + i][col] = 1
                    col += 1
        return {"U": U5, "V": V5, "W": W5}

    # Build 5x5x5 decomposition using hybrid approach:
    # 1. 4x4x4 Strassen core (Rank 49) for indices 0..3
    # 2. Explicit terms for any index being 4
    
    # Total rank calculation:
    # Core: 49
    # i=4 (all j,k): 25 terms
    # k=4, i<4 (all j): 20 terms
    # j=4, i<4, k<4: 16 terms
    # Total: 49 + 25 + 20 + 16 = 110
    
    total_rank = 110
    U5 = [[0] * total_rank for _ in range(25)]
    V5 = [[0] * total_rank for _ in range(25)]
    W5 = [[0] * total_rank for _ in range(25)]
    
    col = 0
    
    # 1. Copy 4x4x4 Strassen part
    for r in range(49):
        for i in range(4):
            for j in range(4):
                U5[i * 5 + j][col] = U4[r][i * 4 + j]
        for j in range(4):
            for k in range(4):
                V5[j * 5 + k][col] = V4[r][j * 4 + k]
        for k in range(4):
            for i in range(4):
                W5[k * 5 + i][col] = W4[r][k * 4 + i]
        col += 1
        
    # 2. Terms where i=4: A[4,j]*B[j,k] for j,k in 0..4
    # These contribute to C[4,k]
    for j in range(5):
        for k in range(5):
            U5[4 * 5 + j][col] = 1
            V5[j * 5 + k][col] = 1
            W5[k * 5 + 4][col] = 1
            col += 1
            
    # 3. Terms where k=4, i<4: A[i,j]*B[j,4] for i in 0..3, j in 0..4
    # These contribute to C[i,4]
    for i in range(4):
        for j in range(5):
            U5[i * 5 + j][col] = 1
            V5[j * 5 + 4][col] = 1
            W5[4 * 5 + i][col] = 1
            col += 1
            
    # 4. Terms where j=4, i<4, k<4: A[i,4]*B[4,k] for i in 0..3, k in 0..3
    # These contribute to C[i,k]
    for i in range(4):
        for k in range(4):
            U5[i * 5 + 4][col] = 1
            V5[4 * 5 + k][col] = 1
            W5[k * 5 + i][col] = 1
            col += 1
            
    # Verify 5x5x5 decomposition
    ok, info = _verify_decomposition(U5, V5, W5, 5, 5, 5)
    
    if not ok:
        # Fallback to schoolbook if verification fails (should not happen)
        rank = 125
        U5 = [[0] * rank for _ in range(25)]
        V5 = [[0] * rank for _ in range(25)]
        W5 = [[0] * rank for _ in range(25)]
        col = 0
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    U5[i * 5 + j][col] = 1
                    V5[j * 5 + k][col] = 1
                    W5[k * 5 + i][col] = 1
                    col += 1
    
    return {"U": U5, "V": V5, "W": W5}
# EVOLVE-BLOCK-END
