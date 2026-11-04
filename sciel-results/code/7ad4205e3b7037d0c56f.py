"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    deadline = time.time() + 140.0  # safety margin before 150s kill

    # Strategy:
    # The tensor for multiplying (n x m) by (m x p) -> (n x p) has:
    #   U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    #   T[a, b, c] = 1 where a = i*m+j, b = j*p+k, c = k*n+i
    #
    # Key insight: if we split the output dimension p=8 into two blocks of 4,
    # the problem decomposes into two independent 4x4 x 4x4 multiplications.
    # Each block can use the known rank-14 BCLR decomposition for 4x4x4.
    # Total rank = 2 * 14 = 28.
    #
    # However, the BCLR rank-14 table in the seed was incorrect (it actually
    # encoded rank-16 schoolbook-like terms). We need a CORRECT rank-14 or
    # better decomposition for 4x4x4.
    #
    # The true minimal rank for 4x4x4 matrix multiplication is not known to be
    # 14 (that was the BCLR upper bound which turned out to be wrong; the
    # actual best known upper bound for 4x4x4 is 49 via Strassen-like methods,
    # and the lower bound is 25).
    #
    # So let's use a correct construction:
    # - Use the Strassen 2x2x2 rank-7 algorithm recursively.
    # - 4x4x4 = 4 blocks of 2x2x2, but Strassen reduces this.
    # - Standard Strassen for 4x4x4: divide into 2x2 blocks of 2x2 matrices.
    #   7 multiplications of 2x2x2 (rank 7 each) -> total rank 7*7 = 49.
    #
    # For our problem 4x4 x 4x8:
    #   A is 4x4, B is 4x8, C is 4x8.
    #   Split B into B1 (4x4) and B2 (4x4), C into C1 (4x4) and C2 (4x4).
    #   C1 = A * B1, C2 = A * B2.
    #   Each is a 4x4x4 multiplication.
    #   Using rank-49 for each: total rank = 98.
    #
    # But we can do better by sharing the A-side Strassen factors.
    # In Strassen, the A-side factors (the 7 combinations of A-blocks) are
    # shared between computing C1 and C2. So:
    #   - Compute the 7 A-combinations (rank contribution from U side).
    #   - For each of the 7 A-combs, compute 7 B-combinations for B1 and
    #     7 B-combinations for B2.
    #   - Total rank = 7 (A-combs) * 7 (B1-combs) + 7 (A-combs) * 7 (B2-combs)
    #     = 49 + 49 = 98.
    #   Wait, that's the same. The sharing is in the U matrix, not in rank.
    #
    # Actually, the rank is the number of scalar multiplications.
    # For 4x4 x 4x8 with Strassen:
    #   A (4x4) split into 4 blocks of 2x2: A11, A12, A21, A22
    #   B (4x8) split into 8 blocks of 2x2: B11, B12, B21, B22, B31, B32, B41, B42
    #   C (4x8) split into 8 blocks of 2x2: C11, C12, C21, C22, C31, C32, C41, C42
    #
    #   C11 = A11*B11 + A12*B21
    #   C12 = A11*B12 + A12*B22
    #   C21 = A21*B11 + A22*B21
    #   C22 = A21*B12 + A22*B22
    #   C31 = A11*B31 + A12*B41
    #   C32 = A11*B32 + A12*B42
    #   C41 = A21*B31 + A22*B41
    #   C42 = A21*B32 + A22*B42
    #
    # Using Strassen on the 2x2 block structure:
    #   M1 = (A11+A22)(B11+B22)
    #   M2 = (A21+A22)B11
    #   M3 = A11(B12-B22)
    #   M4 = A22(B21-B11)
    #   M5 = (A11+A12)B22
    #   M6 = (A21-A11)(B11+B12)
    #   M7 = (A12-A22)(B21+B22)
    #
    # But this is for computing a 4x4 result from 4x4 * 4x4.
    # For 4x4 * 4x8, we need to handle the two 4x4 outputs separately.
    #
    # Let me just use the straightforward rank-98 construction and verify it.
    # Then try to reduce rank if possible.

    # First, let's build a correct rank-98 decomposition using Strassen.
    U, V, W = _strassen_4x4x4x8()
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback: schoolbook rank 128
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


def _strassen_2x2x2():
    """Return U, V, W matrices (each 4x7) for Strassen 2x2x2.
    Indexing: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    """
    U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
    # M1 = (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2 = (A10+A11)B00
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    # M3 = A00(B01-B11)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
    # M4 = A11(B10-B00)
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
    # M5 = (A00+A01)B11
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
    # M6 = (A10-A00)(B00+B01)
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # M7 = (A01-A11)(B10+B11)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W


def _strassen_4x4x4x8():
    """Build rank-98 decomposition for 4x4 * 4x8 -> 4x8.
    Uses Strassen 2x2x2 recursively.
    """
    n, m, p = 4, 4, 8
    rank = 98
    U = np.zeros((n*m, rank))  # 16 x 98
    V = np.zeros((m*p, rank))  # 32 x 98
    W = np.zeros((p*n, rank))  # 32 x 98
    
    U2, V2, W2 = _strassen_2x2x2()
    
    # A is 4x4, split into 4 blocks of 2x2:
    # A[0:2, 0:2] = A00, A[0:2, 2:4] = A01
    # A[2:4, 0:2] = A10, A[2:4, 2:4] = A11
    # In our indexing: A[i,j] has index i*m+j = i*4+j
    # A00: indices (0,0),(0,1),(1,0),(1,1) -> 0,1,4,5
    # A01: indices (0,2),(0,3),(1,2),(1,3) -> 2,3,6,7
    # A10: indices (2,0),(2,1),(3,0),(3,1) -> 8,9,12,13
    # A11: indices (2,2),(2,3),(3,2),(3,3) -> 10,11,14,15
    
    # B is 4x8, split into 8 blocks of 2x2:
    # B[j,k] has index j*p+k = j*8+k
    # B00: j=0, k=0,1 -> 0,1
    # B01: j=0, k=2,3 -> 2,3
    # B02: j=0, k=4,5 -> 4,5
    # B03: j=0, k=6,7 -> 6,7
    # B10: j=1, k=0,1 -> 8,9
    # B11: j=1, k=2,3 -> 10,11
    # B12: j=1, k=4,5 -> 12,13
    # B13: j=1, k=6,7 -> 14,15
    # B20: j=2, k=0,1 -> 16,17
    # B21: j=2, k=2,3 -> 18,19
    # B22: j=2, k=4,5 -> 20,21
    # B23: j=2, k=6,7 -> 22,23
    # B30: j=3, k=0,1 -> 24,25
    # B31: j=3, k=2,3 -> 26,27
    # B32: j=3, k=4,5 -> 28,29
    # B33: j=3, k=6,7 -> 30,31
    
    # C is 4x8, split into 8 blocks of 2x2:
    # C[i,k] has index k*n+i = k*4+i
    # C00: k=0, i=0,1 -> 0,1
    # C01: k=0, i=2,3 -> 2,3
    # C02: k=1, i=0,1 -> 4,5
    # C03: k=1, i=2,3 -> 6,7
    # C04: k=2, i=0,1 -> 8,9
    # C05: k=2, i=2,3 -> 10,11
    # C06: k=3, i=0,1 -> 12,13
    # C07: k=3, i=2,3 -> 14,15
    
    # For each of the 7 Strassen multiplications M_s (s=0..6):
    #   The A-side combination uses 2x2 blocks of A.
    #   The B-side combination uses 2x2 blocks of B.
    #   The C-side combination uses 2x2 blocks of C.
    #
    # But since B is 4x8, we need to handle the two halves separately.
    # Let's define the 2x2 block indices for A, B, C:
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # B blocks: B00..B33 (each 2x2, 16 total)
    # C blocks: C00..C07 (each 2x2, 8 total)
    
    # Strassen formulas for 2x2 matrix multiplication:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For our 4x4 * 4x8 problem, we split B into B_left (4x4) and B_right (4x4).
    # B_left has blocks B00,B01,B10,B11 (j=0,1; k=0,1,2,3)
    # B_right has blocks B02,B03,B12,B13 (j=0,1; k=4,5,6,7)
    # Wait, that's not right. Let me re-index.
    
    # Actually, let's just do it directly. The Strassen algorithm for 4x4 * 4x8:
    # We can think of it as two separate 4x4 * 4x4 multiplications:
    # C_left = A * B_left, C_right = A * B_right
    # where B_left = B[:, 0:4] and B_right = B[:, 4:8].
    # Each 4x4 * 4x4 uses 7 Strassen multiplications of 2x2 blocks.
    # Total rank = 7 + 7 = 14? No, each Strassen multiplication is rank 7,
    # but they share the A-side factors.
    
    # Let me just use the straightforward approach: build the rank-98 decomposition
    # by using Strassen for each of the two 4x4x4 sub-problems.
    
    # For C_left = A * B_left:
    #   A is 4x4, B_left is 4x4, C_left is 4x4.
    #   Using Strassen: 7 multiplications of 2x2x2 (rank 7 each) = rank 49.
    # For C_right = A * B_right:
    #   Similarly rank 49.
    # Total rank = 98.
    
    # Let's build this explicitly.
    
    # Define the 2x2 block indices:
    # A blocks (2x2 each):
    # A00: (0,0),(0,1),(1,0),(1,1) -> U indices 0,1,4,5
    # A01: (0,2),(0,3),(1,2),(1,3) -> U indices 2,3,6,7
    # A10: (2,0),(2,1),(3,0),(3,1) -> U indices 8,9,12,13
    # A11: (2,2),(2,3),(3,2),(3,3) -> U indices 10,11,14,15
    
    # B_left blocks (2x2 each):
    # B00: (0,0),(0,1),(1,0),(1,1) -> V indices 0,1,8,9
    # B01: (0,2),(0,3),(1,2),(1,3) -> V indices 2,3,10,11
    # B10: (2,0),(2,1),(3,0),(3,1) -> V indices 16,17,24,25
    # B11: (2,2),(2,3),(3,2),(3,3) -> V indices 18,19,26,27
    
    # B_right blocks (2x2 each):
    # B02: (0,4),(0,5),(1,4),(1,5) -> V indices 4,5,12,13
    # B03: (0,6),(0,7),(1,6),(1,7) -> V indices 6,7,14,15
    # B12: (2,4),(2,5),(3,4),(3,5) -> V indices 20,21,28,29
    # B13: (2,6),(2,7),(3,6),(3,7) -> V indices 22,23,30,31
    
    # C_left blocks (2x2 each):
    # C00: (0,0),(0,1),(1,0),(1,1) -> W indices 0,1,4,5
    # C01: (0,2),(0,3),(1,2),(1,3) -> W indices 2,3,6,7
    # C10: (2,0),(2,1),(3,0),(3,1) -> W indices 8,9,12,13
    # C11: (2,2),(2,3),(3,2),(3,3) -> W indices 10,11,14,15
    
    # C_right blocks (2x2 each):
    # C02: (0,4),(0,5),(1,4),(1,5) -> W indices 16,17,20,21
    # C03: (0,6),(0,7),(1,6),(1,7) -> W indices 18,19,22,23
    # C12: (2,4),(2,5),(3,4),(3,5) -> W indices 24,25,28,29
    # C13: (2,6),(2,7),(3,6),(3,7) -> W indices 26,27,30,31
    
    # Now, for each of the 7 Strassen multiplications, we need to:
    # 1. Form the A-side 2x2 combination.
    # 2. Form the B-side 2x2 combination.
    # 3. Multiply them (rank 7).
    # 4. Distribute the result to the appropriate C blocks.
    
    # Let's define the Strassen A, B, C combinations:
    # s=0: M1 = (A00+A11)(B00+B11)
    # s=1: M2 = (A10+A11)B00
    # s=2: M3 = A00(B01-B11)
    # s=3: M4 = A11(B10-B00)
    # s=4: M5 = (A00+A01)B11
    # s=5: M6 = (A10-A00)(B00+B01)
    # s=6: M7 = (A01-A11)(B10+B11)
    
    # C combinations:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For C_left:
    # C00_left = M1_left + M4_left - M5_left + M7_left
    # C01_left = M3_left + M5_left
    # C10_left = M2_left + M4_left
    # C11_left = M1_left - M2_left + M3_left + M6_left
    
    # For C_right:
    # C02_right = M1_right + M4_right - M5_right + M7_right
    # C03_right = M3_right + M5_right
    # C12_right = M2_right + M4_right
    # C13_right = M1_right - M2_right + M3_right + M6_right
    
    # Where M_s_left uses B_left blocks and M_s_right uses B_right blocks.
    
    # Now, let's build the U, V, W matrices.
    # For each s (0..6) and each half (left=0, right=1), we have 7 rank-1 terms.
    # Total rank = 7 * 7 * 2 = 98.
    
    # U indices for A blocks:
    A00_U = [0, 1, 4, 5]
    A01_U = [2, 3, 6, 7]
    A10_U = [8, 9, 12, 13]
    A11_U = [10, 11, 14, 15]
    
    # V indices for B_left blocks:
    B00_V = [0, 1, 8, 9]
    B01_V = [2, 3, 10, 11]
    B10_V = [16, 17, 24, 25]
    B11_V = [18, 19, 26, 27]
    
    # V indices for B_right blocks:
    B02_V = [4, 5, 12, 13]
    B03_V = [6, 7, 14, 15]
    B12_V = [20, 21, 28, 29]
    B13_V = [22, 23, 30, 31]
    
    # W indices for C_left blocks:
    C00_W = [0, 1, 4, 5]
    C01_W = [2, 3, 6, 7]
    C10_W = [8, 9, 12, 13]
    C11_W = [10, 11, 14, 15]
    
    # W indices for C_right blocks:
    C02_W = [16, 17, 20, 21]
    C03_W = [18, 19, 22, 23]
    C12_W = [24, 25, 28, 29]
    C13_W = [26, 27, 30, 31]
    
    # Strassen A combinations (as lists of (block, coeff)):
    A_combos = [
        [(A00_U, 1), (A11_U, 1)],  # s=0: A00+A11
        [(A10_U, 1), (A11_U, 1)],  # s=1: A10+A11
        [(A00_U, 1)],               # s=2: A00
        [(A11_U, 1)],               # s=3: A11
        [(A00_U, 1), (A01_U, 1)],  # s=4: A00+A01
        [(A10_U, 1), (A00_U, -1)], # s=5: A10-A00
        [(A01_U, 1), (A11_U, -1)], # s=6: A01-A11
    ]
    
    # Strassen B combinations for B_left:
    B_left_combos = [
        [(B00_V, 1), (B11_V, 1)],  # s=0: B00+B11
        [(B00_V, 1)],               # s=1: B00
        [(B01_V, 1), (B11_V, -1)], # s=2: B01-B11
        [(B10_V, 1), (B00_V, -1)], # s=3: B10-B00
        [(B11_V, 1)],               # s=4: B11
        [(B00_V, 1), (B01_V, 1)],  # s=5: B00+B01
        [(B10_V, 1), (B11_V, 1)],  # s=6: B10+B11
    ]
    
    # Strassen B combinations for B_right:
    B_right_combos = [
        [(B02_V, 1), (B13_V, 1)],  # s=0: B02+B13
        [(B02_V, 1)],               # s=1: B02
        [(B03_V, 1), (B13_V, -1)], # s=2: B03-B13
        [(B12_V, 1), (B02_V, -1)], # s=3: B12-B02
        [(B13_V, 1)],               # s=4: B13
        [(B02_V, 1), (B03_V, 1)],  # s=5: B02+B03
        [(B12_V, 1), (B13_V, 1)],  # s=6: B12+B13
    ]
    
    # C combinations:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    C_combos = [
        [(0, 1), (3, 1), (4, -1), (6, 1)],  # C00
        [(2, 1), (4, 1)],                    # C01
        [(1, 1), (3, 1)],                    # C10
        [(0, 1), (1, -1), (2, 1), (5, 1)],  # C11
    ]
    
    # Now build U, V, W.
    U2, V2, W2 = _strassen_2x2x2()
    
    for half in range(2):  # 0=left, 1=right
        B_combos = B_left_combos if half == 0 else B_right_combos
        C_blocks = [C00_W, C01_W, C10_W, C11_W] if half == 0 else [C02_W, C03_W, C12_W, C13_W]
        
        for s in range(7):
            # For each of the 7 Strassen multiplications, we have 7 rank-1 terms.
            for ms in range(7):
                col = half * 49 + s * 7 + ms
                
                # U side: A combination
                for block, coeff in A_combos[s]:
                    for r in range(4):
                        U[block[r], col] += coeff * U2[r, ms]
                
                # V side: B combination
                for block, coeff in B_combos[s]:
                    for r in range(4):
                        V[block[r], col] += coeff * V2[r, ms]
                
                # W side: C combination
                for c_block_idx, (s_idx, coeff) in enumerate(C_combos[c_block_idx]):
                    if s_idx == s:
                        C_block = C_blocks[c_block_idx]
                        for r in range(4):
                            W[C_block[r], col] += coeff * W2[r, ms]
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return U_list, V_list, W_list


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W correctly decompose the tensor."""
    try:
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1
        return bool(np.allclose(T_recon, T, atol=tol))
    except Exception:
        return False
# EVOLVE-BLOCK-END
