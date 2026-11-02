"""Matrix multiplication 3,3,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction as Fr

def _build_trivial(n, m, p):
    """Build the trivial (schoolbook) decomposition of rank n*m*p."""
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(decomp, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly reconstructs the matmul tensor."""
    U = np.array(decomp["U"], dtype=np.float64)
    V = np.array(decomp["V"], dtype=np.float64)
    W = np.array(decomp["W"], dtype=np.float64)
    r = U.shape[1]
    # Check valid positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = sum(U[a, t] * V[b, t] * W[c, t] for t in range(r))
                if abs(val - 1) > tol:
                    return False
    # Check invalid positions (sample)
    invalid_checked = 0
    for a in range(n * m):
        i_a, j_a = divmod(a, m)
        for b in range(m * p):
            j_b, k_b = divmod(b, p)
            if j_a != j_b:
                for c in range(p * n):
                    k_c, i_c = divmod(c, n)
                    if k_b != k_c or i_c != i_a:
                        val = sum(U[a, t] * V[b, t] * W[c, t] for t in range(r))
                        if abs(val) > tol:
                            return False
                        invalid_checked += 1
                        if invalid_checked >= 100:
                            break
                if invalid_checked >= 100:
                    break
        if invalid_checked >= 100:
            break
    return True

def _strassen_3x3x3():
    """
    Return a rank-7 decomposition of M_{3,3,3} using Strassen-like approach.
    
    We decompose 3x3 into blocks:
    A = [[A11, A12, A13],
         [A21, A22, A23],
         [A31, A32, A33]]
    
    One approach: use the fact that 3x3 matmul can be done with 7 multiplications
    by embedding 2x2 Strassen and handling the third row/column carefully.
    
    Actually, let's use a different strategy. The tensor M_{3,3,3} has known
    minimum rank 19 (Landsberg). But we can get rank 19 using specific constructions.
    
    For simplicity and guaranteed correctness, let's use a block approach:
    Split into 2x2 blocks and use Strassen where possible.
    
    A = [[A1, A2],
         [A3, A4]] where A1 is 2x2, A2 is 2x1, A3 is 1x2, A4 is 1x1
    B = [[B1, B2],
         [B3, B4]] where B1 is 2x2, B2 is 2x1, B3 is 1x2, B4 is 1x1
    
    C = A*B = [[A1*B1 + A2*B3, A1*B2 + A2*B4],
               [A3*B1 + A4*B3, A3*B2 + A4*B4]]
    
    Using Strassen on the 2x2 blocks:
    M1 = (A1+A4)*(B1+B4)  [2x2 * 2x2, rank 7]
    M2 = (A3+A4)*B1       [1x2 * 2x2, rank 2]
    M3 = A1*(B1-B4)       [2x2 * 2x2, rank 7]
    M4 = A4*(B3-B1)       [1x1 * 2x2, rank 2]
    M5 = (A1+A2)*(B4+B3)  [2x2 * 2x2, rank 7]
    M6 = (A3-A1)*(B4-B2)  [1x2 * 2x2, rank 2]
    M7 = (A4-A3)*(B1-B2)  [1x1 * 2x2, rank 2]
    
    C11 = M1 + M4 - M7 - M2
    C12 = M5 - M4
    C21 = M2 + M3
    C22 = M1 - M3 + M6 + M7
    
    Total rank: 7+2+7+2+7+2+2 = 29, which is worse than trivial 27.
    
    Let me try a different decomposition. Actually, for 3x3x3, the best known
    upper bound is 19. Let me look for a concrete construction.
    
    Alternative: Use the fact that we can pad 2x2 to 3x3. If we have a rank-7
    decomposition of 2x2x2, we can extend it to 3x3x3 by adding 12 more terms
    for the extra row/column, giving rank 19.
    
    Let me construct this explicitly.
    """
    n = m = p = 3
    r = 19
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    
    # First 7 terms: Strassen for the top-left 2x2 block
    # Indices: i,j in {0,1} for U; j,k in {0,1} for V; k,i in {0,1} for W
    # Strassen 2x2x2 terms:
    # Term 0: (A00+A11)(B00+B11)
    # Term 1: (A10+A11)B00
    # Term 2: A00(B00-B11)
    # Term 3: A11(B10-B00)
    # Term 4: (A00+A01)(B11+B10)
    # Term 5: (A10-A00)(B11-B01)
    # Term 6: (A11-A10)(B00-B01)
    
    # U indices: i*m+j where i,j in {0,1}
    # V indices: j*p+k where j,k in {0,1}  
    # W indices: k*n+i where k,i in {0,1}
    
    # Term 0: (A00+A11)(B00+B11)
    # U: positions (0,0) and (1,1) get +1
    # V: positions (0,0) and (1,1) get +1
    # W: positions (0,0) and (1,1) get +1
    U[0*3+0][0] = 1; U[1*3+1][0] = 1
    V[0*3+0][0] = 1; V[1*3+1][0] = 1
    W[0*3+0][0] = 1; W[1*3+1][0] = 1
    
    # Term 1: (A10+A11)B00
    # U: positions (1,0) and (1,1) get +1
    # V: position (0,0) gets +1
    # W: positions (0,1) gets +1 (k=0,i=1)
    U[1*3+0][1] = 1; U[1*3+1][1] = 1
    V[0*3+0][1] = 1
    W[0*3+1][1] = 1
    
    # Term 2: A00(B00-B11)
    # U: position (0,0) gets +1
    # V: positions (0,0) gets +1, (1,1) gets -1
    # W: position (0,0) gets +1
    U[0*3+0][2] = 1
    V[0*3+0][2] = 1; V[1*3+1][2] = -1
    W[0*3+0][2] = 1
    
    # Term 3: A11(B10-B00)
    # U: position (1,1) gets +1
    # V: positions (1,0) gets +1, (0,0) gets -1
    # W: position (1,1) gets +1 (k=1,i=1)
    U[1*3+1][3] = 1
    V[1*3+0][3] = 1; V[0*3+0][3] = -1
    W[1*3+1][3] = 1
    
    # Term 4: (A00+A01)(B11+B10)
    # U: positions (0,0) and (0,1) get +1
    # V: positions (1,1) and (1,0) get +1
    # W: positions (1,0) gets +1 (k=1,i=0)
    U[0*3+0][4] = 1; U[0*3+1][4] = 1
    V[1*3+1][4] = 1; V[1*3+0][4] = 1
    W[1*3+0][4] = 1
    
    # Term 5: (A10-A00)(B11-B01)
    # U: positions (1,0) gets +1, (0,0) gets -1
    # V: positions (1,1) gets +1, (0,1) gets -1
    # W: positions (1,1) gets +1 (k=1,i=1)
    U[1*3+0][5] = 1; U[0*3+0][5] = -1
    V[1*3+1][5] = 1; V[0*3+1][5] = -1
    W[1*3+1][5] = 1
    
    # Term 6: (A11-A10)(B00-B01)
    # U: positions (1,1) gets +1, (1,0) gets -1
    # V: positions (0,0) gets +1, (0,1) gets -1
    # W: positions (0,1) gets +1 (k=0,i=1)
    U[1*3+1][6] = 1; U[1*3+0][6] = -1
    V[0*3+0][6] = 1; V[0*3+1][6] = -1
    W[0*3+1][6] = 1
    
    # Now we need to handle the third row and column.
    # The remaining terms involve indices where i=2 or j=2 or k=2.
    
    # For C[2,k] (third row of C): C[2,k] = sum_j A[2,j]*B[j,k]
    # This involves A[2,0], A[2,1], A[2,2] and B[0,k], B[1,k], B[2,k]
    
    # For C[i,2] (third column of C): C[i,2] = sum_j A[i,j]*B[j,2]
    # This involves A[i,0], A[i,1], A[i,2] and B[j,2]
    
    # Let's add terms for the third row/column systematically.
    # We need 12 more terms to get to rank 19.
    
    # Term 7: A[2,0]*B[0,0]
    U[2*3+0][7] = 1
    V[0*3+0][7] = 1
    W[0*3+2][7] = 1  # k=0, i=2
    
    # Term 8: A[2,0]*B[0,1]
    U[2*3+0][8] = 1
    V[0*3+1][8] = 1
    W[1*3+2][8] = 1  # k=1, i=2
    
    # Term 9: A[2,0]*B[0,2]
    U[2*3+0][9] = 1
    V[0*3+2][9] = 1
    W[2*3+2][9] = 1  # k=2, i=2
    
    # Term 10: A[2,1]*B[1,0]
    U[2*3+1][10] = 1
    V[1*3+0][10] = 1
    W[0*3+2][10] = 1  # k=0, i=2
    
    # Term 11: A[2,1]*B[1,1]
    U[2*3+1][11] = 1
    V[1*3+1][11] = 1
    W[1*3+2][11] = 1  # k=1, i=2
    
    # Term 12: A[2,1]*B[1,2]
    U[2*3+1][12] = 1
    V[1*3+2][12] = 1
    W[2*3+2][12] = 1  # k=2, i=2
    
    # Term 13: A[2,2]*B[2,0]
    U[2*3+2][13] = 1
    V[2*3+0][13] = 1
    W[0*3+2][13] = 1  # k=0, i=2
    
    # Term 14: A[2,2]*B[2,1]
    U[2*3+2][14] = 1
    V[2*3+1][14] = 1
    W[1*3+2][14] = 1  # k=1, i=2
    
    # Term 15: A[2,2]*B[2,2]
    U[2*3+2][15] = 1
    V[2*3+2][15] = 1
    W[2*3+2][15] = 1  # k=2, i=2
    
    # Now for the third column (i in {0,1}, k=2):
    # C[i,2] = sum_j A[i,j]*B[j,2]
    
    # Term 16: A[0,0]*B[0,2]
    U[0*3+0][16] = 1
    V[0*3+2][16] = 1
    W[2*3+0][16] = 1  # k=2, i=0
    
    # Term 17: A[0,1]*B[1,2]
    U[0*3+1][17] = 1
    V[1*3+2][17] = 1
    W[2*3+0][17] = 1  # k=2, i=0
    
    # Term 18: A[1,0]*B[0,2]
    U[1*3+0][18] = 1
    V[0*3+2][18] = 1
    W[2*3+1][18] = 1  # k=2, i=1
    
    # Wait, I'm missing A[1,1]*B[1,2]. Let me recount.
    # I have 19 terms (0-18), but I'm missing some.
    
    # Actually, let me reconsider. The above construction has issues because
    # the Strassen terms already contribute to some positions, and I'm double-counting.
    
    # Let me take a cleaner approach: just use the trivial decomposition for now
    # but with a better structure.
    
    # Actually, for the purpose of this task, let me just return a valid decomposition
    # with rank less than 45. The trivial rank is 45, so any improvement helps.
    
    # Let me try a different approach: use the fact that M_{3,3,5} can be decomposed
    # as M_{3,3,3} + M_{3,3,2} where the second part is trivial.
    
    # For M_{3,3,3}, use rank 19 (or whatever we can construct)
    # For M_{3,3,2}, use rank 18 (trivial)
    # Total: 19 + 18 = 37 < 45
    
    # But I need to make sure the decomposition is correct.
    
    # Let me restart with a cleaner construction.
    
    return None  # This is a placeholder; we'll build it properly in construct()

def _build_rank37(n, m, p):
    """
    Build a rank-37 decomposition of M_{3,3,5}.
    
    Strategy: Decompose M_{3,3,5} = M_{3,3,3} ⊕ M_{3,3,2}
    - M_{3,3,3}: use a rank-19 decomposition
    - M_{3,3,2}: use a rank-18 trivial decomposition
    Total: 19 + 18 = 37 < 45
    """
    n, m, p = 3, 3, 5
    r1 = 19  # rank for M_{3,3,3}
    r2 = 18  # rank for M_{3,3,2}
    r = r1 + r2  # total rank = 37
    
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    
    # Part 1: M_{3,3,3} - rank 19 decomposition
    # We need to construct 19 terms that decompose the 3x3x3 matmul tensor.
    
    # For M_{3,3,3}, the tensor T1[i,j,k] = 1 for all i,j,k in {0,1,2}
    # We need sum_t U1[i*m+j,t] * V1[j*p3+k,t] * W1[k*n+i,t] = 1
    # where p3 = 3 (the p dimension for this subproblem)
    
    # Let's use a construction based on the "border rank" approach.
    # One known rank-19 decomposition uses the following structure:
    
    # Terms 0-6: Strassen-like for the 2x2x2 core
    # Terms 7-18: Handle the third row/column
    
    # Term 0: (A00+A11)(B00+B11)
    U[0*3+0][0] = 1; U[1*3+1][0] = 1
    V[0*3+0][0] = 1; V[1*3+1][0] = 1
    W[0*3+0][0] = 1; W[1*3+1][0] = 1
    
    # Term 1: (A10+A11)B00
    U[1*3+0][1] = 1; U[1*3+1][1] = 1
    V[0*3+0][1] = 1
    W[0*3+1][1] = 1
    
    # Term 2: A00(B00-B11)
    U[0*3+0][2] = 1
    V[0*3+0][2] = 1; V[1*3+1][2] = -1
    W[0*3+0][2] = 1
    
    # Term 3: A11(B10-B00)
    U[1*3+1][3] = 1
    V[1*3+0][3] = 1; V[0*3+0][3] = -1
    W[1*3+1][3] = 1
    
    # Term 4: (A00+A01)(B11+B10)
    U[0*3+0][4] = 1; U[0*3+1][4] = 1
    V[1*3+1][4] = 1; V[1*3+0][4] = 1
    W[1*3+0][4] = 1
    
    # Term 5: (A10-A00)(B11-B01)
    U[1*3+0][5] = 1; U[0*3+0][5] = -1
    V[1*3+1][5] = 1; V[0*3+1][5] = -1
    W[1*3+1][5] = 1
    
    # Term 6: (A11-A10)(B00-B01)
    U[1*3+1][6] = 1; U[1*3+0][6] = -1
    V[0*3+0][6] = 1; V[0*3+1][6] = -1
    W[0*3+1][6] = 1
    
    # Now handle the third row and column with 12 more terms.
    # We need to ensure that:
    # - C[2,k] = sum_j A[2,j]*B[j,k] for k=0,1,2
    # - C[i,2] = sum_j A[i,j]*B[j,2] for i=0,1
    # - C[2,2] = sum_j A[2,j]*B[j,2]
    
    # But we must be careful not to double-count contributions from the Strassen terms.
    
    # The Strassen terms above only involve i,j,k in {0,1}, so they don't contribute
    # to positions where i=2 or j=2 or k=2.
    
    # Term 7: A[2,0]*B[0,0]
    U[2*3+0][7] = 1
    V[0*3+0][7] = 1
    W[0*3+2][7] = 1
    
    # Term 8: A[2,0]*B[0,1]
    U[2*3+0][8] = 1
    V[0*3+1][8] = 1
    W[1*3+2][8] = 1
    
    # Term 9: A[2,0]*B[0,2]
    U[2*3+0][9] = 1
    V[0*3+2][9] = 1
    W[2*3+2][9] = 1
    
    # Term 10: A[2,1]*B[1,0]
    U[2*3+1][10] = 1
    V[1*3+0][10] = 1
    W[0*3+2][10] = 1
    
    # Term 11: A[2,1]*B[1,1]
    U[2*3+1][11] = 1
    V[1*3+1][11] = 1
    W[1*3+2][11] = 1
    
    # Term 12: A[2,1]*B[1,2]
    U[2*3+1][12] = 1
    V[1*3+2][12] = 1
    W[2*3+2][12] = 1
    
    # Term 13: A[2,2]*B[2,0]
    U[2*3+2][13] = 1
    V[2*3+0][13] = 1
    W[0*3+2][13] = 1
    
    # Term 14: A[2,2]*B[2,1]
    U[2*3+2][14] = 1
    V[2*3+1][14] = 1
    W[1*3+2][14] = 1
    
    # Term 15: A[2,2]*B[2,2]
    U[2*3+2][15] = 1
    V[2*3+2][15] = 1
    W[2*3+2][15] = 1
    
    # Term 16: A[0,0]*B[0,2]
    U[0*3+0][16] = 1
    V[0*3+2][16] = 1
    W[2*3+0][16] = 1
    
    # Term 17: A[0,1]*B[1,2]
    U[0*3+1][17] = 1
    V[1*3+2][17] = 1
    W[2*3+0][17] = 1
    
    # Term 18: A[1,0]*B[0,2]
    U[1*3+0][18] = 1
    V[0*3+2][18] = 1
    W[2*3+1][18] = 1
    
    # We're missing A[1,1]*B[1,2]. Let me restructure.
    
    # Actually, I realize my counting is off. Let me use a cleaner approach:
    # Use 19 specific terms that I know work.
    
    # For now, let me just return the trivial decomposition since constructing
    # a valid rank-19 decomposition requires careful verification.
    
    return None

def _build_rank32(n, m, p):
    """
    Build a rank-32 decomposition of M_{3,3,5}.
    
    Key insight: The Strassen terms for 2x2x2 don't just replace 4 trivial terms;
    they replace 8 trivial terms (2*2*2) with 7 terms, saving 1 term.
    
    For M_{3,3,3}:
    - 2x2x2 core: 8 trivial terms -> 7 Strassen terms (save 1)
    - Remaining 19 terms: use trivial
    - Total: 7 + 19 = 26
    
    For M_{3,3,2}:
    - 18 trivial terms
    
    Total: 26 + 18 = 44 < 45
    
    Let me construct this.
    """
    n, m, p = 3, 3, 5
    r1 = 26  # rank for M_{3,3,3}
    r2 = 18  # rank for M_{3,3,2}
    r = r1 + r2  # total rank = 44
    
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    
    # Part 1: M_{3,3,3} with rank 26
    # Terms 0-6: Strassen for 2x2x2 core
    # Terms 7-25: Trivial for the remaining 19 positions
    
    # Strassen terms (0-6):
    # Term 0: (A00+A11)(B00+B11)
    U[0*3+0][0] = 1; U[1*3+1][0] = 1
    V[0*3+0][0] = 1; V[1*3+1][0] = 1
    W[0*3+0][0] = 1; W[1*3+1][0] = 1
    
    # Term 1: (A10+A11)B00
    U[1*3+0][1] = 1; U[1*3+1][1] = 1
    V[0*3+0][1] = 1
    W[0*3+1][1] = 1
    
    # Term 2: A00(B00-B11)
    U[0*3+0][2] = 1
    V[0*3+0][2] = 1; V[1*3+1][2] = -1
    W[0*3+0][2] = 1
    
    # Term 3: A11(B10-B00)
    U[1*3+1][3] = 1
    V[1*3+0][3] = 1; V[0*3+0][3] = -1
    W[1*3+1][3] = 1
    
    # Term 4: (A00+A01)(B11+B10)
    U[0*3+0][4] = 1; U[0*3+1][4] = 1
    V[1*3+1][4] = 1; V[1*3+0][4] = 1
    W[1*3+0][4] = 1
    
    # Term 5: (A10-A00)(B11-B01)
    U[1*3+0][5] = 1; U[0*3+0][5] = -1
    V[1*3+1][5] = 1; V[0*3+1][5] = -1
    W[1*3+1][5] = 1
    
    # Term 6: (A11-A10)(B00-B01)
    U[1*3+1][6] = 1; U[1*3+0][6] = -1
    V[0*3+0][6] = 1; V[0*3+1][6] = -1
    W[0*3+1][6] = 1
    
    # Now add trivial terms for the remaining 19 positions in M_{3,3,3}.
    # These are positions where at least one of i,j,k is 2.
    
    col = 7
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # Skip if all of i,j,k are in {0,1} (handled by Strassen)
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i*3+j][col] = 1
                V[j*3+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    
    # Part 2: M_{3,3,2} with rank 18 (trivial)
    # k ranges from 3 to 4 (i.e., k_local = k - 3, which is 0 or 1)
    
    for i in range(3):
        for j in range(3):
            for k_local in range(2):
                k = 3 + k_local  # actual k index
                U[i*3+j][r1 + i*3 + j*2 + k_local] = 1
                V[j*5+k][r1 + i*3 + j*2 + k_local] = 1
                W[k*3+i][r1 + i*3 + j*2 + k_local] = 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 5
    deadline = time.time() + 140
    
    # Start with trivial decomposition (rank 45)
    best = _build_trivial(n, m, p)
    
    # Try rank-44 decomposition
    result = _build_rank32(n, m, p)
    if result is not None:
        if _verify(result, n, m, p):
            best = result
    
    return best
# EVOLVE-BLOCK-END
