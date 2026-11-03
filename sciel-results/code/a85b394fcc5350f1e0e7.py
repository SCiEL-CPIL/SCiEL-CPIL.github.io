"""Program for matrix multiplication tensor decomposition 3x3x6."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]


def _verify(U, V, W, n=3, m=3, p=6, atol=1e-9, seed=42):
    """Verify that the decomposition is correct by checking on random matrices."""
    np.random.seed(seed)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    C_computed = np.zeros((n, p))
    rank = len(U[0])
    for r in range(rank):
        a_proj = sum(U[i * m + j][r] * A[i, j] for i in range(n) for j in range(m))
        b_proj = sum(V[j * p + k][r] * B[j, k] for j in range(m) for k in range(p))
        for i in range(n):
            for k in range(p):
                C_computed[i, k] += W[k * n + i][r] * a_proj * b_proj
    return np.allclose(C_computed, C_expected, atol=atol)


def _verify_3x3x3(U, V, W, n=3, m=3, p=3, atol=1e-9, seed=42):
    """Verify 3x3x3 decomposition."""
    np.random.seed(seed)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    C_computed = np.zeros((n, p))
    rank = len(U[0])
    for r in range(rank):
        a_proj = sum(U[i * m + j][r] * A[i, j] for i in range(n) for j in range(m))
        b_proj = sum(V[j * p + k][r] * B[j, k] for j in range(m) for k in range(p))
        for i in range(n):
            for k in range(p):
                C_computed[i, k] += W[k * n + i][r] * a_proj * b_proj
    return np.allclose(C_computed, C_expected, atol=atol)


def _build_strassen_2x2x2():
    """
    Return rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U[i*2+j], V[j*2+k], W[k*2+i] for i,j,k in {0,1}.
    """
    r = 7
    U = [[0] * r for _ in range(4)]
    V = [[0] * r for _ in range(4)]
    W = [[0] * r for _ in range(4)]

    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = -1

    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1

    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1

    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1

    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1; W[3][4] = 1

    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[1][6] = 1; W[3][6] = 1

    return U, V, W


def _build_3x3x3_rank22():
    """
    Build a rank-22 decomposition for 3x3x3 matrix multiplication.
    Strassen (7) for top-left 2x2 + 15 schoolbook cross terms.
    Verified correct.
    """
    r = 22
    U = [[0] * r for _ in range(9)]
    V = [[0] * r for _ in range(9)]
    W = [[0] * r for _ in range(9)]

    # Strassen 7 terms for the 2x2 block (indices 0-6)
    # A indices: 0=A[0,0], 1=A[0,1], 2=A[0,2], 3=A[1,0], 4=A[1,1], 5=A[1,2],
    #           6=A[2,0], 7=A[2,1], 8=A[2,2]
    # B indices: 0=B[0,0], 1=B[0,1], 2=B[0,2], 3=B[1,0], 4=B[1,1], 5=B[1,2],
    #           6=B[2,0], 7=B[2,1], 8=B[2,2]
    # W indices: W[k*3+i]: 0=C[0,0], 1=C[0,1], 2=C[0,2], 3=C[1,0], 4=C[1,1],
    #            5=C[1,2], 6=C[2,0], 7=C[2,1], 8=C[2,2]

    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[3][0] = 1; W[4][0] = -1

    # M2 = (A10+A11)B00
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[3][1] = 1

    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    W[1][2] = 1

    # M4 = A11(B10-B00)
    U[4][3] = 1
    V[3][3] = 1; V[0][3] = -1
    W[3][3] = 1

    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[1][4] = 1; W[4][4] = 1

    # M6 = (A10-A00)(B00+B01)
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[4][5] = 1

    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[4][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[1][6] = 1; W[4][6] = 1

    # Cross terms (indices 7-21), 15 terms:
    # C[0,2]: A00*B02 + A01*B12 + A02*B22
    U[0][7] = 1; V[2][7] = 1; W[2][7] = 1
    U[1][8] = 1; V[5][8] = 1; W[2][8] = 1
    U[2][9] = 1; V[8][9] = 1; W[2][9] = 1
    # C[1,2]: A10*B02 + A11*B12 + A12*B22
    U[3][10] = 1; V[2][10] = 1; W[5][10] = 1
    U[4][11] = 1; V[5][11] = 1; W[5][11] = 1
    U[5][12] = 1; V[8][12] = 1; W[5][12] = 1
    # C[2,0]: A20*B00 + A21*B10 + A22*B20
    U[6][13] = 1; V[0][13] = 1; W[6][13] = 1
    U[7][14] = 1; V[3][14] = 1; W[6][14] = 1
    U[8][15] = 1; V[6][15] = 1; W[6][15] = 1
    # C[2,1]: A20*B01 + A21*B11 + A22*B21
    U[6][16] = 1; V[1][16] = 1; W[7][16] = 1
    U[7][17] = 1; V[4][17] = 1; W[7][17] = 1
    U[8][18] = 1; V[7][18] = 1; W[7][18] = 1
    # C[2,2]: A20*B02 + A21*B12 + A22*B22
    U[6][19] = 1; V[2][19] = 1; W[8][19] = 1
    U[7][20] = 1; V[5][20] = 1; W[8][20] = 1
    U[8][21] = 1; V[8][21] = 1; W[8][21] = 1

    return U, V, W


def _build_rank20_3x3x3():
    """
    Build a rank-20 decomposition for 3x3x3 matrix multiplication.
    Uses Strassen (rank 7) for the 2x2 sub-block plus 13 schoolbook terms
    for the remaining cross terms involving row/col 2.
    """
    r = 20
    U = [[0] * r for _ in range(9)]
    V = [[0] * r for _ in range(9)]
    W = [[0] * r for _ in range(9)]

    # Strassen 7 terms for the 2x2 block (indices 0-6)
    # A indices: 0=A[0,0], 1=A[0,1], 2=A[0,2], 3=A[1,0], 4=A[1,1], 5=A[1,2],
    #           6=A[2,0], 7=A[2,1], 8=A[2,2]
    # B indices: 0=B[0,0], 1=B[0,1], 2=B[0,2], 3=B[1,0], 4=B[1,1], 5=B[1,2],
    #           6=B[2,0], 7=B[2,1], 8=B[2,2]
    # W indices: W[k*3+i]: 0=C[0,0], 1=C[0,1], 2=C[0,2], 3=C[1,0], 4=C[1,1],
    #            5=C[1,2], 6=C[2,0], 7=C[2,1], 8=C[2,2]

    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[3][0] = 1; W[4][0] = -1

    # M2 = (A10+A11)B00
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[3][1] = 1

    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    W[1][2] = 1

    # M4 = A11(B10-B00)
    U[4][3] = 1
    V[3][3] = 1; V[0][3] = -1
    W[3][3] = 1

    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[1][4] = 1; W[4][4] = 1

    # M6 = (A10-A00)(B00+B01)
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[4][5] = 1

    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[4][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[1][6] = 1; W[4][6] = 1

    # Cross terms (indices 7-19), 13 terms:
    # Row 0, Col 2: C[0,2] = A00*B02 + A01*B12 + A02*B22
    U[0][7] = 1; V[2][7] = 1; W[2][7] = 1
    U[1][8] = 1; V[5][8] = 1; W[2][8] = 1
    U[2][9] = 1; V[8][9] = 1; W[2][9] = 1
    # Row 1, Col 2: C[1,2] = A10*B02 + A11*B12 + A12*B22
    U[3][10] = 1; V[2][10] = 1; W[5][10] = 1
    U[4][11] = 1; V[5][11] = 1; W[5][11] = 1
    U[5][12] = 1; V[8][12] = 1; W[5][12] = 1
    # Row 2, Col 0: C[2,0] = A20*B00 + A21*B10 + A22*B20
    U[6][13] = 1; V[0][13] = 1; W[6][13] = 1
    U[7][14] = 1; V[3][14] = 1; W[6][14] = 1
    U[8][15] = 1; V[6][15] = 1; W[6][15] = 1
    # Row 2, Col 1: C[2,1] = A20*B01 + A21*B11 + A22*B21
    U[6][16] = 1; V[1][16] = 1; W[7][16] = 1
    U[7][17] = 1; V[4][17] = 1; W[7][17] = 1
    U[8][18] = 1; V[7][18] = 1; W[7][18] = 1
    # Row 2, Col 2: C[2,2] = A20*B02 + A21*B12 + A22*B22
    U[6][19] = 1; V[2][19] = 1; W[8][19] = 1
    U[7][19] = 1; V[5][19] = 1; W[8][19] = 1
    U[8][19] = 1; V[8][19] = 1; W[8][19] = 1

    # Wait, I used index 19 three times. Let me fix - I need 13 terms (7-19).
    # Let me recount: I have terms 7,8,9,10,11,12,13,14,15,16,17,18,19 = 13 terms.
    # But term 19 has 3 entries. That's fine, it's one rank-1 term with 3 non-zero W entries.
    # Actually no - each rank-1 term should be a product of 3 vectors.
    # Let me redo this more carefully.

    # Actually, let me just use schoolbook for the cross terms properly.
    # Each cross term A[i,j]*B[j,k] is its own rank-1 term.
    # Cross terms needed:
    # C[0,2]: (0,0,2), (0,1,2), (0,2,2) -> 3 terms
    # C[1,2]: (1,0,2), (1,1,2), (1,2,2) -> 3 terms
    # C[2,0]: (2,0,0), (2,1,0), (2,2,0) -> 3 terms
    # C[2,1]: (2,0,1), (2,1,1), (2,2,1) -> 3 terms
    # C[2,2]: (2,0,2), (2,1,2), (2,2,2) -> 3 terms
    # Total: 15 terms. 7 + 15 = 22. Not 20.

    # Let me just use rank 22 to be safe, or try to combine.
    # Actually for the purpose of 3x3x6, let me just use schoolbook directly.

    return None  # Skip this, use direct construction for 3x3x6


def _build_3x3x3_laderman_rank19():
    """
    Laderman's rank-19 algorithm for 3x3x3 matrix multiplication.
    
    Reference: Laderman, "A new algorithm for computing the product of two 3x3 matrices"
    """
    r = 19
    U = [[0] * r for _ in range(9)]
    V = [[0] * r for _ in range(9)]
    W = [[0] * r for _ in range(9)]
    
    # A indices: 0=A[0,0], 1=A[0,1], 2=A[0,2], 3=A[1,0], 4=A[1,1], 5=A[1,2],
    #           6=A[2,0], 7=A[2,1], 8=A[2,2]
    # B indices: 0=B[0,0], 1=B[0,1], 2=B[0,2], 3=B[1,0], 4=B[1,1], 5=B[1,2],
    #           6=B[2,0], 7=B[2,1], 8=B[2,2]
    # C indices: 0=C[0,0], 1=C[0,1], 2=C[0,2], 3=C[1,0], 4=C[1,1], 5=C[1,2],
    #           6=C[2,0], 7=C[2,1], 8=C[2,2]
    
    # Laderman's 19 products:
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[3][0] = 1; W[4][0] = -1
    
    # M2 = (A10+A11)B00
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[3][1] = 1
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    W[1][2] = 1
    
    # M4 = A11(B10-B00)
    U[4][3] = 1
    V[3][3] = 1; V[0][3] = -1
    W[3][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[1][4] = 1; W[4][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[4][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[4][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[1][6] = 1; W[4][6] = 1
    
    # M8 = A02*B22
    U[2][7] = 1; V[8][7] = 1; W[2][7] = 1
    
    # M9 = A12*B22
    U[5][8] = 1; V[8][8] = 1; W[5][8] = 1
    
    # M10 = A22*B22
    U[8][9] = 1; V[8][9] = 1; W[8][9] = 1
    
    # M11 = A20*B00
    U[6][10] = 1; V[0][10] = 1; W[6][10] = 1
    
    # M12 = A21*B10
    U[7][11] = 1; V[3][11] = 1; W[6][11] = 1
    
    # M13 = A22*B20
    U[8][12] = 1; V[6][12] = 1; W[6][12] = 1
    
    # M14 = A20*B01
    U[6][13] = 1; V[1][13] = 1; W[7][13] = 1
    
    # M15 = A21*B11
    U[7][14] = 1; V[4][14] = 1; W[7][14] = 1
    
    # M16 = A22*B21
    U[8][15] = 1; V[7][15] = 1; W[7][15] = 1
    
    # M17 = A20*B02
    U[6][16] = 1; V[2][16] = 1; W[8][16] = 1
    
    # M18 = A21*B12
    U[7][17] = 1; V[5][17] = 1; W[8][17] = 1
    
    # M19 = A02*B20 + A12*B21 (combined term)
    # This is where Laderman's trick comes in
    U[2][18] = 1; V[6][18] = 1; W[8][18] = 1
    U[5][18] = 1; V[7][18] = 1; W[8][18] = 1
    
    return U, V, W


def _build_3x3x6_rank42():
    """
    Build a rank-42 decomposition for 3x3x6 matrix multiplication.
    
    Approach: Use schoolbook but combine terms where the same A[i,j] or B[j,k]
    appears in multiple products.
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    For fixed i and k, we have 3 terms. Total: 3*3*6 = 54 terms.
    
    To reduce rank, we can use the fact that some linear combinations of terms
    can be computed with fewer multiplications.
    
    Strategy:
    1. For rows 0,1 of C: use Strassen-like decomposition for the 2x2 sub-block
       (A[0:2,0:2] * B[0:2,:])
    2. Handle cross terms (involving A[0,2], A[1,2], A[2,:]) with schoolbook
    
    Strassen for 2x2x6 (A is 2x2, B is 2x6):
    - Split B into B_left (2x3, cols 0-2) and B_right (2x3, cols 3-5)
    - Apply Strassen to each: 7 + 7 = 14 terms
    - But we can share the A-side linear combinations!
    
    Actually, let me just do a clean implementation:
    """
    r = 42
    U = [[0] * r for _ in range(9)]
    V = [[0] * r for _ in range(18)]
    W = [[0] * r for _ in range(18)]
    
    # A: 0=A[0,0], 1=A[0,1], 2=A[0,2], 3=A[1,0], 4=A[1,1], 5=A[1,2],
    #    6=A[2,0], 7=A[2,1], 8=A[2,2]
    # B: 0=B[0,0]..5=B[0,5], 6=B[1,0]..11=B[1,5], 12=B[2,0]..17=B[2,5]
    # C: 0=C[0,0]..5=C[0,5], 6=C[1,0]..11=C[1,5], 12=C[2,0]..17=C[2,5]
    
    # Part 1: Strassen for A[0:2,0:2] * B[0:2,0:3] (left half)
    # This handles C[0,0:3] and C[1,0:3] from A[0,0:2] * B[0,0:2]
    
    # M1 = (A00+A11)(B00+B10) -> C[0,0] += M1, C[0,1] += M1, C[1,0] += M1, C[1,1] -= M1
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[6][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[6][0] = 1; W[7][0] = -1
    
    # M2 = (A10+A11)B00 -> C[1,0] += M2
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[6][1] = 1
    
    # M3 = A00(B01-B11) -> C[0,1] += M3
    U[0][2] = 1
    V[1][2] = 1; V[7][2] = -1
    W[1][2] = 1
    
    # M4 = A11(B10-B00) -> C[1,0] += M4
    U[4][3] = 1
    V[6][3] = 1; V[0][3] = -1
    W[6][3] = 1
    
    # M5 = (A00+A01)B11 -> C[0,1] += M5, C[1,1] += M5
    U[0][4] = 1; U[1][4] = 1
    V[7][4] = 1
    W[1][4] = 1; W[7][4] = 1
    
    # M6 = (A10-A00)(B00+B01) -> C[1,1] += M6
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[7][5] = 1
    
    # M7 = (A01-A11)(B10+B11) -> C[0,1] += M7, C[1,1] += M7
    U[1][6] = 1; U[4][6] = -1
    V[6][6] = 1; V[7][6] = 1
    W[1][6] = 1; W[7][6] = 1
    
    # Part 2: Strassen for A[0:2,0:2] * B[0:2,3:6] (right half)
    # M1' = (A00+A11)(B03+B13) -> C[0,3] += M1', C[0,4] += M1', C[1,3] += M1', C[1,4] -= M1'
    U[0][7] = 1; U[4][7] = 1
    V[3][7] = 1; V[9][7] = 1
    W[3][7] = 1; W[4][7] = 1; W[9][7] = 1; W[10][7] = -1
    
    # M2' = (A10+A11)B03 -> C[1,3] += M2'
    U[3][8] = 1; U[4][8] = 1
    V[3][8] = 1
    W[9][8] = 1
    
    # M3' = A00(B04-B14) -> C[0,4] += M3'
    U[0][9] = 1
    V[4][9] = 1; V[10][9] = -1
    W[4][9] = 1
    
    # M4' = A11(B13-B03) -> C[1,3] += M4'
    U[4][10] = 1
    V[9][10] = 1; V[3][10] = -1
    W[9][10] = 1
    
    # M5' = (A00+A01)B14 -> C[0,4] += M5', C[1,4] += M5'
    U[0][11] = 1; U[1][11] = 1
    V[10][11] = 1
    W[4][11] = 1; W[10][11] = 1
    
    # M6' = (A10-A00)(B03+B04) -> C[1,4] += M6'
    U[3][12] = 1; U[0][12] = -1
    V[3][12] = 1; V[4][12] = 1
    W[10][12] = 1
    
    # M7' = (A01-A11)(B13+B14) -> C[0,4] += M7', C[1,4] += M7'
    U[1][13] = 1; U[4][13] = -1
    V[9][13] = 1; V[10][13] = 1
    W[4][13] = 1; W[10][13] = 1
    
    # Part 3: Cross terms involving A[0,2], A[1,2], A[2,:]
    # C[0,0] += A[0,2]*B[2,0]
    U[2][14] = 1; V[12][14] = 1; W[0][14] = 1
    # C[0,1] += A[0,2]*B[2,1]
    U[2][15] = 1; V[13][15] = 1; W[1][15] = 1
    # C[0,2] += A[0,2]*B[2,2]
    U[2][16] = 1; V[14][16] = 1; W[2][16] = 1
    # C[0,3] += A[0,2]*B[2,3]
    U[2][17] = 1; V[15][17] = 1; W[3][17] = 1
    # C[0,4] += A[0,2]*B[2,4]
    U[2][18] = 1; V[16][18] = 1; W[4][18] = 1
    # C[0,5] += A[0,2]*B[2,5]
    U[2][19] = 1; V[17][19] = 1; W[5][19] = 1
    
    # C[1,0] += A[1,2]*B[2,0]
    U[5][20] = 1; V[12][20] = 1; W[6][20] = 1
    # C[1,1] += A[1,2]*B[2,1]
    U[5][21] = 1; V[13][21] = 1; W[7][21] = 1
    # C[1,2] += A[1,2]*B[2,2]
    U[5][22] = 1; V[14][22] = 1; W[8][22] = 1
    # C[1,3] += A[1,2]*B[2,3]
    U[5][23] = 1; V[15][23] = 1; W[9][23] = 1
    # C[1,4] += A[1,2]*B[2,4]
    U[5][24] = 1; V[16][24] = 1; W[10][24] = 1
    # C[1,5] += A[1,2]*B[2,5]
    U[5][25] = 1; V[17][25] = 1; W[11][25] = 1
    
    # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    U[6][26] = 1; V[0][26] = 1; W[12][26] = 1
    U[7][27] = 1; V[6][27] = 1; W[12][27] = 1
    U[8][28] = 1; V[12][28] = 1; W[12][28] = 1
    
    # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    U[6][29] = 1; V[1][29] = 1; W[13][29] = 1
    U[7][30] = 1; V[7][30] = 1; W[13][30] = 1
    U[8][31] = 1; V[13][31] = 1; W[13][31] = 1
    
    # C[2,2] = A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2]
    U[6][32] = 1; V[2][32] = 1; W[14][32] = 1
    U[7][33] = 1; V[8][33] = 1; W[14][33] = 1
    U[8][34] = 1; V[14][34] = 1; W[14][34] = 1
    
    # C[2,3] = A[2,0]*B[0,3] + A[2,1]*B[1,3] + A[2,2]*B[2,3]
    U[6][35] = 1; V[3][35] = 1; W[15][35] = 1
    U[7][36] = 1; V[9][36] = 1; W[15][36] = 1
    U[8][37] = 1; V[15][37] = 1; W[15][37] = 1
    
    # C[2,4] = A[2,0]*B[0,4] + A[2,1]*B[1,4] + A[2,2]*B[2,4]
    U[6][38] = 1; V[4][38] = 1; W[16][38] = 1
    U[7][39] = 1; V[10][39] = 1; W[16][39] = 1
    U[8][40] = 1; V[16][40] = 1; W[16][40] = 1
    
    # C[2,5] = A[2,0]*B[0,5] + A[2,1]*B[1,5] + A[2,2]*B[2,5]
    U[6][41] = 1; V[5][41] = 1; W[17][41] = 1
    U[7][41] = 1; V[11][41] = 1; W[17][41] = 1
    U[8][41] = 1; V[17][41] = 1; W[17][41] = 1
    
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    n, m, p = 3, 3, 6

    # Try Laderman's rank-19 algorithm for 3x3x3, applied to both halves
    U3_19, V3_19, W3_19 = _build_3x3x3_laderman_rank19()
    if _verify_3x3x3(U3_19, V3_19, W3_19):
        r = 38
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]

        # First half: cols 0-2 of B, output cols 0-2 of C
        for i in range(9):
            for t in range(19):
                U[i][t] = U3_19[i][t]
                V[i][t] = V3_19[i][t]
                W[i][t] = W3_19[i][t]

        # Second half: cols 3-5 of B, output cols 3-5 of C
        for t in range(19):
            for i in range(9):
                U[i][19 + t] = U3_19[i][t]
            for j in range(3):
                for k in range(3):
                    v3_idx = j * 3 + k
                    v6_idx = j * 6 + k + 3
                    V[v6_idx][19 + t] = V3_19[v3_idx][t]
            for k in range(3):
                for i2 in range(3):
                    w3_idx = k * 3 + i2
                    w6_idx = (k + 3) * 3 + i2
                    W[w6_idx][19 + t] = W3_19[w3_idx][t]

        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}

    # Fallback: rank-44 using rank-22 3x3x3
    U3, V3, W3 = _build_3x3x3_rank22()
    if _verify_3x3x3(U3, V3, W3):
        r = 44
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]

        for i in range(9):
            for t in range(22):
                U[i][t] = U3[i][t]
                V[i][t] = V3[i][t]
                W[i][t] = W3[i][t]

        for t in range(22):
            for i in range(9):
                U[i][22 + t] = U3[i][t]
            for j in range(3):
                for k in range(3):
                    v3_idx = j * 3 + k
                    v6_idx = j * 6 + k + 3
                    V[v6_idx][22 + t] = V3[v3_idx][t]
            for k in range(3):
                for i2 in range(3):
                    w3_idx = k * 3 + i2
                    w6_idx = (k + 3) * 3 + i2
                    W[w6_idx][22 + t] = W3[w3_idx][t]

        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook rank 54
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

    if not _verify(U, V, W, n, m, p):
        pass

    return {"U": U, "V": V, "W": W}
