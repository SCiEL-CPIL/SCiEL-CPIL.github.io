"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time

def _strassen_2x2():
    """Return (U, V, W) for 2x2 x 2x2 -> 2x2, rank 7."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    # M3 = A00(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    # M4 = A11(B01-B00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    # M6 = (A10-A00)(B00+B10)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[3][5] = 1
    # M7 = (A01-A11)(B01+B11)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1
    return U, V, W

def _matvec_2x2_2x1():
    """Return (U, V, W) for 2x2 x 2x1 -> 2x1, rank 4."""
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(2)]
    W = [[0]*4 for _ in range(2)]
    # C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    U[0][0] = 1; U[1][0] = 1
    V[0][0] = 1; V[1][0] = 1
    W[0][0] = 1
    # C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1; V[1][1] = 1
    W[1][1] = 1
    # Wait, this is wrong. Let me redo.
    # For 2x2 x 2x1: C[i] = sum_j A[i,j] B[j]
    # Term (i,j): U[i*2+j, c]=1, V[j, c]=1, W[i, c]=1
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(2)]
    W = [[0]*4 for _ in range(2)]
    c = 0
    for i in range(2):
        for j in range(2):
            U[i*2+j][c] = 1
            V[j][c] = 1
            W[i][c] = 1
            c += 1
    return U, V, W

def _matvec_2x1_1x3():
    """Return (U, V, W) for 2x1 x 1x3 -> 2x3, rank 6."""
    # C[i,k] = A[i,0] * B[0,k]
    U = [[0]*6 for _ in range(2)]
    V = [[0]*6 for _ in range(3)]
    W = [[0]*6 for _ in range(6)]  # 3*2 = 6
    c = 0
    for i in range(2):
        for k in range(3):
            U[i][c] = 1
            V[k][c] = 1
            W[k*2+i][c] = 1
            c += 1
    return U, V, W

def _mul_2x2_2x3():
    """Return (U, V, W) for 2x2 x 2x3 -> 2x3, rank 11."""
    # Use Strassen for 2x2 x 2x2 (cols 0,1) + schoolbook for 2x2 x 2x1 (col 2)
    Us, Vs, Ws = _strassen_2x2()
    Um, Vm, Wm = _matvec_2x2_2x1()
    
    U = [[0]*11 for _ in range(4)]
    V = [[0]*11 for _ in range(6)]
    W = [[0]*11 for _ in range(6)]
    
    # Strassen part: 2x2 x 2x2 -> 2x2, maps to cols 0,1 of output
    for r in range(4):
        for c in range(7):
            U[r][c] = Us[r][c]
            V[r][c] = Vs[r][c]  # B rows 0,1, cols 0,1
            W[r*2+0][c] = Ws[r][c] if r < 2 else 0  # This is wrong
    # This mapping is getting complicated. Let me think more carefully.
    # 
    # For 2x2 x 2x3: A is 2x2, B is 2x3, C is 2x3.
    # Strassen handles B[:, 0:2] (2x2), producing C[:, 0:2] (2x2).
    # Schoolbook handles B[:, 2:3] (2x1), producing C[:, 2:3] (2x1).
    #
    # Global indices:
    # A: row i (0,1), col j (0,1) -> index i*2+j
    # B: row j (0,1), col k (0,1,2) -> index j*3+k
    # C: row i (0,1), col k (0,1,2) -> index k*2+i
    #
    # Strassen local: A indices 0-3, B indices 0-3 (for 2x2), C indices 0-3 (for 2x2)
    # Map Strassen B local col k to global B col k (k=0,1)
    # Map Strassen C local (i,k) to global C (i,k) for k=0,1
    
    U = [[0]*11 for _ in range(4)]
    V = [[0]*11 for _ in range(6)]
    W = [[0]*11 for _ in range(6)]
    
    for r in range(4):
        for c in range(7):
            U[r][c] = Us[r][c]
            # B local index r for Strassen: r = j*2+k where j,k in {0,1}
            # Global B index: j*3+k
            j = r // 2
            k = r % 2
            V[j*3+k][c] = Vs[r][c]
            # C local index for Strassen: r = k*2+i where k,i in {0,1}
            # Global C index: k*2+i
            i = r % 2
            kk = r // 2
            W[kk*2+i][c] = Ws[r][c]
    
    # Schoolbook for col 2: 2x2 x 2x1 -> 2x1
    for c in range(4):
        for r in range(4):
            U[r][7+c] = Um[r][c]
        for r in range(2):
            V[r*3+2][7+c] = Vm[r][c]  # B col 2
        for r in range(2):
            W[2*2+r][7+c] = Wm[r][c]  # C col 2, i.e., k=2
    
    return U, V, W

def _mul_2x3_3x3():
    """Return (U, V, W) for 2x3 x 3x3 -> 2x3, rank 17."""
    # Use 2x2 x 2x3 for rows 0,1 of B + 2x1 x 1x3 for row 2 of B
    U1, V1, W1 = _mul_2x2_2x3()  # 2x2 x 2x3, rank 11
    U2, V2, W2 = _matvec_2x1_1x3()  # 2x1 x 1x3, rank 6
    
    U = [[0]*17 for _ in range(6)]
    V = [[0]*17 for _ in range(9)]
    W = [[0]*17 for _ in range(6)]
    
    # Part 1: 2x2 x 2x3 using rows 0,1 of B
    # Local A: 2x2 (rows 0,1 of global A, cols 0,1)
    # Local B: 2x3 (rows 0,1 of global B, all cols)
    # Local C: 2x3
    for r in range(4):
        for c in range(11):
            # A local index r -> global A index: i*3+j where i = r//2, j = r%2
            i = r // 2
            j = r % 2
            U[i*3+j][c] = U1[r][c]
    for r in range(6):
        for c in range(11):
            # B local index r -> global B index: j*3+k where j = r//3, k = r%3
            j = r // 3
            k = r % 3
            V[j*3+k][c] = V1[r][c]
    for r in range(6):
        for c in range(11):
            # C local index r -> global C index: k*2+i where k = r//2, i = r%2
            k = r // 2
            i = r % 2
            W[k*2+i][c] = W1[r][c]
    
    # Part 2: 2x1 x 1x3 using row 2 of B
    for c in range(6):
        for r in range(2):
            # A local index r -> global A index: i*3+2 where i = r
            U[r*3+2][11+c] = U2[r][c]
        for r in range(3):
            # B local index r -> global B index: 2*3+r = 6+r
            V[6+r][11+c] = V2[r][c]
        for r in range(6):
            # C local index r -> global C index: k*2+i
            W[r][11+c] = W2[r][c]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # n=4, m=5, p=6
    # Split: A (4x5) into rows (2,2), cols (3,2)
    # B (5x6) into rows (3,2), cols (3,3)
    #
    # A = [[A11(2x3), A12(2x2)],
    #      [A21(2x3), A22(2x2)]]
    # B = [[B11(3x3), B12(3x3)],
    #      [B21(2x3), B22(2x3)]]
    #
    # C11(2x3) = A11*B11 + A12*B21
    # C12(2x3) = A11*B12 + A12*B22
    # C21(2x3) = A21*B11 + A22*B21
    # C22(2x3) = A21*B12 + A22*B22
    #
    # Sub-problems:
    # A11*B11: 2x3 x 3x3 -> 2x3, rank 17
    # A12*B21: 2x2 x 2x3 -> 2x3, rank 11
    # A11*B12: 2x3 x 3x3 -> 2x3, rank 17
    # A12*B22: 2x2 x 2x3 -> 2x3, rank 11
    # A21*B11: 2x3 x 3x3 -> 2x3, rank 17
    # A22*B21: 2x2 x 2x3 -> 2x3, rank 11
    # A21*B12: 2x3 x 3x3 -> 2x3, rank 17
    # A22*B22: 2x2 x 2x3 -> 2x3, rank 11
    #
    # Total rank: 4*17 + 4*11 = 68 + 44 = 112
    
    rank = 112
    
    U = [[0]*rank for _ in range(n*m)]  # 20 x 112
    V = [[0]*rank for _ in range(m*p)]  # 30 x 112
    W = [[0]*rank for _ in range(p*n)]  # 24 x 112
    
    # Global index mappings:
    # A[i,j]: i in [0,4), j in [0,5) -> index i*5+j
    # B[j,k]: j in [0,5), k in [0,6) -> index j*6+k
    # C[i,k]: i in [0,4), k in [0,6) -> index k*4+i
    
    # Block definitions:
    # A11: i in [0,2), j in [0,3)
    # A12: i in [0,2), j in [3,5)
    # A21: i in [2,4), j in [0,3)
    # A22: i in [2,4), j in [3,5)
    # B11: j in [0,3), k in [0,3)
    # B12: j in [0,3), k in [3,6)
    # B21: j in [3,5), k in [0,3)
    # B22: j in [3,5), k in [3,6)
    # C11: i in [0,2), k in [0,3)
    # C12: i in [0,2), k in [3,6)
    # C21: i in [2,4), k in [0,3)
    # C22: i in [2,4), k in [3,6)
    
    def place_2x3_3x3(U_local, V_local, W_local, a_i0, a_j0, b_j0, b_k0, c_i0, c_k0, offset):
        """Place a 2x3 x 3x3 -> 2x3 decomposition at the given block position."""
        for r in range(6):
            for c in range(17):
                li = r // 3
                lj = r % 3
                U[(a_i0+li)*5 + (a_j0+lj)][offset+c] = U_local[r][c]
        for r in range(9):
            for c in range(17):
                lj = r // 3
                lk = r % 3
                V[(b_j0+lj)*6 + (b_k0+lk)][offset+c] = V_local[r][c]
        for r in range(6):
            for c in range(17):
                lk = r // 2
                li = r % 2
                W[(c_k0+lk)*4 + (c_i0+li)][offset+c] = W_local[r][c]
    
    def place_2x2_2x3(U_local, V_local, W_local, a_i0, a_j0, b_j0, b_k0, c_i0, c_k0, offset):
        """Place a 2x2 x 2x3 -> 2x3 decomposition at the given block position."""
        for r in range(4):
            for c in range(11):
                li = r // 2
                lj = r % 2
                U[(a_i0+li)*5 + (a_j0+lj)][offset+c] = U_local[r][c]
        for r in range(6):
            for c in range(11):
                lj = r // 3
                lk = r % 3
                V[(b_j0+lj)*6 + (b_k0+lk)][offset+c] = V_local[r][c]
        for r in range(6):
            for c in range(11):
                lk = r // 2
                li = r % 2
                W[(c_k0+lk)*4 + (c_i0+li)][offset+c] = W_local[r][c]
    
    U_2x3_3x3, V_2x3_3x3, W_2x3_3x3 = _mul_2x3_3x3()
    U_2x2_2x3, V_2x2_2x3, W_2x2_2x3 = _mul_2x2_2x3()
    
    offset = 0
    # C11 = A11*B11 + A12*B21
    place_2x3_3x3(U_2x3_3x3, V_2x3_3x3, W_2x3_3x3, 0, 0, 0, 0, 0, 0, offset)  # A11*B11
    offset += 17
    place_2x2_2x3(U_2x2_2x3, V_2x2_2x3, W_2x2_2x3, 0, 3, 3, 0, 0, 0, offset)  # A12*B21
    offset += 11
    
    # C12 = A11*B12 + A12*B22
    place_2x3_3x3(U_2x3_3x3, V_2x3_3x3, W_2x3_3x3, 0, 0, 0, 3, 0, 3, offset)  # A11*B12
    offset += 17
    place_2x2_2x3(U_2x2_2x3, V_2x2_2x3, W_2x2_2x3, 0, 3, 3, 3, 0, 3, offset)  # A12*B22
    offset += 11
    
    # C21 = A21*B11 + A22*B21
    place_2x3_3x3(U_2x3_3x3, V_2x3_3x3, W_2x3_3x3, 2, 0, 0, 0, 2, 0, offset)  # A21*B11
    offset += 17
    place_2x2_2x3(U_2x2_2x3, V_2x2_2x3, W_2x2_2x3, 2, 3, 3, 0, 2, 0, offset)  # A22*B21
    offset += 11
    
    # C22 = A21*B12 + A22*B22
    place_2x3_3x3(U_2x3_3x3, V_2x3_3x3, W_2x3_3x3, 2, 0, 0, 3, 2, 3, offset)  # A21*B12
    offset += 17
    place_2x2_2x3(U_2x2_2x3, V_2x2_2x3, W_2x2_2x3, 2, 3, 3, 3, 2, 3, offset)  # A22*B22
    offset += 11
    
    assert offset == rank, f"Offset {offset} != rank {rank}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
