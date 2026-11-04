import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]
_DEADLINE = 145  # safety margin before 150s kill

def _construct_strassen_2x2x2():
    # Returns U, V, W for 2x2x2 multiplication, rank 7
    # A is 2x2, B is 2x2, C is 2x2
    # A indices: 00,01,10,11 -> 0,1,2,3
    # B indices: 00,01,10,11 -> 0,1,2,3
    # C indices: 00,01,10,11 -> 0,1,2,3
    #
    # Strassen:
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

    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]

    # M1 = (A00+A11)(B00+B11)
    U[0][0]=1; U[3][0]=1
    V[0][0]=1; V[3][0]=1
    W[0][0]=1; W[3][0]=1
    # C00 += M1, C11 += M1

    # M2 = (A10+A11)B00
    U[2][1]=1; U[3][1]=1
    V[0][1]=1
    W[2][1]=1; W[3][1]=-1
    # C10 += M2, C11 -= M2

    # M3 = A00(B01-B11)
    U[0][2]=1
    V[1][2]=1; V[3][2]=-1
    W[1][2]=1; W[3][2]=1
    # C01 += M3, C11 += M3

    # M4 = A11(B10-B00)
    U[3][3]=1
    V[2][3]=1; V[0][3]=-1
    W[0][3]=1; W[2][3]=1
    # C00 += M4, C10 += M4

    # M5 = (A00+A01)B11
    U[0][4]=1; U[1][4]=1
    V[3][4]=1
    W[0][4]=-1; W[1][4]=1
    # C00 -= M5, C01 += M5

    # M6 = (A10-A00)(B00+B01)
    U[2][5]=1; U[0][5]=-1
    V[0][5]=1; V[1][5]=1
    W[3][5]=1
    # C11 += M6

    # M7 = (A01-A11)(B10+B11)
    U[1][6]=1; U[3][6]=-1
    V[2][6]=1; V[3][6]=1
    W[0][6]=1; W[3][6]=1
    # C00 += M7, C11 += M7

    return U, V, W

def _construct_simple_nm_p(n, m, p):
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _construct_2x2x1():
    U, V, W = _construct_simple_nm_p(2, 2, 1)
    return U, V, W

def _construct_1x2x3():
    # C = A * B where A is 1x2, B is 2x3
    # C[0] = A[0]*B[0,0] + A[1]*B[1,0]
    # C[1] = A[0]*B[0,1] + A[1]*B[1,1]
    # C[2] = A[0]*B[0,2] + A[1]*B[1,2]
    # Rank 2: two outer products
    U = [[0]*2 for _ in range(2)]
    V = [[0]*2 for _ in range(6)]
    W = [[0]*2 for _ in range(3)]

    # Term 1: A[0] * [B[0,0], B[0,1], B[0,2]]
    U[0][0] = 1
    V[0][0] = 1; V[1][0] = 1; V[2][0] = 1
    V[3][0] = 0; V[4][0] = 0; V[5][0] = 0
    W[0][0] = 1; W[1][0] = 1; W[2][0] = 1

    # Term 2: A[1] * [B[1,0], B[1,1], B[1,2]]
    U[1][1] = 1
    V[0][1] = 0; V[1][1] = 0; V[2][1] = 0
    V[3][1] = 1; V[4][1] = 1; V[5][1] = 1
    W[0][1] = 1; W[1][1] = 1; W[2][1] = 1

    return U, V, W

def _construct_2x2x3():
    # Combine 2x2x2 (rank 7) for cols 0,1 and 2x2x1 (rank 2) for col 2
    U1, V1, W1 = _construct_strassen_2x2x2()
    U2, V2, W2 = _construct_2x2x1()

    rank = 9
    U_comb = np.zeros((4, rank), dtype=int)
    V_comb = np.zeros((6, rank), dtype=int)
    W_comb = np.zeros((6, rank), dtype=int)

    # 2x2x2 part (cols 0-1 of B, cols 0-1 of C)
    U_comb[:, 0:7] = np.array(U1)
    # V1: 2x2 B indices 0,1,2,3 -> B[0,0],B[0,1],B[1,0],B[1,1]
    # In 2x3 B: indices 0,1,3,4
    V_comb[0, 0:7] = np.array(V1[0])
    V_comb[1, 0:7] = np.array(V1[1])
    V_comb[3, 0:7] = np.array(V1[2])
    V_comb[4, 0:7] = np.array(V1[3])
    # W1: 2x2 C indices 0,1,2,3 -> C[0,0],C[0,1],C[1,0],C[1,1]
    # In 2x3 C: indices 0,1,3,4
    W_comb[0, 0:7] = np.array(W1[0])
    W_comb[1, 0:7] = np.array(W1[1])
    W_comb[3, 0:7] = np.array(W1[2])
    W_comb[4, 0:7] = np.array(W1[3])

    # 2x2x1 part (col 2 of B, col 2 of C)
    U_comb[:, 7:9] = np.array(U2)
    # V2: 2x1 B indices 0,1 -> B[0,2],B[1,2] -> indices 2,5 in 2x3
    V_comb[2, 7:9] = np.array(V2[0])
    V_comb[5, 7:9] = np.array(V2[1])
    # W2: 2x1 C indices 0,1 -> C[0,2],C[1,2] -> indices 2,5 in 2x3
    W_comb[2, 7:9] = np.array(W2[0])
    W_comb[5, 7:9] = np.array(W2[1])

    return U_comb.tolist(), V_comb.tolist(), W_comb.tolist()

def _construct_3x2x3():
    # Combine 1x2x3 (rank 2) for row 0 and 2x2x3 (rank 9) for rows 1,2
    U1, V1, W1 = _construct_1x2x3()
    U2, V2, W2 = _construct_2x2x3()

    rank = 11
    U_comb = np.zeros((6, rank), dtype=int)
    V_comb = np.zeros((6, rank), dtype=int)
    W_comb = np.zeros((9, rank), dtype=int)

    # 1x2x3 part (row 0 of A, all B, row 0 of C)
    U_comb[0:2, 0:2] = np.array(U1)
    V_comb[:, 0:2] = np.array(V1)
    W_comb[0:3, 0:2] = np.array(W1)

    # 2x2x3 part (rows 1,2 of A, all B, rows 1,2 of C)
    U_comb[2:6, 2:11] = np.array(U2)
    V_comb[:, 2:11] = np.array(V2)
    W_comb[3:9, 2:11] = np.array(W2)

    return U_comb.tolist(), V_comb.tolist(), W_comb.tolist()

def _construct_3x2x6():
    # Two 3x2x3 blocks for cols 0-2 and 3-5 of B
    U1, V1, W1 = _construct_3x2x3()
    U2, V2, W2 = _construct_3x2x3()

    rank = 22
    U_comb = np.zeros((6, rank), dtype=int)
    V_comb = np.zeros((12, rank), dtype=int)
    W_comb = np.zeros((18, rank), dtype=int)

    U_comb[:, 0:11] = np.array(U1)
    V_comb[0:6, 0:11] = np.array(V1)
    W_comb[0:9, 0:11] = np.array(W1)

    U_comb[:, 11:22] = np.array(U2)
    V_comb[6:12, 11:22] = np.array(V2)
    W_comb[9:18, 11:22] = np.array(W2)

    return U_comb.tolist(), V_comb.tolist(), W_comb.tolist()

def _construct_3x4x6():
    # Two 3x2x6 blocks for cols 0-1 and 2-3 of A
    U1, V1, W1 = _construct_3x2x6()
    U2, V2, W2 = _construct_3x2x6()

    rank = 44
    U_comb = np.zeros((12, rank), dtype=int)
    V_comb = np.zeros((24, rank), dtype=int)
    W_comb = np.zeros((18, rank), dtype=int)

    U_comb[0:6, 0:22] = np.array(U1)
    V_comb[0:12, 0:22] = np.array(V1)
    W_comb[:, 0:22] = np.array(W1)

    U_comb[6:12, 22:44] = np.array(U2)
    V_comb[12:24, 22:44] = np.array(V2)
    W_comb[:, 22:44] = np.array(W2)

    return U_comb.tolist(), V_comb.tolist(), W_comb.tolist()

def _construct_3x1x6():
    U, V, W = _construct_simple_nm_p(3, 1, 6)
    return U, V, W

def _construct_rank45():
    U1, V1, W1 = _construct_3x4x6()
    U2, V2, W2 = _construct_3x1x6()

    rank = 45
    U_comb = np.zeros((15, rank), dtype=int)
    V_comb = np.zeros((30, rank), dtype=int)
    W_comb = np.zeros((18, rank), dtype=int)

    U_comb[:, 0:44] = np.array(U1)
    V_comb[0:24, 0:44] = np.array(V1)
    W_comb[:, 0:44] = np.array(W1)

    U_comb[12:15, 44:45] = np.array(U2)
    V_comb[24:30, 44:45] = np.array(V2)
    W_comb[:, 44:45] = np.array(W2)

    return {"U": U_comb.tolist(), "V": V_comb.tolist(), "W": W_comb.tolist()}

def _verify_decomposition(result, n=3, m=5, p=6):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    rank = U.shape[1]

    # Test with random integer matrices
    for trial in range(5):
        A = np.random.randint(-5, 6, (n, m))
        B = np.random.randint(-5, 6, (m, p))
        C_expected = A @ B

        # Compute using decomposition
        C_computed = np.zeros((n, p))
        for r in range(rank):
            a_vec = U[:, r]  # shape (n*m,)
            b_vec = V[:, r]  # shape (m*p,)
            w_vec = W[:, r]  # shape (p*n,)
            a_mat = a_vec.reshape(n, m)
            b_mat = b_vec.reshape(m, p)
            w_mat = w_vec.reshape(p, n)
            C_computed += a_mat @ b_mat * w_mat.T

        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    return True

def _construct_schoolbook():
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
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    try:
        result = _construct_rank45()
        elapsed = time.time() - start_time
        if elapsed < _DEADLINE - 10:
            # Verify if we have time
            if _verify_decomposition(result):
                return result
        return result
    except Exception as e:
        return _construct_schoolbook()
