import time
import numpy as np

def construct():
    """
    Rank-33 decomposition for 2x5 * 5x5 matrix multiplication.
    Uses Strassen (rank 7) for the four 2x2x2 sub-blocks and
    rank-4 (direct) for the eight 2x2x1 / 2x1x2 mixed blocks.
    """
    rank = 33
    U = [[0] * rank for _ in range(10)]   # 2*5
    V = [[0] * rank for _ in range(25)]   # 5*5
    W = [[0] * rank for _ in range(10)]   # 5*2, trace convention W[k*2+i]
    col = 0

    def strassen(A00, A01, A10, A11, B00, B01, B10, B11,
                 C00, C01, C10, C11):
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        U[A00][col] += 1; U[A11][col] += 1
        V[B00][col] += 1; V[B11][col] += 1
        W[C00][col] += 1; W[C11][col] += 1
        col += 1
        # M2 = (A10+A11)B00
        U[A10][col] += 1; U[A11][col] += 1
        V[B00][col] += 1
        W[C10][col] += 1; W[C11][col] += -1
        col += 1
        # M3 = A00(B01-B11)
        U[A00][col] += 1
        V[B01][col] += 1; V[B11][col] += -1
        W[C01][col] += 1; W[C11][col] += 1
        col += 1
        # M4 = A11(B10-B00)
        U[A11][col] += 1
        V[B10][col] += 1; V[B00][col] += -1
        W[C00][col] += 1; W[C10][col] += 1
        col += 1
        # M5 = (A00+A01)B11
        U[A00][col] += 1; U[A01][col] += 1
        V[B11][col] += 1
        W[C00][col] += -1; W[C01][col] += 1
        col += 1
        # M6 = (A10-A00)(B00+B01)
        U[A10][col] += 1; U[A00][col] += -1
        V[B00][col] += 1; V[B01][col] += 1
        W[C11][col] += 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[A01][col] += 1; U[A11][col] += -1
        V[B10][col] += 1; V[B11][col] += 1
        W[C00][col] += 1
        col += 1

    def rank4_2x2x1(A00, A01, A10, A11, B0, B1, C0, C1):
        """C0 = A00*B0 + A01*B1, C1 = A10*B0 + A11*B1"""
        nonlocal col
        U[A00][col] += 1; V[B0][col] += 1; W[C0][col] += 1; col += 1
        U[A01][col] += 1; V[B1][col] += 1; W[C0][col] += 1; col += 1
        U[A10][col] += 1; V[B0][col] += 1; W[C1][col] += 1; col += 1
        U[A11][col] += 1; V[B1][col] += 1; W[C1][col] += 1; col += 1

    def rank4_2x1x2(A0, A1, B0, B1, C00, C01, C10, C11):
        """C[i,k] = A[i]*B[k]"""
        nonlocal col
        U[A0][col] += 1; V[B0][col] += 1; W[C00][col] += 1; col += 1
        U[A0][col] += 1; V[B1][col] += 1; W[C01][col] += 1; col += 1
        U[A1][col] += 1; V[B0][col] += 1; W[C10][col] += 1; col += 1
        U[A1][col] += 1; V[B1][col] += 1; W[C11][col] += 1; col += 1

    def u_idx(i, j): return i * 5 + j
    def v_idx(j, k): return j * 5 + k
    def w_idx(i, k): return k * 2 + i

    # Group output columns {0,1}
    strassen(
        u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1),
        v_idx(0,0), v_idx(0,1), v_idx(1,0), v_idx(1,1),
        w_idx(0,0), w_idx(0,1), w_idx(1,0), w_idx(1,1)
    )
    strassen(
        u_idx(0,2), u_idx(0,3), u_idx(1,2), u_idx(1,3),
        v_idx(2,0), v_idx(2,1), v_idx(3,0), v_idx(3,1),
        w_idx(0,0), w_idx(0,1), w_idx(1,0), w_idx(1,1)
    )
    rank4_2x1x2(
        u_idx(0,4), u_idx(1,4),
        v_idx(4,0), v_idx(4,1),
        w_idx(0,0), w_idx(0,1), w_idx(1,0), w_idx(1,1)
    )

    # Group output columns {2,3}
    strassen(
        u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1),
        v_idx(0,2), v_idx(0,3), v_idx(1,2), v_idx(1,3),
        w_idx(0,2), w_idx(0,3), w_idx(1,2), w_idx(1,3)
    )
    strassen(
        u_idx(0,2), u_idx(0,3), u_idx(1,2), u_idx(1,3),
        v_idx(2,2), v_idx(2,3), v_idx(3,2), v_idx(3,3),
        w_idx(0,2), w_idx(0,3), w_idx(1,2), w_idx(1,3)
    )
    rank4_2x1x2(
        u_idx(0,4), u_idx(1,4),
        v_idx(4,2), v_idx(4,3),
        w_idx(0,2), w_idx(0,3), w_idx(1,2), w_idx(1,3)
    )

    # Group output column {4}
    rank4_2x2x1(
        u_idx(0,0), u_idx(0,1), u_idx(1,0), u_idx(1,1),
        v_idx(0,4), v_idx(1,4),
        w_idx(0,4), w_idx(1,4)
    )
    rank4_2x2x1(
        u_idx(0,2), u_idx(0,3), u_idx(1,2), u_idx(1,3),
        v_idx(2,4), v_idx(3,4),
        w_idx(0,4), w_idx(1,4)
    )
    rank4_2x2x1(
        u_idx(0,4), u_idx(0,4), u_idx(1,4), u_idx(1,4),
        v_idx(4,4), v_idx(4,4),
        w_idx(0,4), w_idx(1,4)
    )

    assert col == rank, f"Expected rank {rank}, got {col}"
    return {"U": U, "V": V, "W": W}
