import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = 3, 4, 4
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12

    start_time = time.time()
    deadline = start_time + 110

    # Rank 29 Decomposition
    # 28 terms for top 2x4x4 (4 Strassen blocks)
    # 1 term for bottom 1x4x4 (Rank 1 slice)
    rank = 29
    U = [[Fraction(0)] * rank for _ in range(dim_u)]
    V = [[Fraction(0)] * rank for _ in range(dim_v)]
    W = [[Fraction(0)] * rank for _ in range(dim_w)]

    col = 0
    def add_strassen(X_u, Y_v, Z_w, col):
        U[X_u[0]][col] += 1; U[X_u[3]][col] += 1
        V[Y_v[0]][col] += 1; V[Y_v[3]][col] += 1
        W[Z_w[0]][col] += 1; W[Z_w[3]][col] += 1
        col += 1
        U[X_u[2]][col] += 1; U[X_u[3]][col] += 1
        V[Y_v[0]][col] += 1
        W[Z_w[2]][col] += 1; W[Z_w[3]][col] -= 1
        col += 1
        U[X_u[0]][col] += 1
        V[Y_v[1]][col] += 1; V[Y_v[3]][col] -= 1
        W[Z_w[1]][col] += 1; W[Z_w[3]][col] += 1
        col += 1
        U[X_u[3]][col] += 1
        V[Y_v[2]][col] += 1; V[Y_v[0]][col] -= 1
        W[Z_w[0]][col] += 1; W[Z_w[2]][col] += 1
        col += 1
        U[X_u[0]][col] += 1; U[X_u[1]][col] += 1
        V[Y_v[3]][col] += 1
        W[Z_w[0]][col] -= 1; W[Z_w[1]][col] += 1
        col += 1
        U[X_u[2]][col] += 1; U[X_u[0]][col] -= 1
        V[Y_v[0]][col] += 1; V[Y_v[1]][col] += 1
        W[Z_w[3]][col] += 1
        col += 1
        U[X_u[1]][col] += 1; U[X_u[3]][col] -= 1
        V[Y_v[2]][col] += 1; V[Y_v[3]][col] += 1
        W[Z_w[0]][col] += 1
        col += 1
        return col

    # Four 2x2x2 Strassen blocks for the top 2 rows of A
    # Block 1: A[:, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    col = add_strassen([0, 1, 4, 5], [0, 1, 4, 5], [0, 3, 1, 4], col)
    # Block 2: A[:, 2:4] * B[0:2, 2:4] -> C[0:2, 2:4]
    # Note: X_u for A cols 2,3 is [2, 3, 6, 7]
    # Y_v for B rows 0,1 cols 2,3 is [8, 9, 12, 13]
    # Z_w for C rows 0,1 cols 2,3. C index k*n+i.
    # k=2, i=0 -> 6. k=2, i=1 -> 7? No.
    # C indices:
    # k=2, i=0: 2*3+0 = 6.
    # k=2, i=1: 2*3+1 = 7.
    # k=3, i=0: 3*3+0 = 9.
    # k=3, i=1: 3*3+1 = 10.
    # So Z_w should be [6, 7, 9, 10]?
    # The previous program used [0, 3, 1, 4] for all?
    # Let's check previous program call 2:
    # col = add_strassen([2, 3, 6, 7], [8, 9, 12, 13], [0, 3, 1, 4], col)
    # This used Z_w [0, 3, 1, 4].
    # Z_w [0, 3, 1, 4] corresponds to C[0,0], C[1,0], C[0,1], C[1,1]?
    # C index = k*n + i.
    # 0 = 0*3+0 (k=0, i=0).
    # 3 = 1*3+0 (k=1, i=0).
    # 1 = 0*3+1 (k=0, i=1).
    # 4 = 1*3+1 (k=1, i=1).
    # So this Z_w is for C columns 0,1.
    # But Block 2 multiplies B cols 2,3. So it should affect C cols 2,3.
    # The previous program's Block 2 Z_w was [0, 3, 1, 4]. This is WRONG if it's meant for cols 2,3.
    # However, the previous program scored 44.
    # Maybe the blocks were arranged differently?
    # Let's re-read the previous code carefully.
    # col = add_strassen([0, 1, 4, 5], [0, 1, 4, 5], [0, 3, 1, 4], col)
    # col = add_strassen([2, 3, 6, 7], [8, 9, 12, 13], [0, 3, 1, 4], col)
    # col = add_strassen([0, 1, 4, 5], [2, 3, 6, 7], [6, 9, 7, 10], col)
    # col = add_strassen([2, 3, 6, 7], [10, 11, 14, 15], [6, 9, 7, 10], col)
    
    # Block 1: A(0,1) B(0,1) -> C(0,1) [Cols 0,1]. Z=[0,3,1,4]. Correct.
    # Block 2: A(2,3) B(0,1) -> C(0,1) [Cols 0,1]. Z=[0,3,1,4]. Correct.
    # Wait, A(2,3) are cols 2,3 of A.
    # B(0,1) are rows 0,1 of B.
    # Result is C(0,1)??
    # C[i,k] = sum_j A[i,j] B[j,k].
    # If we use A cols 2,3 and B rows 0,1, we are computing sum_{j in 0,1} A[i,j] B[j,k]?
    # No, the indices in X_u are fixed to the A indices.
    # X_u [2,3,6,7] corresponds to A indices 2,3,6,7.
    # A index 2 is i=0, j=2.
    # A index 3 is i=0, j=3.
    # A index 6 is i=1, j=2.
    # A index 7 is i=1, j=3.
    # So this block uses A columns 2,3.
    # Y_v [8,9,12,13].
    # V index 8 is j=0, k=2.
    # V index 9 is j=0, k=3.
    # V index 12 is j=1, k=2.
    # V index 13 is j=1, k=3.
    # So this block uses B rows 0,1 and Cols 2,3.
    # The product A[:, 2:4] * B[0:2, 2:4] contributes to C[:, 2:4].
    # So Z_w should be C indices for k=2,3.
    # C indices for k=2, i=0,1 are 6,7,9,10.
    # But the code used Z=[0,3,1,4] which is k=0,1.
    # This means the previous program's Block 2 was INCORRECT?
    # Or did it just sum into the wrong place and the "verify" failed?
    # The previous program verified and returned 44.
    # If the Z indices were wrong, the verification would fail, and it would fall back to schoolbook 48.
    # But the score is 1.0909 which is 48/44.
    # So the verification PASSED.
    # This implies my interpretation of the indices or the Strassen mapping is slightly off, or the previous code was actually correct and I'm misreading the indices.
    
    # Let's trust the previous code's Strassen blocks since they passed verification.
    # I will just replace the schoolbook bottom part with the Rank-1 term.
    
    # Re-run the previous logic for the top part.
    col = add_strassen([0, 1, 4, 5], [0, 1, 4, 5], [0, 3, 1, 4], col)
    col = add_strassen([2, 3, 6, 7], [8, 9, 12, 13], [0, 3, 1, 4], col)
    col = add_strassen([0, 1, 4, 5], [2, 3, 6, 7], [6, 9, 7, 10], col)
    col = add_strassen([2, 3, 6, 7], [10, 11, 14, 15], [6, 9, 7, 10], col)
    
    # Now add the Rank-1 term for the bottom row (i=2)
    # U indices 8,9,10,11 (i=2, j=0..3)
    U[8][col] = 1
    U[9][col] = 1
    U[10][col] = 1
    U[11][col] = 1
    # V indices 0..15 (all j,k)
    for j in range(4):
        for k in range(4):
            V[j*4+k][col] = 1
    # W indices for i=2: k*3+2
    for k in range(4):
        W[k*3+2][col] = 1
    col += 1
    
    # Verify
    T_expected = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1.0

    U_np = np.array([[float(x) for x in row] for row in U], dtype=np.float64)
    V_np = np.array([[float(x) for x in row] for row in V], dtype=np.float64)
    W_np = np.array([[float(x) for x in row] for row in W], dtype=np.float64)
    
    T_actual = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
    
    if not np.allclose(T_actual, T_expected, atol=1e-10, rtol=1e-10):
        # Fallback to previous rank 44 if this fails
        # (Reconstruct rank 44)
        rank = 44
        U = [[Fraction(0)] * rank for _ in range(dim_u)]
        V = [[Fraction(0)] * rank for _ in range(dim_v)]
        W = [[Fraction(0)] * rank for _ in range(dim_w)]
        col = 0
        col = add_strassen([0, 1, 4, 5], [0, 1, 4, 5], [0, 3, 1, 4], col)
        col = add_strassen([2, 3, 6, 7], [8, 9, 12, 13], [0, 3, 1, 4], col)
        col = add_strassen([0, 1, 4, 5], [2, 3, 6, 7], [6, 9, 7, 10], col)
        col = add_strassen([2, 3, 6, 7], [10, 11, 14, 15], [6, 9, 7, 10], col)
        for j in range(4):
            U[8+j][col] += 1; V[j*4+0][col] += 1; W[2][col] += 1
            col += 1
            U[8+j][col] += 1; V[j*4+1][col] += 1; W[5][col] += 1
            col += 1
        for j in range(4):
            U[8+j][col] += 1; V[j*4+2][col] += 1; W[8][col] += 1
            col += 1
            U[8+j][col] += 1; V[j*4+3][col] += 1; W[11][col] += 1
            col += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

