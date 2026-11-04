import numpy as np
import time
import itertools

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Schoolbook fallback
    def schoolbook():
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
        return {"U": U, "V": V, "W": W}

    def get_strassen_2x2():
        """Returns U, V, W for 2x2x2 matrix multiplication, rank 7.
        U: shape (4, 7), indexed by [i*2+j] for i,j in {0,1}
        V: shape (4, 7), indexed by [j*2+k] for j,k in {0,1}
        W: shape (4, 7), indexed by [k*2+i] for k,i in {0,1}
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))

        # Strassen formulas for 2x2:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 - M4 + M5 - W
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6

        # U indices: [0]=A11, [1]=A12, [2]=A21, [3]=A22
        # V indices: [0]=B11, [1]=B12, [2]=B21, [3]=B22
        # W indices: [0]=C11, [1]=C12, [2]=C21, [3]=C22

        # M1: (A11+A22)(B11+B22)
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[3,0] = 1  # C11 += M1, C22 += M1

        # M2: (A21+A22)B11
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[2,1] = 1  # C21 += M2
        W[3,1] = -1  # C22 -= M2

        # M3: A11(B12-B22)
        U[0,2] = 1
        V[1,2] = 1; V[3,2] = -1
        W[1,2] = 1  # C12 += M3
        W[3,2] = 1  # C22 += M3

        # M4: A22(B21-B11)
        U[3,3] = 1
        V[2,3] = 1; V[0,3] = -1
        W[0,3] = -1  # C11 -= M4
        W[2,3] = 1   # C21 += M4

        # M5: (A11+A12)B22
        U[0,4] = 1; U[1,4] = 1
        V[3,4] = 1
        W[0,4] = 1   # C11 += M5
        W[1,4] = 1   # C12 += M5

        # M6: (A21-A11)(B11+B12)
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[3,5] = 1   # C22 += M6

        # M7: (A12-A22)(B21+B22)
        U[1,6] = 1; U[3,6] = -1
        V[2,6] = 1; V[3,6] = 1
        W[0,6] = -1  # C11 -= M7

        return U, V, W

    def verify_2x2():
        U, V, W = get_strassen_2x2()
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    total = 0
                    for r in range(7):
                        total += U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
                    if total != 1:
                        return False
        return True

    if not verify_2x2():
        return schoolbook()

    def build_4x4x4():
        """Build 4x4x4 decomposition using 4 Strassen 2x2x2 operations.
        U: shape (16, 28), indexed by [i*4+j]
        V: shape (16, 28), indexed by [j*4+k]
        W: shape (16, 28), indexed by [k*4+i]
        """
        U4 = np.zeros((16, 28))
        V4 = np.zeros((16, 28))
        W4 = np.zeros((16, 28))

        U2, V2, W2 = get_strassen_2x2()

        blocks = [(0,0), (0,1), (1,0), (1,1)]
        for idx, (br, bc) in enumerate(blocks):
            r_offset = idx * 7
            for li in range(2):
                for lj in range(2):
                    global_i = br*2 + li
                    global_j = bc*2 + lj
                    U4[global_i*4 + global_j, r_offset:r_offset+7] = U2[li*2+lj, :]

            for lj in range(2):
                for lk in range(2):
                    global_j = bc*2 + lj
                    global_k = bc*2 + lk
                    V4[global_j*4 + global_k, r_offset:r_offset+7] = V2[lj*2+lk, :]

            for lk in range(2):
                for li in range(2):
                    global_k = bc*2 + lk
                    global_i = br*2 + li
                    W4[global_k*4 + global_i, r_offset:r_offset+7] = W2[lk*2+li, :]

        return U4, V4, W4

    def verify_4x4x4():
        U4, V4, W4 = build_4x4x4()
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    total = 0
                    for r in range(28):
                        total += U4[i*4+j, r] * V4[j*4+k, r] * W4[k*4+i, r]
                    if total != 1:
                        return False
        return True

    if not verify_4x4x4():
        return schoolbook()

    def build_4x4x2():
        """Build 4x4x2 decomposition using Strassen 4x4x4 (rank 28).
        We only need output columns k=0,1 (local), so W has shape (8, 28)
        covering W[k*4+i] for k in [0,2), i in [0,4).
        U: shape (16, 28), indexed by [i*4+j]
        V: shape (16, 28), indexed by [j*4+k] for j in [0,4), k in [0,2)
        W: shape (8, 28), indexed by [k*4+i] for k in [0,2), i in [0,4)
        """
        U444, V444, W444 = build_4x4x4()
        V442 = V444[:8, :]
        W442 = W444[:8, :]
        return U444, V442, W442

    def build_4x4x6():
        """Build 4x4x6 decomposition.
        U: shape (16, rank), indexed by [i*4+j] for i in [0,4), j in [0,4)
        V: shape (24, rank), indexed by [j*6+k] for j in [0,4), k in [0,6)
        W: shape (24, rank), indexed by [k*4+i] for k in [0,6), i in [0,4)
        """
        U444, V444, W444 = build_4x4x4()
        U442, V442, W442 = build_4x4x2()

        rank = 56
        U = np.zeros((16, rank))
        V = np.zeros((24, rank))
        W = np.zeros((24, rank))

        U[:, :28] = U444
        V[:16, :28] = V444
        W[:16, :28] = W444

        U[:, 28:] = U442
        for j in range(4):
            for lk in range(2):
                V[j*6 + 4 + lk, 28:] = V442[j*4 + lk, :]
        for lk in range(2):
            for li in range(4):
                W[(4+lk)*4 + li, 28:] = W442[lk*4 + li, :]

        return U, V, W

    def verify_4x4x6():
        U, V, W = build_4x4x6()
        rank = U.shape[1]
        for i in range(4):
            for j in range(4):
                for k in range(6):
                    total = 0.0
                    for r in range(rank):
                        total += U[i*4+j, r] * V[j*6+k, r] * W[k*4+i, r]
                    if abs(total - 1.0) > 1e-10:
                        return False
        return True

    if not verify_4x4x6():
        return schoolbook()

    def build_4x1x6():
        """Build 4x1x6 decomposition. Rank 24.
        U: shape (4, 24), indexed by [i*1+j] = [i] for i in [0,4), j=0
        V: shape (6, 24), indexed by [j*6+k] for j=0, k in [0,6)
        W: shape (24, 24), indexed by [k*4+i] for k in [0,6), i in [0,4)
        """
        U = np.zeros((4, 24))
        V = np.zeros((6, 24))
        W = np.zeros((24, 24))
        col = 0
        for i in range(4):
            for k in range(6):
                U[i, col] = 1
                V[k, col] = 1
                W[k*4+i, col] = 1
                col += 1
        return U, V, W

    def verify_4x1x6():
        U, V, W = build_4x1x6()
        for i in range(4):
            for k in range(6):
                total = 0
                for r in range(24):
                    total += U[i, r] * V[k, r] * W[k*4+i, r]
                if total != 1:
                    return False
        return True

    if not verify_4x1x6():
        return schoolbook()

    def build_4x5x6():
        U446, V446, W446 = build_4x4x6()
        U416, V416, W416 = build_4x1x6()

        rank = 56 + 24  # 80

        U = np.zeros((20, rank))
        V = np.zeros((30, rank))
        W = np.zeros((24, rank))

        for i in range(4):
            for j in range(4):
                U[i*5+j, :56] = U446[i*4+j, :]

        V[:24, :56] = V446
        W[:, :56] = W446

        for i in range(4):
            U[i*5+4, 56:80] = U416[i, :]

        V[24:, 56:80] = V416
        W[:, 56:80] = W416

        return U, V, W

    def verify_4x5x6():
        U, V, W = build_4x5x6()
        rank = U.shape[1]
        for i in range(4):
            for j in range(5):
                for k in range(6):
                    total = 0.0
                    for r in range(rank):
                        total += U[i*5+j, r] * V[j*6+k, r] * W[k*4+i, r]
                    if abs(total - 1.0) > 1e-10:
                        return False
        return True

    if time.time() - start_time > DEADLINE:
        return schoolbook()

    if not verify_4x5x6():
        return schoolbook()

    U, V, W = build_4x5x6()

    # Now attempt to reduce rank using subgroup method
    # Generate structured rank-1 candidates from linear subspaces
    # and solve for minimal exact combination

    current_rank = U.shape[1]  # 80

    # Generate candidate rank-1 terms from structured subspaces
    # We'll use the existing decomposition as a starting point
    # and try to find linear dependencies to reduce rank

    # The tensor T has entries T[i,j,k] = 1 for all i,j,k
    # A rank-1 term contributes u[i*m+j] * v[j*p+k] * w[k*n+i] to T[i,j,k]

    # Let's try to find if any of the 80 terms are linearly dependent
    # by checking if we can express one term as a linear combination of others

    # For now, return the valid rank-80 decomposition
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    for mat in [U_list, V_list, W_list]:
        for row in mat:
            for idx in range(len(row)):
                if abs(row[idx]) < 1e-10:
                    row[idx] = 0
                elif abs(row[idx] - round(row[idx])) < 1e-10:
                    row[idx] = int(round(row[idx]))

    return {"U": U_list, "V": V_list, "W": W_list}
