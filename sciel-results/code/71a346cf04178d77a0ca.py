import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank_school = n * m * p  # 40

    start_time = time.time()
    deadline = start_time + 110.0

    # Build schoolbook decomposition as fallback
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Winograd Fourier-based decomposition for 2x4x5
    # Strategy: decompose into sub-tensors and use low-rank decompositions
    # 2x4x5 = 2x(2x2)x5
    # For each 2x2 block of columns, we can use Winograd/Strassen
    # 2x2x5 = 2x2x2 + 2x2x2 + 2x2x1
    # Each 2x2x2 has rank 7 (Strassen)
    # Each 2x2x1 has rank 4 (schoolbook, can't reduce)
    # Total: 2*(7+7+4) = 36

    # But we can try to share some multiplications
    # Actually, let's try a different decomposition
    # 2x4x5 = (2x2)x4x5 + (2x2)x4x5
    # Each 2x2x4x5 can be decomposed as:
    # 2x2x(2x2)x5 = 2x(2x2x5)
    # 2x2x5 = 2x2x2 + 2x2x2 + 2x2x1
    # = 7 + 7 + 4 = 18
    # Total: 2*18 = 36

    # Let me try to find a rank 35 decomposition
    # by sharing some of the 2x2x1 blocks

    # Alternative: use the fact that 2x4x5 can be viewed as
    # multiplying a 2x4 matrix by a 4x5 matrix
    # We can partition the 4 columns of A and 4 rows of B into
    # two groups of 2, and the 5 columns of B into groups of 2,2,1

    # Let me implement the Winograd algorithm for 2x2x3
    # which has rank 8 (vs schoolbook 12)
    # Then 2x4x5 = 2x(2x2)x(2+2+1) = 2*(2x2x2 + 2x2x2 + 2x2x1)
    # = 2*(7 + 7 + 4) = 36

    # Or: 2x4x5 = (2x4)x(4x5)
    # = (2x2 + 2x2)x(4x5)
    # = 2x2x4x5 + 2x2x4x5
    # = 2*(2x2x(2+2+1))
    # = 2*(2x2x2 + 2x2x2 + 2x2x1)
    # = 2*(7 + 7 + 4) = 36

    # Let me try a different approach: use the fact that
    # 2x4x5 = 2x(4x5)
    # and 4x5 can be decomposed using Winograd
    # 4x5 = (2+2)x(2+2+1)
    # = 2x2x2 + 2x2x2 + 2x2x1 + 2x2x2 + 2x2x2 + 2x2x1
    # = 2*(7 + 7 + 4) = 36
    # Then 2x4x5 = 2*36 = 72? No, that's not right.

    # Actually, the tensor is T[i,j,k] where i in [0,2), j in [0,4), k in [0,5)
    # T[i,j,k] = 1 if j=k, else 0
    # This is the matrix multiplication tensor for 2x4 * 4x5

    # Let me just implement the rank 36 decomposition and verify it
    total_rank = 36
    U = np.zeros((nm, total_rank), dtype=np.float64)
    V = np.zeros((mp, total_rank), dtype=np.float64)
    W = np.zeros((pn, total_rank), dtype=np.float64)

    def add_strassen_2x2x2(U, V, W, i0, j0, k0, col):
        # Strassen decomposition for 2x2x2
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-f)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # i = M1+M4-M5+M7
        # j = M3+M5
        # k = M2+M4
        # l = M1-M2+M3+M6

        # M1 = (a+d)(e+h)
        U[i0*4 + j0, col] = 1
        U[i0*4 + j0+1, col] = 1
        V[j0*5 + k0, col] = 1
        V[j0*4 + k0+1, col] = 1
        W[k0*2 + i0, col] = 1
        W[k0*2 + i0+1, col] = 1
        col += 1

        # M2 = (c+d)e
        U[i0*4 + j0+2, col] = 1
        U[i0*4 + j0+3, col] = 1
        V[j0*5 + k0, col] = 1
        W[k0*2 + i0+1, col] = 1
        col += 1

        # M3 = a(f-h)
        U[i0*4 + j0, col] = 1
        V[j0*4 + k0+1, col] = 1
        V[j0*4 + k0+2, col] = -1
        W[k0*2 + i0, col] = 1
        W[k0*2 + i0+1, col] = 1
        col += 1

        # M4 = d(g-f)
        U[i0*4 + j0+3, col] = 1
        V[j0*4 + k0+1, col] = -1
        V[j0*4 + k0+2, col] = 1
        W[k0*2 + i0, col] = 1
        W[k0*2 + i0+1, col] = 1
        col += 1

        # M5 = (a+b)h
        U[i0*4 + j0, col] = 1
        U[i0*4 + j0+1, col] = 1
        V[j0*4 + k0+1, col] = 1
        W[k0*2 + i0, col] = -1
        W[k0*2 + i0+1, col] = 1
        col += 1

        # M6 = (c-a)(e+f)
        U[i0*4 + j0+2, col] = 1
        U[i0*4 + j0, col] = -1
        V[j0*5 + k0, col] = 1
        V[j0*4 + k0+1, col] = 1
        W[k0*2 + i0+1, col] = 1
        col += 1

        # M7 = (b-d)(g+h)
        U[i0*4 + j0+1, col] = 1
        U[i0*4 + j0+3, col] = -1
        V[j0*4 + k0+1, col] = 1
        V[j0*4 + k0+2, col] = 1
        W[k0*2 + i0, col] = 1
        col += 1

        return col

    def add_schoolbook_2x2x1(U, V, W, i0, j0, k0, col):
        # Schoolbook for 2x2x1
        # A = [[a,b],[c,d]], B = [[e],[f]], C = [[g],[h]]
        # g = ae + bf
        # h = ce + df
        # 4 multiplications

        # ae -> g
        U[i0*4 + j0, col] = 1
        V[j0*5 + k0, col] = 1
        W[k0*2 + i0, col] = 1
        col += 1

        # bf -> g
        U[i0*4 + j0+1, col] = 1
        V[j0*4 + k0+1, col] = 1
        W[k0*2 + i0, col] = 1
        col += 1

        # ce -> h
        U[i0*4 + j0+2, col] = 1
        V[j0*5 + k0, col] = 1
        W[k0*2 + i0+1, col] = 1
        col += 1

        # df -> h
        U[i0*4 + j0+3, col] = 1
        V[j0*4 + k0+1, col] = 1
        W[k0*2 + i0+1, col] = 1
        col += 1

        return col

    col = 0
    # First 2x2 block of columns (j=0,1)
    col = add_strassen_2x2x2(U, V, W, 0, 0, 0, col)  # k=0,1
    col = add_strassen_2x2x2(U, V, W, 0, 0, 2, col)  # k=2,3
    col = add_schoolbook_2x2x1(U, V, W, 0, 0, 4, col)  # k=4

    # Second 2x2 block of columns (j=2,3)
    col = add_strassen_2x2x2(U, V, W, 0, 2, 0, col)  # k=0,1
    col = add_strassen_2x2x2(U, V, W, 0, 2, 2, col)  # k=2,3
    col = add_schoolbook_2x2x1(U, V, W, 0, 2, 4, col)  # k=4

    assert col == total_rank, f"Expected {total_rank}, got {col}"

    def verify_decomposition(U_mat, V_mat, W_mat):
        rank = U_mat.shape[1]
        # Check that T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 if j==k, else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_mat[i * m + j]
                    v_val = V_mat[j * p + k]
                    w_val = W_mat[k * n + i]
                    s = np.sum(u_val * v_val * w_val)
                    expected = 1.0 if j == k else 0.0
                    if abs(s - expected) > 1e-9:
                        return False
        return True

    if verify_decomposition(U, V, W):
        U_list = [[int(round(U[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(nm)]
        V_list = [[int(round(V[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(mp)]
        W_list = [[int(round(W[i, r] * 2)) / 2 for r in range(total_rank)] for i in range(pn)]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        return best_result
# EVOLVE-BLOCK-END
