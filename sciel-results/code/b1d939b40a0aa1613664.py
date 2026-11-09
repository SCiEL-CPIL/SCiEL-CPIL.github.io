import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 150.0 - 2.0  # safety margin

    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1

    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school

    # Try to find rank 33 using Strassen on 2x2 sub-blocks
    # The idea: A (2x3) = [A_L (2x2) | A_R (2x1)]
    #            B (3x6) = [B_T (2x6); B_B (1x6)]
    # C = A_L * B_T + A_R * B_B
    # A_L * B_T is 2x2 * 2x6 = 2x6, which can be split into three 2x2*2x2 multiplications
    # Each 2x2*2x2 uses Strassen (rank 7)
    # A_R * B_B is 2x1 * 1x6 = 2x6, which is schoolbook (rank 2*1*6 = 12)
    # Total rank = 3 * 7 + 12 = 21 + 12 = 33

    target_rank = 33
    U = np.zeros((n*m, target_rank))
    V = np.zeros((m*p, target_rank))
    W = np.zeros((p*n, target_rank))

    col = 0
    # Strassen on three 2x2x2 blocks
    # For block b in range(3): B_T[:, 2*b:2*b+2] is a 2x2 block
    # A_L is always the same 2x2 block (columns 0,1 of A)
    for b in range(3):
        # Local Strassen indices:
        # A_L: a00=A[0,0], a01=A[0,1], a10=A[1,0], a11=A[1,1]
        # B_block: b00=B[0,2*b], b01=B[0,2*b+1], b10=B[1,2*b], b11=B[1,2*b+1]
        # C_block: c00=C[0,2*b], c01=C[0,2*b+1], c10=C[1,2*b], c11=C[1,2*b+1]

        # Strassen formulas:
        # M1 = a00*(b00+b11)
        # M2 = (a10+a11)*b00
        # M3 = a00*(b01-b11)
        # M4 = a11*(b10-b00)
        # M5 = (a00+a11)*(b00+b10)
        # M6 = (a10-a00)*(b01+b11)
        # M7 = (a01-a11)*(b10+b11)
        # C00 = M1 - M2 + M4
        # C01 = M3 + M5
        # C10 = M6 + M2
        # C11 = M1 - M3 + M7

        # M1
        U[0*3+0, col] = 1   # a00
        V[0*6+2*b, col] = 1   # b00
        V[1*6+2*b+1, col] = 1 # b11
        W[(2*b)*2+0, col] = 1 # c00
        W[(2*b+1)*2+1, col] = 1 # c11
        col += 1

        # M2
        U[1*3+0, col] = 1   # a10
        U[1*3+1, col] = 1   # a11
        V[0*6+2*b, col] = 1 # b00
        W[(2*b)*2+1, col] = 1 # c10
        W[(2*b+1)*2+0, col] = 1 # c01 -- wait, let me recheck
        # C10 = M6 + M2, so M2 contributes to c10
        # C01 = M3 + M5, so M2 does NOT contribute to c01
        # Let me re-examine the Strassen formulas more carefully
        col -= 1  # undo and redo

        # Let me use the standard Strassen decomposition for 2x2*2x2:
        # M1 = A11*(B11+B22)
        # M2 = (A21+A22)*B11
        # M3 = A11*(B12-B22)
        # M4 = A22*(B21-B11)
        # M5 = (A11+A22)*(B11+B21)
        # M6 = (A21-A11)*(B12+B22)
        # M7 = (A12-A22)*(B21+B22)
        # C11 = M1 - M2 + M4
        # C12 = M3 + M5
        # C21 = M6 + M2
        # C22 = M1 - M3 + M7

        # Where A11=a00, A12=a01, A21=a10, A22=a11
        #       B11=b00, B12=b01, B21=b10, B22=b11
        #       C11=c00, C12=c01, C21=c10, C22=c11

        # M1 = a00*(b00+b11)
        U[0*3+0, col] = 1   # a00
        V[0*6+2*b, col] = 1   # b00
        V[1*6+2*b+1, col] = 1 # b11
        W[(2*b)*2+0, col] = 1 # c00 (C11)
        W[(2*b+1)*2+1, col] = 1 # c11 (C22)
        col += 1

        # M2 = (a10+a11)*b00
        U[1*3+0, col] = 1   # a10
        U[1*3+1, col] = 1   # a11
        V[0*6+2*b, col] = 1 # b00
        W[(2*b)*2+1, col] = 1 # c10 (C21)
        col += 1

        # M3 = a00*(b01-b11)
        U[0*3+0, col] = 1   # a00
        V[0*6+2*b+1, col] = 1 # b01
        V[1*6+2*b+1, col] = -1 # -b11
        W[(2*b+1)*2+0, col] = 1 # c01 (C12)
        col += 1

        # M4 = a11*(b10-b00)
        U[1*3+1, col] = 1   # a11
        V[1*6+2*b, col] = 1 # b10
        V[0*6+2*b, col] = -1 # -b00
        W[(2*b)*2+0, col] = 1 # c00 (C11)
        col += 1

        # M5 = (a00+a11)*(b00+b10)
        U[0*3+0, col] = 1   # a00
        U[1*3+1, col] = 1   # a11
        V[0*6+2*b, col] = 1 # b00
        V[1*6+2*b, col] = 1 # b10
        W[(2*b+1)*2+0, col] = 1 # c01 (C12)
        col += 1

        # M6 = (a10-a00)*(b01+b11)
        U[1*3+0, col] = 1   # a10
        U[0*3+0, col] = -1  # -a00
        V[0*6+2*b+1, col] = 1 # b01
        V[1*6+2*b+1, col] = 1 # b11
        W[(2*b)*2+1, col] = 1 # c10 (C21)
        col += 1

        # M7 = (a01-a11)*(b10+b11)
        U[0*3+1, col] = 1   # a01
        U[1*3+1, col] = -1  # -a11
        V[1*6+2*b, col] = 1 # b10
        V[1*6+2*b+1, col] = 1 # b11
        W[(2*b+1)*2+1, col] = 1 # c11 (C22)
        col += 1

    # Now schoolbook for A_R * B_B: A_R is 2x1 (column 2 of A), B_B is 1x6 (row 2 of B)
    # C = A_R * B_B, so C[i,k] = A[i,2]*B[2,k]
    for i in range(n):
        for k in range(p):
            U[i*3+2, col] = 1
            V[2*6+k, col] = 1
            W[k*2+i, col] = 1
            col += 1

    # Verify the decomposition
    def verify_decomposition(U_mat, V_mat, W_mat, n, m, p):
        rank = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0 if j == j else 0.0  # This is just for checking the structure
                    # The tensor T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    # For matrix multiplication, T[i,j,k] = delta_{jk} (1 if j==k... no, that's not right)
                    # Actually, T[i,j,k] = 1 if this is the (i,j,k) entry of the multiplication tensor
                    # The multiplication tensor has T[i,j,k] = 1 when we're computing C[i,k] using A[i,j]*B[j,k]
                    # So T[i,j,k] = 1 for all valid i,j,k? No.
                    # The tensor is T[i,j,k] = 1 if j is the "middle" index, but actually it's a 3-tensor
                    # where T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k]
                    # So T[i,j,k] = 1 for all i,j,k (each term appears exactly once)
                    # Wait, that's the schoolbook. The tensor is T[i,j,k] = 1 for all valid (i,j,k).
                    # No, that's not right either. Let me think again.
                    # The matrix multiplication tensor T is in R^{n x m x p} where
                    # T[i,j,k] = 1 if this entry corresponds to A[i,j]*B[j,k] contributing to C[i,k]
                    # But actually, the tensor is defined such that for any matrices A,B,C:
                    # C[i,k] = sum_{j} A[i,j] * B[j,k]
                    # So the tensor T has T[i,j,k] = 1 for all i,j,k? No, that would mean
                    # C[i,k] = sum_{j} sum_{all i',j',k'} T[i',j',k'] A[i',j'] B[j',k'] delta...
                    # I think I'm overcomplicating this. Let me just verify numerically.
                    pass

        # Numerical verification: pick random A, B, compute C = A@B
        # and also compute C from the decomposition
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test

        C_decomp = np.zeros((n, p))
        for r in range(rank):
            u_vec = U_mat[:, r]  # length n*m
            v_vec = V_mat[:, r]  # length m*p
            w_vec = W_mat[:, r]  # length p*n

            # u_vec reshaped to (n, m)
            u_mat = u_vec.reshape((n, m))
            v_mat = v_vec.reshape((m, p))
            w_mat = w_vec.reshape((p, n))

            # The decomposition says: T = sum_r outer(u_r, v_r, w_r)
            # where T[i,j,k] = sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
            # And C[i,k] = sum_j T[i,j,k] * ... no, that's not right.
            # Actually, the tensor decomposition gives us:
            # sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,k} C[i,k]
            # So T[i,j,k] should equal 1 for all i,j,k? Let me check with schoolbook.
            # For schoolbook, T[i,j,k] = 1 for all i,j,k.
            # Then sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,j,k} A[i,j] B[j,k]
            # = sum_i sum_j sum_k A[i,j] B[j,k] which is not C[i,k].
            # I think the convention is different. Let me re-read the task.
            # "The output factor follows the official trace convention W[k*n+i,r]"
            # I think the tensor is such that:
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # And the rank-r decomposition gives:
            # C[i,k] = sum_{r=1}^r (sum_j U[i*m+j, r] * A[i,j]) * (sum_k V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
            # No, that doesn't make sense dimensionally.
            # Let me just do the numerical check properly.
            C_decomp[i, k] = 0
            for r in range(rank):
                term = 0
                for j in range(m):
                    term += U_mat[i*m+j, r] * A_test[i, j] * V_mat[j*p+k, r]
                C_decomp[i, k] += W_mat[k*n+i, r] * term

        if not np.allclose(C_decomp, C_expected, atol=1e-10):
            return False
        return True

    if verify_decomposition(U, V, W, n, m, p):
        best_rank = target_rank
        best_U = U
        best_V = V
        best_W = W
    else:
        # Fall back to schoolbook
        best_rank = rank_school
        best_U = U_school
        best_V = V_school
        best_W = W_school

    return {"U": best_U, "V": best_V, "W": best_W}
