"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0
    n, m, p = DIMENSIONS  # 2, 4, 8

    # Strassen 2x2x2 decomposition
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    # C00 = P1+P4-P5+P7
    # C01 = P3+P5
    # C10 = P2+P4
    # C11 = P1-P2+P3+P6

    # strassen list: (U_dict, V_dict, W_dict)
    # U_dict: {(a_row, a_col): coef} for the 2x2 A block
    # V_dict: {(b_row, b_col): coef} for the 2x2 B block
    # W_dict: {(c_row, c_col): coef} for the 2x2 C block
    strassen = [
        # P1: (A00+A11)(B00+B11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # P2: (A10+A11)B00
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):-1, (1,0):1}),
        # P3: A00(B01-B11)
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # P4: A11(B10-B00)
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # P5: (A00+A01)B11
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,1):1}),
        # P6: (A10-A00)(B00+B01)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # P7: (A01-A11)(B10+B11)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    rank = 56
    U = [[0] * rank for _ in range(n * m)]   # 8 x 56
    V = [[0] * rank for _ in range(m * p)]   # 32 x 56
    W = [[0] * rank for _ in range(p * n)]   # 16 x 56

    def add_block(a_col, b_row, b_col, start_col):
        c = start_col
        for uc, vc, wc in strassen:
            for (ri, cj), coef in uc.items():
                gi = ri
                gj = a_col + cj
                U[gi * m + gj][c] = coef
            for (rj, ck), coef in vc.items():
                gj = b_row + rj
                gk = b_col + ck
                V[gj * p + gk][c] = coef
            for (rk, ri2), coef in wc.items():
                gk = b_col + rk
                gi = ri2
                W[gk * n + gi][c] = coef
            c += 1
        return c

    # 8 blocks: for each pair (a_col in {0,2}, b_row in {0,2}) and b_col in {0,2,4,6}
    col = 0
    for b_col in [0, 2, 4, 6]:
        col = add_block(0, 0, b_col, col)
        col = add_block(2, 2, b_col, col)

    # Verify decomposition quickly
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # Check: for each rank term r, U[:,r] @ V[:,r] should contribute to W[:,r] via the contraction
    # The tensor is: T[i,j,k] = sum_r U[i,m+r?] ... 
    # Actually the convention is:
    # A is n x m = 2 x 4, indexed as A[i,j], flat index i*m+j
    # B is m x p = 4 x 8, indexed as B[j,k], flat index j*p+k
    # C is n x p = 2 x 8, indexed as C[i,k], flat index i*p+k
    # But W is p x n = 8 x 2, with W[k*n+i, r]
    # 
    # The decomposition says: C[i,k] = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, that's not standard. Let me re-read the task.
    #
    # "output factor follows the official trace convention W[k*n+i,r]"
    # So W is indexed as W[k*n+i, r] for output entry C[i,k].
    #
    # The standard bilinear form: C[i,k] = sum_r U_A[i,j1(r), r] * U_B[j2(r),k,r]
    # But with three factors U, V, W:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r])
    #
    # Wait, that's not right either. Let me think about this more carefully.
    #
    # In a 3-tensor decomposition of matrix multiplication:
    # The matrix multiplication tensor T[i,j,k] = delta[i,j]delta[j,k] (with appropriate dims)
    # Actually T[i,j,k] where i indexes rows of A (and C), j indexes cols of A (and rows of B),
    # k indexes cols of B (and C). T[i,j,k] = delta[i,j] * delta[j,k]... no.
    #
    # T[i,j,k] is the coefficient such that C = A*B means C[i,k] = sum_j A[i,j]*B[j,k].
    # So the tensor T has entries T[i,j,k] = 1 if j is the shared index, else 0.
    # More precisely, T[i,j,k] = delta_{i,j} is wrong.
    #
    # The correct tensor: T[i,j,k] where the multiplication is
    # C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k]
    # So T[i,j,k] = delta_{i,j} * delta_{j,k}? No.
    # T[i,j,k] = 1 for all valid (i,j,k) where j is the contraction index.
    # Actually T[i,j,k] = delta_{i,i} * delta_{j,j} * delta_{k,k}... 
    # 
    # Let me just think of it as: the tensor T has shape (n, m, p) and
    # T[i,j,k] = 1 if the indices are consistent with matrix multiplication.
    # For matrix mult: C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p].
    # No wait, that's not a tensor decomposition issue.
    #
    # The tensor for matrix multiplication M_{n,m,p} is defined as:
    # M[i,j,k] = 1 if j is the middle index (always true for valid indices)
    # So M is just the all-ones tensor of shape (n,m,p)? No, that's not right.
    #
    # Actually, the matrix multiplication tensor T^{n,m,p} has entries:
    # T[i,j,k] = delta_{i,j} * delta_{j,k}... 
    #
    # Let me look at this differently. The standard definition:
    # T[i,j,k] = 1 if the indices form a valid term in the product, 0 otherwise.
    # For C[i,k] = sum_j A[i,j] B[j,k], every triple (i,j,k) contributes,
    # so T[i,j,k] = 1 for all i,j,k. But that can't be right because then the rank
    # would be 1 (just one factor).
    #
    # I think the correct definition is:
    # T[i,j,k] = delta_{i,j} * delta_{j,k} where the deltas are Kronecker deltas
    # in the sense that i and j must match for A, and j and k must match for B.
    # But i indexes rows of A (size n), j indexes columns of A = rows of B (size m),
    # k indexes columns of B (size p). These are different index sets!
    #
    # So T[i,j,k] = 1 for all (i,j,k) where i in [n], j in [m], k in [p].
    # The tensor is just the all-ones tensor of shape (n,m,p).
    # But that doesn't make sense for rank either.
    #
    # OK let me just verify numerically.
    # Generate random A, B, compute C = A@B.
    # Then check: sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r]) == C[i,k]
    
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B

    # Build the decomposition result
    C_dec = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            total = 0.0
            for r in range(rank):
                u_val = 0.0
                for j in range(m):
                    u_val += U[i * m + j][r] * A[i, j]
                v_val = 0.0
                for j in range(m):
                    v_val += V[j * p + k][r] * B[j, k]
                total += W[k * n + i][r] * u_val * v_val
            C_dec[i, k] = total

    max_err = np.max(np.abs(C - C_dec))
    if max_err > 1e-9:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][idx] = 1
                    V[j * p + k][idx] = 1
                    W[k * n + i][idx] = 1
                    idx += 1

    if time.time() > deadline:
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
