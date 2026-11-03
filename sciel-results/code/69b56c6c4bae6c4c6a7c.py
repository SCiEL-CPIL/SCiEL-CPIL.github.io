import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    # We construct a rank-minimal (or near-minimal) decomposition for 3x3x7.
    # Strategy:
    # 1. Use Strassen's algorithm (rank 7) for the top-left 2x2 block multiplication (A[0:2,0:2] * B[0:2,0:2]).
    # 2. Use schoolbook for the remaining terms involving row 2 or col 2.
    #
    # Let's analyze the structure:
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1.
    # B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x1, B21 is 1x2, B22 is 1x1.
    # C = A * B.
    #
    # C11 (2x2) = A11*B11 + A12*B21
    # C12 (2x1) = A11*B12 + A12*B22
    # C21 (1x2) = A21*B11 + A22*B21
    # C22 (1x1) = A21*B12 + A22*B22
    #
    # We can decompose this into:
    # 1. A11 * B11 (2x2 * 2x2) -> Rank 7 (Strassen)
    # 2. A12 * B21 (2x1 * 1x2) -> Rank 2 (Schoolbook: 2*1*2 = 4? No, 2x1 * 1x2 is outer product. Rank is 2*1=2? No, rank of tensor is min rank. For 2x1 * 1x2, the tensor is T[i,k] = sum_j A12[i,j] B21[j,k]. Here j is only 0. So T[i,k] = A12[i,0] * B21[0,k]. This is rank 1? No, it's a matrix product of 2x1 and 1x2. The tensor rank of the multiplication of 2x1 and 1x2 matrices is 2. Wait.
    #    Let's check. A12 is 2x1 (vector u), B21 is 1x2 (vector v). Product is u*v^T (2x2 matrix).
    #    The tensor for multiplying u (dim 2) and v (dim 2) to get a 2x2 matrix? No, the output is a 2x2 matrix.
    #    The tensor T_{i,k} = u_i v_k. This is rank 1? No, the tensor is T(i,k) = u_i v_k. This is a rank 1 tensor in the space of 2x2 matrices?
    #    Wait, the tensor decomposition of matrix multiplication A(2x1) * B(1x2) -> C(2x2).
    #    C_{i,k} = sum_{j=0}^0 A_{i,j} B_{j,k} = A_{i,0} B_{0,k}.
    #    This is a single term: U_i = A_{i,0}, V_k = B_{0,k}, W_{i,k} = 1.
    #    So rank is 1?
    #    Let's verify. If we have 1 term, C_{i,k} = U_{i,1} V_{1,k} W_{i,k,1}.
    #    Here U_{i,1} = A_{i,0}, V_{1,k} = B_{0,k}, W_{i,k,1} = 1.
    #    Yes, rank 1.
    #    So A12*B21 is rank 1.
    #
    # 3. A11 * B12 (2x2 * 2x1) -> Rank 4 (Schoolbook: 2*2*1 = 4). Can it be lower?
    #    A11 (2x2) * B12 (2x1) -> C12 (2x1).
    #    C_{i,0} = sum_j A11_{i,j} B12_{j,0}.
    #    This is standard 2x2 * 2x1. Rank is 4? Or can we use Strassen-like?
    #    Strassen is for 2x2 * 2x2. For 2x2 * 2x1, we can view B12 as [B12; 0] and take the top half of the result?
    #    If we use Strassen for 2x2 * 2x2, we get 7 products. The result is 2x2. We only need the first column.
    #    Does the first column depend on all 7 products?
    #    C11 = M1 + M4 - M5 + M7
    #    C12 = M3 + M5
    #    So C12 (first col of result) depends on M1, M4, M5, M7. (4 terms).
    #    But M1, M4, M5, M7 are linear combinations of A and B.
    #    So we need 4 products. Rank 4.
    #
    # 4. A21 * B11 (1x2 * 2x2) -> Rank 4 (Schoolbook). Similar to above, rank 4.
    #
    # 5. A22 * B21 (1x1 * 1x2) -> Rank 1.
    # 6. A21 * B12 (1x2 * 2x1) -> Rank 2?
    #    C_{0,0} = A21_{0,0} B12_{0,0} + A21_{0,1} B12_{1,0}.
    #    This is dot product of two vectors of dim 2. Rank 2.
    # 7. A22 * B22 (1x1 * 1x1) -> Rank 1.
    #
    # Let's sum the ranks:
    # A11*B11: 7
    # A12*B21: 1
    # A11*B12: 4
    # A21*B11: 4
    # A22*B21: 1
    # A21*B12: 2
    # A22*B22: 1
    # Total: 7 + 1 + 4 + 4 + 1 + 2 + 1 = 20.
    #
    # Wait, is it that simple? We can just sum the decompositions.
    # The U, V, W matrices will have columns corresponding to each term.
    # We need to ensure the indices are correct.
    #
    # Let's verify the rank 20 construction.
    #
    # Terms:
    # 1. Strassen for A11*B11 (indices 0,1 in A and B).
    #    M1 = (A00+A11)(B00+B11)
    #    M2 = (A10+A11)B00
    #    M3 = A00(B10+B11)
    #    M4 = A11(B00-B11)
    #    M5 = (A00+A10)(B10+B11)
    #    M6 = (A10-A00)(B00+B10)
    #    M7 = (A11-A10)(B10+B11)
    #
    #    C00 = M1 + M4 - M5 + M7
    #    C01 = M3 + M5
    #    C10 = M2 + M4
    #    C11 = M1 - M2 + M6 + M7
    #
    #    These contribute to C[0:2, 0:2].
    #
    # 2. A12*B21 (A[0:2, 2] * B[2, 0:2]).
    #    C[i, k] += A[i, 2] * B[2, k] for i in 0,1 and k in 0,1.
    #    This is rank 1.
    #    U: A[0,2]=1, A[1,2]=1 (Wait, U is vector for A. For term t, U[i,j,t]. Here i=0,1; j=2. So U[0,2]=1, U[1,2]=1? No.
    #    In Strassen, U is a vector of length 9.
    #    For A12*B21:
    #    We need to compute C[i,k] += A[i,2] B[2,k].
    #    This can be done with 1 term?
    #    Term: U[0,2]=1, U[1,2]=1? No.
    #    If U[0,2]=1 and U[1,2]=1, then U is [0,0,1, 0,0,1, 0,0,0].
    #    V[2,0]=1, V[2,1]=1? V is length 6. V[j*p+k]. j=2. k=0,1.
    #    V[2*2+0]=1, V[2*2+1]=1. So V[4]=1, V[5]=1.
    #    W[i,k]=1 for i=0,1 and k=0,1.
    #    W[0*3+0]=1, W[0*3+1]=1, W[1*3+0]=1, W[1*3+1]=1.
    #    W[0]=1, W[1]=1, W[3]=1, W[4]=1.
    #    Check: Sum_t U[i,j,t] V[j,k,t] W[k,i,t]? No, W[k*n+i].
    #    Term value for C[i,k] is U[i,2] * V[2,k] * W[k, i].
    #    If U[0,2]=1, U[1,2]=1, V[2,0]=1, V[2,1]=1, W[0,0]=1, W[1,0]=1, W[0,1]=1, W[1,1]=1.
    #    Then C[0,0] gets 1*1*1 = 1. (A[0,2]*B[2,0]) Correct.
    #    C[0,1] gets 1*1*1 = 1. (A[0,2]*B[2,1]) Correct.
    #    C[1,0] gets 1*1*1 = 1. (A[1,2]*B[2,0]) Correct.
    #    C[1,1] gets 1*1*1 = 1. (A[1,2]*B[2,1]) Correct.
    #    So yes, rank 1.
    #
    # 3. A11*B12 (A[0:2,0:2] * B[0:2, 2]).
    #    C[i, 2] += sum_j A[i,j] B[j,2] for i in 0,1.
    #    This is 2x2 * 2x1. Rank 4.
    #    Terms:
    #    T1: A[0,0]B[0,2] -> C[0,2]
    #    T2: A[0,1]B[1,2] -> C[0,2]
    #    T3: A[1,0]B[0,2] -> C[1,2]
    #    T4: A[1,1]B[1,2] -> C[1,2]
    #
    # 4. A21*B11 (A[2,0:2] * B[0:2,0:2]).
    #    C[2, k] += sum_j A[2,j] B[j,k] for k in 0,1.
    #    This is 1x2 * 2x2. Rank 4.
    #    Terms:
    #    T5: A[2,0]B[0,0] -> C[2,0]
    #    T6: A[2,0]B[0,1] -> C[2,1]
    #    T7: A[2,1]B[1,0] -> C[2,0]
    #    T8: A[2,1]B[1,1] -> C[2,1]
    #
    # 5. A22*B21 (A[2,2] * B[2,0:2]).
    #    C[2, k] += A[2,2] B[2,k] for k in 0,1.
    #    Rank 1.
    #    Term: U[2,2]=1, V[2,0]=1, V[2,1]=1, W[0,2]=1, W[1,2]=1.
    #    Wait, W[k*n+i]. k=0, i=2 -> W[2]=1. k=1, i=2 -> W[5]=1.
    #
    # 6. A21*B12 (A[2,0:2] * B[0:2,2]).
    #    C[2, 2] += A[2,0]B[0,2] + A[2,1]B[1,2].
    #    Rank 2.
    #    Term 1: A[2,0]B[0,2] -> C[2,2].
    #    Term 2: A[2,1]B[1,2] -> C[2,2].
    #
    # 7. A22*B22 (A[2,2] * B[2,2]).
    #    C[2, 2] += A[2,2] B[2,2].
    #    Rank 1.
    #
    # Total Rank = 7 + 1 + 4 + 4 + 1 + 2 + 1 = 20.
    #
    # Let's implement this.

    U_list = []
    V_list = []
    W_list = []

    def add_term(u_idx, v_idx, w_idx):
        # u_idx: dict { (i,j): val }
        # v_idx: dict { (j,k): val }
        # w_idx: dict { (k,i): val }
        u = [0.0] * (n * m)
        v = [0.0] * (m * p)
        w = [0.0] * (p * n)

        for (i, j), val in u_idx.items():
            u[i * m + j] = val
        for (j, k), val in v_idx.items():
            v[j * p + k] = val
        for (k, i), val in w_idx.items():
            w[k * n + i] = val

        U_list.append(u)
        V_list.append(v)
        W_list.append(w)

    # 1. Strassen for A11 * B11 (i,j in 0,1; j,k in 0,1; i,k in 0,1)
    # M1 = (A00+A11)(B00+B11)
    add_term({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1})
    # M2 = (A10+A11)B00
    add_term({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1})
    # M3 = A00(B10+B11)
    add_term({(0,0):1}, {(1,0):1, (1,1):1}, {(0,1):1})
    # M4 = A11(B00-B11)
    add_term({(1,1):1}, {(0,0):1, (1,1):-1}, {(0,0):1, (1,0):1})
    # M5 = (A00+A10)(B10+B11)
    add_term({(0,0):1, (1,0):1}, {(1,0):1, (1,1):1}, {(0,0):-1, (0,1):1})
    # M6 = (A10-A00)(B00+B10)
    add_term({(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}, {(1,1):1})
    # M7 = (A11-A10)(B10+B11)
    add_term({(1,1):1, (1,0):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):1})

    # 2. A12 * B21 (i in 0,1; j=2; k in 0,1)
    # C[i,k] += A[i,2] B[2,k]
    add_term({(0,2):1, (1,2):1}, {(2,0):1, (2,1):1}, {(0,0):1, (0,1):1, (1,0):1, (1,1):1})

    # 3. A11 * B12 (i in 0,1; j in 0,1; k=2)
    # C[i,2] += sum_j A[i,j] B[j,2]
    add_term({(0,0):1}, {(0,2):1}, {(2,0):1})
    add_term({(0,1):1}, {(1,2):1}, {(2,0):1})
    add_term({(1,0):1}, {(0,2):1}, {(2,1):1})
    add_term({(1,1):1}, {(1,2):1}, {(2,1):1})

    # 4. A21 * B11 (i=2; j in 0,1; k in 0,1)
    # C[2,k] += sum_j A[2,j] B[j,k]
    add_term({(2,0):1}, {(0,0):1}, {(0,2):1})
    add_term({(2,0):1}, {(0,1):1}, {(1,2):1})
    add_term({(2,1):1}, {(1,0):1}, {(0,2):1})
    add_term({(2,1):1}, {(1,1):1}, {(1,2):1})

    # 5. A22 * B21 (i=2; j=2; k in 0,1)
    # C[2,k] += A[2,2] B[2,k]
    add_term({(2,2):1}, {(2,0):1, (2,1):1}, {(0,2):1, (1,2):1})

    # 6. A21 * B12 (i=2; j in 0,1; k=2)
    # C[2,2] += A[2,0]B[0,2] + A[2,1]B[1,2]
    add_term({(2,0):1}, {(0,2):1}, {(2,2):1})
    add_term({(2,1):1}, {(1,2):1}, {(2,2):1})

    # 7. A22 * B22 (i=2; j=2; k=2)
    # C[2,2] += A[2,2] B[2,2]
    add_term({(2,2):1}, {(2,2):1}, {(2,2):1})

    rank = len(U_list)

    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T

    # Verify correctness with random matrices
    if time.time() < deadline - 10:
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_test = A_test @ B_test

        # Compute C from decomposition
        # C[i,k] = sum_t U[i*m+j, t] V[j*p+k, t] W[k*n+i, t]
        # This is sum over t of (U * V) elementwise? No.
        # It is sum_t (sum_j U[i,j,t] V[j,k,t]) W[k,i,t]
        # Let M = U^T @ V? No.
        # U is (n*m, rank). V is (m*p, rank).
        # We can reshape U to (n, m, rank) and V to (m, p, rank).
        # Then (U @ V) is not directly possible.
        # But sum_t U[i,j,t] V[j,k,t] is (U_reshaped * V_reshaped).sum(axis=2)?
        # U_reshaped: (n, m, rank). V_reshaped: (m, p, rank).
        # We need to sum over t and j.
        # Let's just loop for verification since rank is small.
        C_calc = np.zeros((n, p))
        for t in range(rank):
            u_vec = U[:, t].reshape(n, m)
            v_vec = V[:, t].reshape(m, p)
            w_vec = W[:, t].reshape(p, n)
            # C[i,k] += sum_j u_vec[i,j] v_vec[j,k] * w_vec[k,i]
            # This is not a standard matrix product.
            # Let's do it explicitly.
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        C_calc[i, k] += u_vec[i, j] * v_vec[j, k] * w_vec[k, i]

        if not np.allclose(C_calc, C_test):
            # Fallback to schoolbook if verification fails
            U_list = []
            V_list = []
            W_list = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u = [0.0] * (n * m)
                        v = [0.0] * (m * p)
                        w = [0.0] * (p * n)
                        u[i * m + j] = 1.0
                        v[j * p + k] = 1.0
                        w[k * n + i] = 1.0
                        U_list.append(u)
                        V_list.append(v)
                        W_list.append(w)
            U = np.array(U_list).T
            V = np.array(V_list).T
            W = np.array(W_list).T
            rank = len(U_list)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
