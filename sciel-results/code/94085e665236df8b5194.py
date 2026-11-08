import time
import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    t0 = time.time()
    DEADLINE = 135.0

    def elapsed():
        return time.time() - t0

    # Build a rank-68 decomposition using Strassen on 2x2 blocks of A
    # partitioned as rows {0,1} and cols {0,1}, with direct terms for the rest.
    #
    # A is 3x3, B is 3x8, C is 3x8.
    #
    # Partition A:
    #   A = [[a11, a12, a13],
    #        [a21, a22, a23],
    #        [a31, a32, a33]]
    #
    # Partition B:
    #   B = [[b11, b12, ..., b18],
    #        [b21, b22, ..., b28],
    #        [b31, b32, ..., b38]]
    #
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # We use Strassen for the 2x2 block A[0:2, 0:2] * B[0:2, k:k+2] for k=0,2,4,6.
    # Then direct terms for:
    #   A[0:2, 2] * B[2, :]   -> 2*8 = 16 terms
    #   A[2, 0:2] * B[0:2, :] -> 2*8 = 16 terms
    #   A[2, 2] * B[2, :]     -> 8 terms
    #
    # Strassen: 4 blocks * 7 = 28 terms
    # Total: 28 + 16 + 16 + 8 = 68

    U_entries = []  # list of list of (index, coeff)
    V_entries = []
    W_entries = []

    def add_strassen(row_off, col_off, b_row_off, b_col_off):
        """Strassen for A[row_off:row_off+2, col_off:col_off+2] * B[b_row_off:b_row_off+2, b_col_off:b_col_off+2]"""
        # A entries
        a00 = (row_off, col_off)
        a01 = (row_off, col_off + 1)
        a10 = (row_off + 1, col_off)
        a11 = (row_off + 1, col_off + 1)
        # B entries
        b00 = (b_row_off, b_col_off)
        b01 = (b_row_off, b_col_off + 1)
        b10 = (b_row_off + 1, b_col_off)
        b11 = (b_row_off + 1, b_col_off + 1)
        # C entries
        c00 = (row_off, b_col_off)
        c01 = (row_off, b_col_off + 1)
        c10 = (row_off + 1, b_col_off)
        c11 = (row_off + 1, b_col_off + 1)

        # m1 = (a00+a11)(b00+b11)
        U_entries.append([(a00, 1), (a11, 1)])
        V_entries.append([(b00, 1), (b11, 1)])
        W_entries.append([(c00, 1), (c11, 1)])

        # m2 = (a10+a11)*b00
        U_entries.append([(a10, 1), (a11, 1)])
        V_entries.append([(b00, 1)])
        W_entries.append([(c10, 1), (c11, -1)])

        # m3 = a00*(b01-b11)
        U_entries.append([(a00, 1)])
        V_entries.append([(b01, 1), (b11, -1)])
        W_entries.append([(c01, 1), (c11, 1)])

        # m4 = a11*(b10-b00)
        U_entries.append([(a11, 1)])
        V_entries.append([(b10, 1), (b00, -1)])
        W_entries.append([(c00, 1), (c10, 1)])

        # m5 = (a00+a01)*b11
        U_entries.append([(a00, 1), (a01, 1)])
        V_entries.append([(b11, 1)])
        W_entries.append([(c00, -1), (c01, 1)])

        # m6 = (a10-a00)*(b00+b01)
        U_entries.append([(a10, 1), (a00, -1)])
        V_entries.append([(b00, 1), (b01, 1)])
        W_entries.append([(c11, 1)])

        # m7 = (a01-a11)*(b10+b11)
        U_entries.append([(a01, 1), (a11, -1)])
        V_entries.append([(b10, 1), (b11, 1)])
        W_entries.append([(c00, 1)])

    def add_direct(ai, aj, bj, bk):
        """Direct term A[ai,aj]*B[bj,bk] -> C[ai,bk]"""
        U_entries.append([((ai, aj), 1)])
        V_entries.append([((bj, bk), 1)])
        W_entries.append([((ai, bk), 1)])

    # Strassen blocks for A[0:2, 0:2] * B[0:2, k:k+2] for k = 0, 2, 4, 6
    for k in range(0, 8, 2):
        add_strassen(0, 0, 0, k)

    # Direct: A[0:2, 2] * B[2, :]
    for i in range(2):
        for k in range(8):
            add_direct(i, 2, 2, k)

    # Direct: A[2, 0:2] * B[0:2, :]
    for j in range(2):
        for k in range(8):
            add_direct(2, j, j, k)

    # Direct: A[2, 2] * B[2, :]
    for k in range(8):
        add_direct(2, 2, 2, k)

    rank = len(W_entries)
    assert rank == 68, f"Expected rank 68, got {rank}"

    # Build matrices U (n*m x rank), V (m*p x rank), W (p*n x rank)
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)

    for idx in range(rank):
        for (i, j), coeff in U_entries[idx]:
            U[i * m + j, idx] = coeff
        for (j, k), coeff in V_entries[idx]:
            V[j * p + k, idx] = coeff
        for (i, k), coeff in W_entries[idx]:
            W[k * n + i, idx] = coeff

    def to_list(mat):
        res = []
        for row in mat:
            nrow = []
            for v in row:
                if v is None:
                    nrow.append(0)
                else:
                    fv = float(v)
                if abs(fv - round(fv * 2) / 2) < 1e-9:
                    nv = round(fv * 2) / 2
                    if abs(nv - round(nv)) < 1e-9:
                        nrow.append(int(round(nv)))
                    else:
                        nrow.append(nv)
                else:
                    nrow.append(fv)
            res.append(nrow)
        return res

    U_list = to_list(U)
    V_list = to_list(V)
    W_list = to_list(W)

    # Verify correctness
    if elapsed() < DEADLINE:
        U_np = np.array(U_list, dtype=float)
        V_np = np.array(V_list, dtype=float)
        W_np = np.array(W_list, dtype=float)

        # Test with random matrices
        rng = np.random.RandomState(42)
        A_test = rng.randn(n, m)
        B_test = rng.randn(m, p)
        C_expected = A_test @ B_test

        # Compute C from decomposition
        C_comp = np.zeros((n, p))
        for r in range(rank):
            u_col = U_np[:, r]
            v_col = V_np[:, r]
            w_col = W_np[:, r]
            # u_col is (n*m,), v_col is (m*p,), w_col is (p*n,)
            # Reshape
            u_mat = u_col.reshape(n, m)
            v_mat = v_col.reshape(m, p)
            w_mat = w_col.reshape(p, n)
            # Contribution: sum over i,j of u_mat[i,j]*A[i,j] * sum over j,k of v_mat[j,k]*B[j,k] * w_mat[k,i]
            # Actually the tensor decomposition is:
            # C[i,k] = sum_r (sum_j A[i,j]*U[j*m+i, r]) * (sum_j B[j,k]*V[k*p+j, r]) * W[k*n+i, r]
            # Wait, let me re-check the convention.
            # U has shape (n*m, rank), indexed by [i*m+j, r] for A[i,j]
            # V has shape (m*p, rank), indexed by [j*p+k, r] for B[j,k]
            # W has shape (p*n, rank), indexed by [k*n+i, r] for C[i,k]
            # So C[i,k] = sum_r U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r]
            # Hmm, that's not quite right. Let me think again.
            # The tensor T[i,j,k,l] = delta_{j,k} for matrix multiplication A[i,j]*B[j,k]->C[i,k]
            # Decomposition: T = sum_r U_r[i,j] * V_r[j,k] * W_r[i,k]
            # where U_r is n x m, V_r is m x p, W_r is n x p (but W is stored as p*n)
            # So C[i,k] = sum_r (sum_{j} U_r[i,j] * A[i,j]) * (sum_{j} V_r[j,k] * B[j,k]) * W_r[i,k]
            # But that's not standard. Let me re-read the problem.
            # "The output factor follows the official trace convention W[k*n+i,r]"
            # So W[k*n+i, r] corresponds to C[i,k] with index k*n+i.
            # The decomposition should be:
            # C[i,k] = sum_r (sum_j A[i,j] * U[i*m+j, r]) * (sum_j B[j,k] * V[j*p+k, r]) * W[k*n+i, r]
            # No wait, that would make each term a product of three scalars, but we need:
            # C[i,k] = sum_r U_r[i,j]*A[i,j] * V_r[j,k]*B[j,k] * W_r[i,k] summed over j
            # Actually the standard tensor decomposition for matrix multiplication is:
            # C[i,k] = sum_r (sum_j A[i,j] * U_r[i,j]) * (sum_j B[j,k] * V_r[j,k]) * W_r[i,k]
            # But that's not right either. The correct form is:
            # C[i,k] = sum_r sum_j U_r[i,j] * A[i,j] * V_r[j,k] * B[j,k] * W_r[i,k] / (normalization)
            # No, the standard form is:
            # C[i,k] = sum_r (sum_j A[i,j] * U_r[i,j]) * (sum_j B[j,k] * V_r[j,k]) * W_r[i,k]
            # Hmm, let me just verify numerically.
            pass

        # Let me just do a direct numerical check
        C_from_decomp = np.zeros((n, p))
        for r in range(rank):
            # U[:, r] is (n*m,), V[:, r] is (m*p,), W[:, r] is (p*n,)
            u_vec = U_np[:, r]  # n*m
            v_vec = V_np[:, r]  # m*p
            w_vec = W_np[:, r]  # p*n

            # u_vec[i*m+j] = U_r[i,j]
            # v_vec[j*p+k] = V_r[j,k]
            # w_vec[k*n+i] = W_r[i,k]

            # C[i,k] += sum_j U_r[i,j] * A[i,j] * V_r[j,k] * B[j,k] * W_r[i,k]
            # But this is O(n*m*p) per r, total O(rank*n*m*p)
            for i in range(n):
                for k in range(p):
                    w_val = w_vec[k * n + i]
                    if w_val == 0:
                        continue
                    s = 0.0
                    for j in range(m):
                        u_val = u_vec[i * m + j]
                        v_val = v_vec[j * p + k]
                        s += u_val * A_test[i, j] * v_val * B_test[j, k]
                    C_from_decomp[i, k] += s * w_val

        max_err = np.max(np.abs(C_from_decomp - C_expected))
        if max_err > 1e-6:
            # Fallback: return identity-like decomposition (rank 72)
            U_id = []
            V_id = []
            W_id = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_id.append([((i, j), 1)])
                        V_id.append([((j, k), 1)])
                        W_id.append([((i, k), 1)])
            rank_id = len(W_id)
            U_id_mat = np.zeros((n * m, rank_id), dtype=float)
            V_id_mat = np.zeros((m * p, rank_id), dtype=float)
            W_id_mat = np.zeros((p * n, rank_id), dtype=float)
            for idx in range(rank_id):
                for (i, j), coeff in U_id[idx]:
                    U_id_mat[i * m + j, idx] = coeff
                for (j, k), coeff in V_id[idx]:
                    V_id_mat[j * p + k, idx] = coeff
                for (i, k), coeff in W_id[idx]:
                    W_id_mat[k * n + i, idx] = coeff
            return {"U": to_list(U_id_mat), "V": to_list(V_id_mat), "W": to_list(W_id_mat)}

    return {"U": U_list, "V": V_list, "W": W_list}
