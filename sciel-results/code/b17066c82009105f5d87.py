import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s limit

    n, m, p = DIMENSIONS  # 2, 4, 4

    def schoolbook_solution():
        rank = n * m * p  # 32
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify_tensor(U, V, W):
        """Fast tensor-level verification: check that the decomposed tensor
        matches the matrix multiplication tensor exactly."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)  # (n*m, rank)
        V_arr = np.array(V, dtype=float)  # (m*p, rank)
        W_arr = np.array(W, dtype=float)  # (p*n, rank)

        # Build the target multiplication tensor T of shape (n*m, m*p, p*n)
        # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k, else 0
        T = np.zeros((n * m, m * p, p * n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0

        # Compute decomposed tensor: T_dec[a, b, c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Using einsum for speed
        T_dec = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)

        if not np.allclose(T_dec, T, atol=1e-10, rtol=1e-10):
            return False
        return True

    def verify_randomized(U, V, W, max_samples=50):
        """Fast randomized verification using NumPy matrix operations."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        np.random.seed(42)
        for _ in range(max_samples):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_true = A @ B

            # C[i,k] = sum_r W[k*n+i, r] * (sum_j A[i,j]*U[i*m+j, r]) * (sum_j B[j,k]*V[j*p+k, r])
            # a_proj[i, r] = sum_j A[i,j] * U[i*m+j, r]
            A_flat = A.reshape(n * m)  # (n*m,)
            a_proj = A_flat @ U_arr  # (rank,) -- wait, this isn't right
            # Actually: a_proj[i, r] = sum_j A[i,j] * U[i*m+j, r]
            # This is: for each i, sum over j of A[i,j]*U[i*m+j, r]
            # = (A reshaped) contracted with U
            # a_proj has shape (n, rank)
            a_proj = np.zeros((n, rank))
            for i in range(n):
                for j in range(m):
                    a_proj[i] += A[i, j] * U_arr[i * m + j]

            b_proj = np.zeros((p, rank))
            for k in range(p):
                for j in range(m):
                    b_proj[k] += B[j, k] * V_arr[j * p + k]

            # C_tensor[i,k] = sum_r W[k*n+i, r] * a_proj[i, r] * b_proj[k, r]
            C_tensor = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    C_tensor[i, k] = np.sum(W_arr[k * n + i] * a_proj[i] * b_proj[k])

            if not np.allclose(C_tensor, C_true, atol=1e-9, rtol=1e-9):
                return False
        return True

    def strassen_decomposition():
        """
        Rank-28 decomposition using Strassen for 2x2 blocks.
        A is 2x4, split as A = [A1 | A2] where A1, A2 are 2x2
        B is 4x4, split as B = [[B11, B12], [B21, B22]] where each is 2x2
        C is 2x4, split as C = [C1 | C2] where C1 = A1*B11 + A2*B21, C2 = A1*B12 + A2*B22
        Total: 4 Strassen instances * 7 = 28 products
        """
        rank = 28
        U = [[0.0] * rank for _ in range(n * m)]  # 8 rows
        V = [[0.0] * rank for _ in range(m * p)]  # 16 rows
        W = [[0.0] * rank for _ in range(p * n)]  # 8 rows

        def add_strassen(col_offset, A_block, B_block, C_block_W_indices):
            a00_idx = A_block[(0, 0)]
            a01_idx = A_block[(0, 1)]
            a10_idx = A_block[(1, 0)]
            a11_idx = A_block[(1, 1)]

            b00_idx = B_block[(0, 0)]
            b01_idx = B_block[(0, 1)]
            b10_idx = B_block[(1, 0)]
            b11_idx = B_block[(1, 1)]

            c00_w = C_block_W_indices[(0, 0)]
            c01_w = C_block_W_indices[(0, 1)]
            c10_w = C_block_W_indices[(1, 0)]
            c11_w = C_block_W_indices[(1, 1)]

            # M1 = (A00+A11)*(B00+B11)
            U[a00_idx][col_offset] += 1.0
            U[a11_idx][col_offset] += 1.0
            V[b00_idx][col_offset] += 1.0
            V[b11_idx][col_offset] += 1.0
            W[c00_w][col_offset] += 1.0
            W[c11_w][col_offset] += 1.0

            # M2 = (A10+A11)*B00
            U[a10_idx][col_offset + 1] += 1.0
            U[a11_idx][col_offset + 1] += 1.0
            V[b00_idx][col_offset + 1] += 1.0
            W[c10_w][col_offset + 1] += 1.0

            # M3 = A00*(B10-B11)
            U[a00_idx][col_offset + 2] += 1.0
            V[b10_idx][col_offset + 2] += 1.0
            V[b11_idx][col_offset + 2] += -1.0
            W[c01_w][col_offset + 2] += 1.0
            W[c11_w][col_offset + 2] += 1.0

            # M4 = A11*(B01-B00)
            U[a11_idx][col_offset + 3] += 1.0
            V[b01_idx][col_offset + 3] += 1.0
            V[b00_idx][col_offset + 3] += -1.0
            W[c00_w][col_offset + 3] += 1.0
            W[c10_w][col_offset + 3] += 1.0

            # M5 = (A00+A01)*B11
            U[a00_idx][col_offset + 4] += 1.0
            U[a01_idx][col_offset + 4] += 1.0
            V[b11_idx][col_offset + 4] += 1.0
            W[c00_w][col_offset + 4] += -1.0
            W[c01_w][col_offset + 4] += 1.0

            # M6 = (A10-A00)*(B00+B01)
            U[a10_idx][col_offset + 5] += 1.0
            U[a00_idx][col_offset + 5] += -1.0
            V[b00_idx][col_offset + 5] += 1.0
            V[b01_idx][col_offset + 5] += 1.0
            W[c11_w][col_offset + 5] += 1.0

            # M7 = (A01-A11)*(B10+B11)
            U[a01_idx][col_offset + 6] += 1.0
            U[a11_idx][col_offset + 6] += -1.0
            V[b10_idx][col_offset + 6] += 1.0
            V[b11_idx][col_offset + 6] += 1.0
            W[c00_w][col_offset + 6] += 1.0
            W[c11_w][col_offset + 6] += -1.0

            return col_offset + 7

        # A1 = A[0:2, 0:2]: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
        A1 = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
        # A2 = A[0:2, 2:4]: (0,0)->4, (0,1)->5, (1,0)->6, (1,1)->7
        A2 = {(0, 0): 4, (0, 1): 5, (1, 0): 6, (1, 1): 7}

        # B11 = B[0:2, 0:2]: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5
        B11 = {(0, 0): 0, (0, 1): 1, (1, 0): 4, (1, 1): 5}
        # B12 = B[0:2, 2:4]: (0,0)->2, (0,1)->3, (1,0)->6, (1,1)->7
        B12 = {(0, 0): 2, (0, 1): 3, (1, 0): 6, (1, 1): 7}
        # B21 = B[2:4, 0:2]: (0,0)->8, (0,1)->9, (1,0)->12, (1,1)->13
        B21 = {(0, 0): 8, (0, 1): 9, (1, 0): 12, (1, 1): 13}
        # B22 = B[2:4, 2:4]: (0,0)->10, (0,1)->11, (1,0)->14, (1,1)->15
        B22 = {(0, 0): 10, (0, 1): 11, (1, 0): 14, (1, 1): 15}

        # C1 W indices: C[i,k] -> W[k*2+i]
        C1_W = {(0, 0): 0, (0, 1): 2, (1, 0): 1, (1, 1): 3}
        # C2 W indices
        C2_W = {(0, 0): 4, (0, 1): 6, (1, 0): 5, (1, 1): 7}

        col = 0
        col = add_strassen(col, A1, B11, C1_W)  # products 0-6
        col = add_strassen(col, A2, B21, C1_W)  # products 7-13
        col = add_strassen(col, A1, B12, C2_W)  # products 14-20
        col = add_strassen(col, A2, B22, C2_W)  # products 21-27

        return {"U": U, "V": V, "W": W}

    # Try Strassen rank-28 first
    try:
        strassen_sol = strassen_decomposition()
        if time.time() < deadline:
            if verify_tensor(strassen_sol["U"], strassen_sol["V"], strassen_sol["W"]):
                return strassen_sol
    except Exception:
        pass

    # Fallback to schoolbook
    return schoolbook_solution()
# EVOLVE-BLOCK-END
