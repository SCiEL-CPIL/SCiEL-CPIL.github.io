"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = 56

    # Initialize matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Strassen products definition
    # (A_factors, B_factors)
    # A_factors: list of (elem_idx, coeff)
    # B_factors: list of (elem_idx, coeff)
    # elem_idx: 0=a00, 1=a01, 2=a10, 3=a11
    strassen_products = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),  # P1: (a00+a11)*(b00+b11)
        ([(2, 1), (3, 1)], [(0, 1)]),           # P2: (a10+a11)*b00
        ([(0, 1)], [(1, 1), (3, -1)]),          # P3: a00*(b01-b11)
        ([(3, 1)], [(2, 1), (0, -1)]),          # P4: a11*(b10-b00)
        ([(0, 1), (1, 1)], [(3, 1)]),           # P5: (a00+a01)*b11
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),  # P6: (a10-a00)*(b00+b01)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),  # P7: (a01-a11)*(b10+b11)
    ]

    # C reconstruction
    # C_elem: list of (prod_idx, coeff)
    # C00: P1 + P4 - P5 + P7
    # C01: P3 + P5
    # C10: P2 + P4
    # C11: P1 - P2 + P3 + P6
    c_reconstruction = [
        [(0, 1), (3, 1), (4, -1), (6, 1)],  # C00
        [(2, 1), (4, 1)],                   # C01
        [(1, 1), (3, 1)],                   # C10
        [(0, 1), (1, -1), (2, 1), (5, 1)],  # C11
    ]

    # Instances: (a_col_off, b_row_off, b_col_off)
    # A is 2x4. Split into A_L (cols 0-1) and A_R (cols 2-3).
    # B is 4x8. Split into 4 blocks of 2x2?
    # B_{2i} is rows 0-1, cols 2i-2i+1.
    # B_{2i+1} is rows 2-3, cols 2i-2i+1.
    #
    # C_{2i} = A_L @ B_{2i}
    # C_{2i+1} = A_R @ B_{2i+1}
    
    # Let's list the 8 instances explicitly to be safe.
    # i=0: C_0 = A_L @ B[0:2, 0:2]  -> a_col=0, b_row=0, b_col=0
    #      C_1 = A_R @ B[2:4, 0:2]  -> a_col=2, b_row=2, b_col=0
    # i=1: C_2 = A_L @ B[0:2, 2:4]  -> a_col=0, b_row=0, b_col=2
    #      C_3 = A_R @ B[2:4, 2:4]  -> a_col=2, b_row=2, b_col=2
    # i=2: C_4 = A_L @ B[0:2, 4:6]  -> a_col=0, b_row=0, b_col=4
    #      C_5 = A_R @ B[2:4, 4:6]  -> a_col=2, b_row=2, b_col=4
    # i=3: C_6 = A_L @ B[0:2, 6:8]  -> a_col=0, b_row=0, b_col=6
    #      C_7 = A_R @ B[2:4, 6:8]  -> a_col=2, b_row=2, b_col=6

    instances = [
        (0, 0, 0),
        (2, 2, 0),
        (0, 0, 2),
        (2, 2, 2),
        (0, 0, 4),
        (2, 2, 4),
        (0, 0, 6),
        (2, 2, 6),
    ]

    col = 0
    for inst_idx, (a_col_off, b_row_off, b_col_off) in enumerate(instances):
        for prod_idx in range(7):
            a_factors, b_factors = strassen_products[prod_idx]
            for a_elem, a_coeff in a_factors:
                a_row = a_elem // 2
                a_col = a_elem % 2
                u_idx = a_row * m + (a_col_off + a_col)
                U[u_idx, col] += a_coeff
            for b_elem, b_coeff in b_factors:
                b_row = b_elem // 2
                b_col = b_elem % 2
                v_idx = (b_row_off + b_row) * p + (b_col_off + b_col)
                V[v_idx, col] += b_coeff
            col += 1

    # W contributions
    col = 0
    for inst_idx, (a_col_off, b_row_off, b_col_off) in enumerate(instances):
        for c_elem in range(4):
            c_row = c_elem // 2
            c_col = c_elem % 2
            w_idx = (b_col_off + c_col) * n + c_row
            for prod_idx, coeff in c_reconstruction[c_elem]:
                W[w_idx, col + prod_idx] += coeff
        col += 7

    # Verification
    def verify_full():
        if time.time() > deadline:
            return False
        A = np.random.randint(-10, 10, (n, m))
        B = np.random.randint(-10, 10, (m, p))
        C_ref = A @ B

        U_mat = U.T # rank x n*m
        V_mat = V.T # rank x m*p
        W_mat = W.T # rank x p*n

        C_calc = np.zeros((n, p))
        for r in range(rank):
            w_val_vec = W_mat[r, :]
            if not np.any(w_val_vec):
                continue
            # u_sum[i] = sum_j A[i,j] U_mat[r, i*m+j]
            # v_sum[k] = sum_j B[j,k] V_mat[r, j*p+k]
            # C_calc[i,k] += W_mat[r, k*n+i] * u_sum[i] * v_sum[k]
            
            # Vectorized u_sum
            # U_mat[r, :] is shape (n*m,). Reshape to (n, m).
            u_flat = U_mat[r, :]
            u_reshaped = u_flat.reshape(n, m)
            u_sums = A @ u_reshaped # shape (n,)
            
            # Vectorized v_sum
            # V_mat[r, :] is shape (m*p,). Reshape to (m, p).
            v_flat = V_mat[r, :]
            v_reshaped = v_flat.reshape(m, p)
            v_sums = B.T @ v_reshaped # shape (p,)
            
            # W_mat[r, :] is shape (p*n,). Reshape to (p, n).
            w_flat = W_mat[r, :]
            w_reshaped = w_flat.reshape(p, n)
            
            # C_calc += w_reshaped * u_sums[:, None] * v_sums[None, :]
            # w_reshaped is (p, n). u_sums is (n,). v_sums is (p,).
            # We need C[i, k] += w_reshaped[k, i] * u_sums[i] * v_sums[k]
            # This is an outer product scaled by w_reshaped.
            C_calc += w_reshaped * u_sums[None, :] * v_sums[:, None]

        return np.allclose(C_calc, C_ref)

    if not verify_full():
        # Fallback to schoolbook if verification fails
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col_fb] = 1
                    V_fb[j * p + k][col_fb] = 1
                    W_fb[k * n + i][col_fb] = 1
                    col_fb += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
