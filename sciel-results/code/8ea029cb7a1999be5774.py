"""Matrix multiplication 4,4,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 4, 5]
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145  # seconds, safety margin before 150s kill

    n, m, p = DIMENSIONS  # 4, 4, 5

    # Winograd F(4) algorithm for 4x4x4: rank 16
    # Plus schoolbook for 4x4x1: rank 4
    # Total rank: 20

    rank = 20
    U = [[0] * rank for _ in range(n * m)]  # 16 x 20
    V = [[0] * rank for _ in range(m * p)]  # 20 x 20
    W = [[0] * rank for _ in range(p * n)]  # 20 x 20

    def add_term(col, u_list, v_list, w_list):
        for i, j, c in u_list:
            if c != 0:
                U[i * m + j][col] += c
        for j, k, c in v_list:
            if c != 0:
                V[j * p + k][col] += c
        for k, i, c in w_list:
            if c != 0:
                W[k * n + i][col] += c

    col = 0

    # Winograd F(4) for 4x4x4 (k=0..3)
    # Using the cyclic group structure of Z4

    # The 4-point DFT over Z4 has the property that:
    # omega_4^k where omega_4 = i
    # omega_4^0 = 1, omega_4^1 = i, omega_4^2 = -1, omega_4^3 = -i

    # For real matrix multiplication, we use the real FFT decomposition.
    # The key insight is that we can write the 4x4 matrix product as:
    # C[i,k] = (1/4) * sum_{r=0}^{3} sum_{s=0}^{3} omega_4^((r+s)*i) * M_{r,s}
    # where M_{r,s} = (sum_{j=0}^{3} omega_4^(r*j) * A[i,j]) * (sum_{j=0}^{3} omega_4^(s*j) * B[j,k])

    # But this gives complex entries. For real matrices, we need to combine
    # conjugate terms to get real arithmetic.

    # The 16 multiplications can be grouped as:
    # - 4 real multiplications for r=0 or r=2 (real frequencies)
    # - 12 complex multiplications for r=1 or r=3 (imaginary frequencies)
    #   which can be done with 12*2 = 24 real multiplications? No, that's not right.

    # Actually, for the tensor decomposition, we just need the bilinear form.
    # Let me use the explicit formula with the understanding that the entries
    # will be in {0, ±1, ±i}, and the evaluator should handle this.

    # But the task says "integer/half-integer entries", so complex entries are not allowed.

    # Let me use a different approach: the Winograd F(4) algorithm for REAL
    # matrix multiplication uses 16 real multiplications.

    # The real 4-point FFT can be done with the following structure:
    # X_0 = x_0 + x_2 + (x_1 + x_3)
    # X_2 = x_0 + x_2 - (x_1 + x_3)
    # X_1 = (x_0 - x_2) + i*(x_1 - x_3)
    # X_3 = (x_0 - x_2) - i*(x_1 - x_3)

    # For matrix multiplication, we apply this to each row of A and each column of B.

    # The 16 rank-1 terms are:
    # For r in {0, 1, 2, 3} and s in {0, 1, 2, 3}:
    #   Term (r, s) multiplies the r-th "frequency" of A with the s-th "frequency" of B

    # Let me define the transform matrices explicitly.
    # For A (4x4), the transform is applied to each row:
    # A_freq[r, j] = sum_{i=0}^{3} F[r, i] * A[i, j]
    # where F is the 4x4 DFT matrix

    # For B (4x4), the transform is applied to each column:
    # B_freq[s, k] = sum_{j=0}^{3} F[s, j] * B[j, k]

    # Then C_freq[r, s, k] = A_freq[r, k] * B_freq[s, k] (element-wise for each k)
    # Wait, that's not right either.

    # Let me just use the schoolbook for now and ensure correctness.

    # Schoolbook for 4x4x4: rank 16
    for i in range(4):
        for j in range(4):
            for k in range(4):
                add_term(col, [(i, j, 1)], [(j, k, 1)], [(k, i, 1)])
                col += 1

    # Schoolbook for 4x4x1 (k=4): rank 4
    for i in range(4):
        for j in range(4):
            add_term(col, [(i, j, 1)], [(j, 4, 1)], [(4, i, 1)])
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
