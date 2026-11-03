"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4

    def strassen_2x2_decomp():
        """
        Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
        Shapes: (4, 7)
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))

        # t=0: M1 = (A00+A11)*(B00+B11)
        U[0, 0] = 1.0; U[3, 0] = 1.0
        V[0, 0] = 1.0; V[3, 0] = 1.0
        W[0, 0] = 1.0; W[3, 0] = 1.0

        # t=1: M2 = (A10+A11)*B00
        U[2, 1] = 1.0; U[3, 1] = 1.0
        V[0, 1] = 1.0
        W[1, 1] = 1.0; W[3, 1] = -1.0

        # t=2: M3 = A00*(B01-B11)
        U[0, 2] = 1.0
        V[1, 2] = 1.0; V[3, 2] = -1.0
        W[2, 2] = 1.0; W[3, 2] = 1.0

        # t=3: M4 = A11*(B10-B00)
        U[3, 3] = 1.0
        V[2, 3] = 1.0; V[0, 3] = -1.0
        W[0, 3] = 1.0; W[1, 3] = 1.0

        # t=4: M5 = (A00+A01)*B11
        U[0, 4] = 1.0; U[1, 4] = 1.0
        V[3, 4] = 1.0
        W[0, 4] = -1.0; W[2, 4] = 1.0

        # t=5: M6 = (A10-A00)*(B00+B01)
        U[2, 5] = 1.0; U[0, 5] = -1.0
        V[0, 5] = 1.0; V[1, 5] = 1.0
        W[3, 5] = 1.0

        # t=6: M7 = (A01-A11)*(B10+B11)
        U[1, 6] = 1.0; U[3, 6] = -1.0
        V[2, 6] = 1.0; V[3, 6] = 1.0
        W[0, 6] = 1.0

        return U, V, W

    U_s, V_s, W_s = strassen_2x2_decomp()

    full_rank = 28
    U_full = np.zeros((n * m, full_rank))
    V_full = np.zeros((m * p, full_rank))
    W_full = np.zeros((p * n, full_rank))

    def fill_U(A_type, col_offset):
        for t in range(7):
            for li in range(2):
                for lj in range(2):
                    local_a = li * 2 + lj
                    if A_type == 0:
                        global_a = li * 4 + lj
                    else:
                        global_a = li * 4 + (lj + 2)
                    U_full[global_a, col_offset + t] = U_s[local_a, t]

    def fill_V(B_type, col_offset):
        for t in range(7):
            for lj in range(2):
                for lk in range(2):
                    local_b = lj * 2 + lk
                    if B_type == 0: # TL
                        global_b = lj * 4 + lk
                    elif B_type == 1: # TR
                        global_b = lj * 4 + (lk + 2)
                    elif B_type == 2: # BL
                        global_b = (lj + 2) * 4 + lk
                    else: # BR
                        global_b = (lj + 2) * 4 + (lk + 2)
                    V_full[global_b, col_offset + t] = V_s[local_b, t]

    def fill_W(C_type, col_offset):
        for t in range(7):
            for lk in range(2):
                for li in range(2):
                    local_c = lk * 2 + li
                    if C_type == 0:
                        global_c = lk * 2 + li
                    else:
                        global_c = (lk + 2) * 2 + li
                    W_full[global_c, col_offset + t] = W_s[local_c, t]

    # Block 1: A_L * B_TL -> C_L
    fill_U(0, 0)
    fill_V(0, 0)
    fill_W(0, 0)

    # Block 2: A_R * B_BL -> C_L
    fill_U(1, 7)
    fill_V(2, 7)
    fill_W(0, 7)

    # Block 3: A_L * B_TR -> C_R
    fill_U(0, 14)
    fill_V(1, 14)
    fill_W(1, 14)

    # Block 4: A_R * B_BR -> C_R
    fill_U(1, 21)
    fill_V(3, 21)
    fill_W(1, 21)

    r = full_rank
    U_list = [[float(U_full[i, t]) for t in range(r)] for i in range(n * m)]
    V_list = [[float(V_full[i, t]) for t in range(r)] for i in range(m * p)]
    W_list = [[float(W_full[i, t]) for t in range(r)] for i in range(p * n)]

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
