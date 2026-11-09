"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    n, m, p = 2, 4, 8
    # We use a block-structured Strassen decomposition.
    # Split A (2x4) into A_L (2x2, cols 0-1) and A_R (2x2, cols 2-3).
    # Split B (4x8) into 4 blocks B_0..B_3, each 4x2.
    # Each B_l splits into B_{l,L} (rows 0-1, 2x2) and B_{l,R} (rows 2-3, 2x2).
    # C_l = A_L * B_{l,L} + A_R * B_{l,R} for each l in 0..3.
    # Each 2x2 multiplication uses Strassen (rank 7).
    # Total rank = 4 blocks * 2 sub-products * 7 Strassen = 56.

    # Strassen decomposition for 2x2 * 2x2:
    # X (2x2) * Y (2x2) = Z (2x2)
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    #
    # m1 = (x00+x11)(y00+y11)
    # m2 = (x10+x11)y00
    # m3 = x00(y10-y11)
    # m4 = x11(y01-y00)
    # m5 = (x00+x01)y11
    # m6 = (x10-x00)(y00+y01)
    # m7 = (x01-x11)(y10+y11)
    #
    # z00 = m1 + m4 - m5 + m7
    # z01 = m3 + m5
    # z10 = m2 + m4
    # z11 = m1 - m2 + m3 + m6

    # For each Strassen step t, we have:
    # U coefficients for X entries (4 values): [u00, u01, u10, u11]
    # V coefficients for Y entries (4 values): [v00, v01, v10, v11]
    # W coefficients for Z entries (4 values): [w00, w01, w10, w11]

    # Define the 7 Strassen steps:
    # (u_coeffs, v_coeffs, w_coeffs)
    strassen_steps = [
        # m1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # m2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # m3
        ([1, 0, 0, 0], [0, -1, 1, 0], [0, 1, 0, 0]),
        # m4
        ([0, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 0]),
        # m5
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]),
        # m6
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
        # m7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    rank = 56
    # U: (n*m, rank) = (8, 56)
    # V: (m*p, rank) = (32, 56)
    # W: (p*n, rank) = (16, 56)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for l in range(4):  # output block l (columns 2l, 2l+1 of C)
        # B_l rows: 0-3, cols: 2l, 2l+1
        for side in range(2):  # 0: L (A_L, B_{l,L}), 1: R (A_R, B_{l,R})
            for step_idx in range(7):
                u_coeffs, v_coeffs, w_coeffs = strassen_steps[step_idx]
                # u_coeffs: [x00, x01, x10, x11] for the 2x2 matrix X
                # v_coeffs: [y00, y01, y10, y11] for the 2x2 matrix Y
                # w_coeffs: [z00, z01, z10, z11] for the 2x2 result Z

                if side == 0:
                    # X = A_L = A[:, 0:2]
                    # X[0,0] = A[0,0] -> U[0]
                    # X[0,1] = A[0,1] -> U[1]
                    # X[1,0] = A[1,0] -> U[2]
                    # X[1,1] = A[1,1] -> U[3]
                    u_indices = [0, 1, 2, 3]
                    # Y = B_{l,L} = B[0:2, 2l:2l+2]
                    # Y[0,0] = B[0, 2l] -> V[0*p + 2l] = V[2l]
                    # Y[0,1] = B[0, 2l+1] -> V[2l+1]
                    # Y[1,0] = B[1, 2l] -> V[1*p + 2l] = V[p+2l] = V[8+2l]
                    # Y[1,1] = B[1, 2l+1] -> V[8+2l+1] = V[9+2l]
                    v_indices = [2*l, 2*l+1, 8+2*l, 8+2*l+1]
                    # Z contributes to C_l = C[0:2, 2l:2l+2]
                    # Z[0,0] -> C[0, 2l] -> W index: k*n+i where k=2l, i=0 -> 2l*2+0 = 4l
                    # Z[0,1] -> C[0, 2l+1] -> k=2l+1, i=0 -> (2l+1)*2+0 = 4l+2
                    # Z[1,0] -> C[1, 2l] -> k=2l, i=1 -> 2l*2+1 = 4l+1
                    # Z[1,1] -> C[1, 2l+1] -> k=2l+1, i=1 -> (2l+1)*2+1 = 4l+3
                    w_indices = [4*l, 4*l+2, 4*l+1, 4*l+3]
                else:
                    # X = A_R = A[:, 2:4]
                    # X[0,0] = A[0,2] -> U[0*m+2] = U[2]
                    # X[0,1] = A[0,3] -> U[3]
                    # X[1,0] = A[1,2] -> U[4]
                    # X[1,1] = A[1,3] -> U[5]
                    u_indices = [2, 3, 4, 5]
                    # Y = B_{l,R} = B[2:4, 2l:2l+2]
                    # Y[0,0] = B[2, 2l] -> V[2*p + 2l] = V[16+2l]
                    # Y[0,1] = B[2, 2l+1] -> V[16+2l+1]
                    # Y[1,0] = B[3, 2l] -> V[3*p + 2l] = V[24+2l]
                    # Y[1,1] = B[3, 2l+1] -> V[24+2l+1]
                    v_indices = [16+2*l, 16+2*l+1, 24+2*l, 24+2*l+1]
                    # Z contributes to C_l same as before
                    w_indices = [4*l, 4*l+2, 4*l+1, 4*l+3]

                for idx_u, coeff in zip(u_indices, u_coeffs):
                    U[idx_u][col] += coeff
                for idx_v, coeff in zip(v_indices, v_coeffs):
                    V[idx_v][col] += coeff
                for idx_w, coeff in zip(w_indices, w_coeffs):
                    W[idx_w][col] += coeff

                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
