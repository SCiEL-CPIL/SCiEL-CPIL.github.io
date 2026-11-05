"""Rank-55 tensor decomposition for 2x3 @ 3x10 matrix multiplication."""
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    n, m, p = 2, 3, 10

    # Split m=3 into 2+1:
    # Part 1: A[:,0] * B[0,:] -> 2x1x10, rank 20 (schoolbook)
    # Part 2: A[:,1:3] * B[1:3,:] -> 2x2x10, split into 5 blocks of 2x2x2, rank 7 each (Strassen)
    # Total rank = 20 + 5*7 = 55

    rank = 55
    U = [[0] * rank for _ in range(n * m)]   # 6 x 55
    V = [[0] * rank for _ in range(m * p)]   # 30 x 55
    W = [[0] * rank for _ in range(p * n)]   # 20 x 55

    col = 0

    # Part 1: A[:,0] * B[0,:] schoolbook
    # U index: i*m + 0 = i*3
    # V index: 0*p + k = k
    # W index: k*n + i = k*2 + i
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] += 1
            V[0 * p + k][col] += 1
            W[k * n + i][col] += 1
            col += 1

    # Part 2: A[:,1:3] * B[1:3,:] via 5 Strassen blocks
    # A indices in U:
    #   A[0,1] -> 0*3+1 = 1
    #   A[0,2] -> 0*3+2 = 2
    #   A[1,1] -> 1*3+1 = 4
    #   A[1,2] -> 1*3+2 = 5
    # B indices in V:
    #   B[1,k0] -> 1*10+k0
    #   B[1,k1] -> 1*10+k1
    #   B[2,k0] -> 2*10+k0
    #   B[2,k1] -> 2*10+k1
    # C indices in W:
    #   C[0,k0] -> k0*2+0
    #   C[0,k1] -> k1*2+0
    #   C[1,k0] -> k0*2+1
    #   C[1,k1] -> k1*2+1

    a00 = 1  # A[0,1]
    a01 = 2  # A[0,2]
    a10 = 4  # A[1,1]
    a11 = 5  # A[1,2]

    for j in range(5):
        k0 = 2 * j
        k1 = 2 * j + 1

        b00 = 1 * p + k0  # B[1,k0]
        b01 = 1 * p + k1  # B[1,k1]
        b10 = 2 * p + k0  # B[2,k0]
        b11 = 2 * p + k1  # B[2,k1]

        c00 = k0 * n + 0  # C[0,k0]
        c01 = k1 * n + 0  # C[0,k1]
        c10 = k0 * n + 1  # C[1,k0]
        c11 = k1 * n + 1  # C[1,k1]

        # M1: (A00 + A11) * (B00 + B11)
        U[a00][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        W[c11][col] += 1
        col += 1

        # M2: (A10 + A11) * B00
        U[a10][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        W[c10][col] += 1
        W[c11][col] -= 1
        col += 1

        # M3: A00 * (B01 - B11)
        U[a00][col] += 1
        V[b01][col] += 1
        V[b11][col] -= 1
        W[c01][col] += 1
        W[c11][col] += 1
        col += 1

        # M4: A11 * (B10 - B00)
        U[a11][col] += 1
        V[b10][col] += 1
        V[b00][col] -= 1
        W[c00][col] += 1
        W[c10][col] += 1
        col += 1

        # M5: (A00 + A01) * B11
        U[a00][col] += 1
        U[a01][col] += 1
        V[b11][col] += 1
        W[c00][col] -= 1
        W[c01][col] += 1
        col += 1

        # M6: (A10 - A00) * (B00 + B01)
        U[a10][col] += 1
        U[a00][col] -= 1
        V[b00][col] += 1
        V[b01][col] += 1
        W[c11][col] += 1
        col += 1

        # M7: (A01 - A11) * (B10 + B11)
        U[a01][col] += 1
        U[a11][col] -= 1
        V[b10][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        col += 1

        # Check deadline
        if time.time() > deadline:
            break

    return {'U': U, 'V': V, 'W': W}
# EVOLVE-BLOCK-END
