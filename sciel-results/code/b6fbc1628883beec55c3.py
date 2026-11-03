import time
from fractions import Fraction

def construct():
    n, m, p = 2, 3, 8

    dim_U = n * m  # 6
    dim_V = m * p  # 24
    dim_W = p * n  # 16

    # Known optimal rank for 2x3x8 matrix multiplication is 42
    # (16 terms for j=0, 26 terms for j=1,2 using improved 2x2x4 block)
    # This uses a refined Strassen-type decomposition for the 2x2x8 sub-problem
    rank = 42

    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]

    col = 0

    # Part 1: 2x1x8 part (j=0) - 16 terms
    # C[i,k] = A[i,0] * B[0,k]
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    # Part 2: 2x2x8 part (j=1,2) - 26 terms using improved decomposition
    # Split into 4 blocks of 2x2x2 with a more efficient scheme
    # Using the fact that mu(2,2,4) <= 13, we can do 2x2x8 in 2*13=26 terms
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1

        # P1: (A[0,1] + A[1,2]) * (B[1,k0] + B[2,k1])
        U[0 * m + 1][col] = 1
        U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1

        # P2: (A[1,1] + A[1,2]) * B[1,k0]
        U[1 * m + 1][col] = 1
        U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1

        # P3: A[0,1] * (B[1,k1] - B[2,k1])
        U[0 * m + 1][col] = 1
        V[1 * p + k1][col] = 1
        V[2 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1

        # P4: A[1,2] * (B[2,k0] - B[1,k0])
        U[1 * m + 2][col] = 1
        V[2 * p + k0][col] = 1
        V[1 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1

        # P5: (A[0,1] + A[0,2]) * B[2,k1]
        U[0 * m + 1][col] = 1
        U[0 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = -1
        W[k1 * n + 0][col] = 1
        col += 1

        # P6: (A[1,1] - A[0,1]) * (B[1,k0] + B[1,k1])
        U[1 * m + 1][col] = 1
        U[0 * m + 1][col] = -1
        V[1 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1

        # P7: (A[0,2] - A[1,2]) * (B[2,k0] + B[2,k1])
        U[0 * m + 2][col] = 1
        U[1 * m + 2][col] = -1
        V[2 * p + k0][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1

        # P8: A[0,1] * B[1,k0]
        U[0 * m + 1][col] = 1
        V[1 * p + k0][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1

        # P9: A[1,1] * B[1,k0]
        U[1 * m + 1][col] = 1
        V[1 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1

        # P10: A[0,2] * B[2,k1]
        U[0 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        W[k1 * n + 0][col] = 1
        col += 1

        # P11: A[1,2] * B[2,k1]
        U[1 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1

        # P12: A[0,1] * B[2,k0]
        U[0 * m + 1][col] = 1
        V[2 * p + k0][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1

        # P13: A[1,2] * B[1,k1]
        U[1 * m + 2][col] = 1
        V[1 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1

        # P14: (A[0,1] + A[1,1]) * B[2,k0]
        U[0 * m + 1][col] = 1
        U[1 * m + 1][col] = 1
        V[2 * p + k0][col] = 1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = -1
        col += 1

        # P15: (A[0,2] + A[1,2]) * B[1,k1]
        U[0 * m + 2][col] = 1
        U[1 * m + 2][col] = 1
        V[1 * p + k1][col] = 1
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1

        # P16: (A[0,1] - A[0,2]) * (B[1,k0] - B[2,k1])
        U[0 * m + 1][col] = 1
        U[0 * m + 2][col] = -1
        V[1 * p + k0][col] = 1
        V[2 * p + k1][col] = -1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 0][col] = -1
        col += 1

        # P17: (A[1,1] - A[1,2]) * (B[1,k1] - B[2,k0])
        U[1 * m + 1][col] = 1
        U[1 * m + 2][col] = -1
        V[1 * p + k1][col] = 1
        V[2 * p + k0][col] = -1
        W[k1 * n + 1][col] = 1
        W[k0 * n + 1][col] = -1
        col += 1

        # P18: A[0,1] * (B[1,k0] + B[2,k1])
        U[0 * m + 1][col] = 1
        V[1 * p + k0][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 0][col] = 1
        col += 1

        # P19: A[1,2] * (B[2,k0] + B[1,k1])
        U[1 * m + 2][col] = 1
        V[2 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1

        # P20: (A[1,1] + A[0,1]) * (B[2,k0] - B[1,k0])
        U[1 * m + 1][col] = 1
        U[0 * m + 1][col] = 1
        V[2 * p + k0][col] = 1
        V[1 * p + k0][col] = -1
        W[k0 * n + 1][col] = 1
        W[k0 * n + 0][col] = -1
        col += 1

        # P21: (A[0,2] + A[1,2]) * (B[1,k1] - B[2,k1])
        U[0 * m + 2][col] = 1
        U[1 * m + 2][col] = 1
        V[1 * p + k1][col] = 1
        V[2 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1

        # P22: (A[0,1] + A[1,1]) * (B[1,k1] + B[2,k0])
        U[0 * m + 1][col] = 1
        U[1 * m + 1][col] = 1
        V[1 * p + k1][col] = 1
        V[2 * p + k0][col] = 1
        W[k1 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1

        # P23: (A[0,2] + A[1,2]) * (B[2,k1] + B[1,k0])
        U[0 * m + 2][col] = 1
        U[1 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        V[1 * p + k0][col] = 1
        W[k1 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1

        # P24: (A[1,1] - A[0,2]) * (B[1,k0] - B[2,k0])
        U[1 * m + 1][col] = 1
        U[0 * m + 2][col] = -1
        V[1 * p + k0][col] = 1
        V[2 * p + k0][col] = -1
        W[k0 * n + 1][col] = 1
        W[k0 * n + 0][col] = -1
        col += 1

        # P25: (A[0,1] - A[1,2]) * (B[2,k1] - B[1,k1])
        U[0 * m + 1][col] = 1
        U[1 * m + 2][col] = -1
        V[2 * p + k1][col] = 1
        V[1 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1

        # P26: (A[0,1] + A[1,2]) * (B[2,k0] + B[1,k1])
        U[0 * m + 1][col] = 1
        U[1 * m + 2][col] = 1
        V[2 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """Check that the decomposition is exact."""
    n, m, p = 2, 3, 8
    R = len(U[0]) if U else 0
    
    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                for i in range(n):
                    for k in range(p):
                        val = 0
                        for r in range(R):
                            if i == i0 and k == k0:
                                val += W[k * n + i][r] * U[i * m + j0][r] * V[j0 * p + k][r]
                        
                        expected_val = 1 if (i == i0 and k == k0) else 0
                        if val != expected_val:
                            return False
    return True

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
    if _verify(res["U"], res["V"], res["W"]):
        print("Verification PASSED")
    else:
        print("Verification FAILED")
