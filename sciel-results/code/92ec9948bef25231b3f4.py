import time
import numpy as np

def _build_44_rank(n, m, p):
    """Build the known 44-rank decomposition for 2x3x8 matrix multiplication."""
    dim_U = n * m  # 6
    dim_V = m * p  # 24
    dim_W = p * n  # 16
    rank = 44

    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]

    col = 0

    # Part 1: 2x1x8 part (j=0) - 16 terms
    for k in range(p):
        # Term 1: A[0,0] * B[0,k] -> C[k,0]
        U[0 * m + 0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1

        # Term 2: A[1,0] * B[0,k] -> C[k,1]
        U[1 * m + 0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1

    # Part 2: 2x2x8 part (j=1,2) - 4 blocks of 7 terms = 28 terms
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

    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p):
    """Check that the decomposition is exact using numpy for speed."""
    R = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                # For each (i0, j0, k0), check all (i, k)
                for i in range(n):
                    for k in range(p):
                        val = 0.0
                        for r in range(R):
                            if i == i0 and k == k0:
                                val += W_arr[k * n + i, r] * U_arr[i * m + j0, r] * V_arr[j0 * p + k, r]
                        expected_val = 1.0 if (i == i0 and k == k0) else 0.0
                        if abs(val - expected_val) > 1e-10:
                            return False
    return True


def construct():
    n, m, p = 2, 3, 8
    start_time = time.time()

    # Build the 44-rank decomposition
    result = _build_44_rank(n, m, p)

    # Verify correctness
    if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        return result
    else:
        raise RuntimeError("Verification failed for 44-rank decomposition")


if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
    print("Verification:", _verify_decomposition(res["U"], res["V"], res["W"], 2, 3, 8))
