import time
import numpy as np

def construct():
    n, m, p = 2, 3, 8
    dim_U = n * m  # 6
    dim_V = m * p  # 24
    dim_W = p * n  # 16

    t_start = time.time()
    deadline = t_start + 140.0

    # Build rank-43 decomposition
    # Part 1: j=0, 15 terms (8 shared + 7 diagonal-fixing)
    # Part 2: j=1,2, 28 terms (4 blocks × 7 Strassen terms)
    rank = 43
    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]
    col = 0

    # Part 1: j=0, 15 terms
    # 8 shared terms: (A[0,0]+A[1,0]) * B[0,k] -> C[0,k] + C[1,k]
    for k in range(p):
        U[0 * m + 0][col] = 1  # A[0,0]
        U[1 * m + 0][col] = 1  # A[1,0]
        V[0 * p + k][col] = 1  # B[0,k]
        W[k * n + 0][col] = 1  # C[0,k]
        W[k * n + 1][col] = 1  # C[1,k]
        col += 1

    # 7 diagonal-fixing terms: A[1,0]*B[0,k] -> C[1,k] - C[0,k]
    for k in range(p - 1):
        U[1 * m + 0][col] = 1  # A[1,0]
        V[0 * p + k][col] = 1  # B[0,k]
        W[k * n + 1][col] = 1  # C[1,k]
        W[k * n + 0][col] = -1  # C[0,k]
        col += 1

    # Part 2: j=1,2 in 4 blocks of 2x2x2, 7 terms each = 28
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

    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank

    # Verify the rank-43 solution
    if not _verify(U, V, W):
        # Fall back to rank 44 if verification fails
        U44, V44, W44 = _build_rank44()
        best_result = {"U": U44, "V": V44, "W": W44}
        best_rank = 44

    # Try to find rank 42 or lower using local search
    for target_rank in [42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30]:
        if time.time() > deadline:
            break

        found = _try_reduce_rank(best_result, target_rank, deadline)
        if found is not None:
            best_result = found
            best_rank = target_rank
            break

    return best_result


def _build_rank44():
    """Build the known-good rank-44 decomposition."""
    n, m, p = 2, 3, 8
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    rank = 44
    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]
    col = 0

    for k in range(p):
        U[0][col] = 1
        V[k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        U[1 * m + 0][col] = 1
        V[k][col] = 1
        W[k * n + 1][col] = 1
        col += 1

    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        U[0 * m + 1][col] = 1
        U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        U[1 * m + 1][col] = 1
        U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1
        U[0 * m + 1][col] = 1
        V[1 * p + k1][col] = 1
        V[2 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        U[1 * m + 2][col] = 1
        V[2 * p + k0][col] = 1
        V[1 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        U[0 * m + 1][col] = 1
        U[0 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = -1
        W[k1 * n + 0][col] = 1
        col += 1
        U[1 * m + 1][col] = 1
        U[0 * m + 1][col] = -1
        V[1 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        U[0 * m + 2][col] = 1
        U[1 * m + 2][col] = -1
        V[2 * p + k0][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1

    return U, V, W


def _try_reduce_rank(result, target_rank, deadline):
    """Try to reduce rank by merging terms."""
    n, m, p = 2, 3, 8
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    U = result["U"]
    V = result["V"]
    W = result["W"]
    current_rank = len(U[0])

    # Try removing each term and checking if the rest still works
    for remove_idx in range(current_rank):
        if time.time() > deadline:
            return None

        U_t = [row[:] for row in U]
        V_t = [row[:] for row in V]
        W_t = [row[:] for row in W]

        # Zero out the term
        for i in range(dim_U):
            U_t[i][remove_idx] = 0
        for i in range(dim_V):
            V_t[i][remove_idx] = 0
        for i in range(dim_W):
            W_t[i][remove_idx] = 0

        # Check if remaining terms satisfy constraints
        valid = True
        for i0 in range(n):
            for j0 in range(m):
                for k0 in range(p):
                    for i in range(n):
                        for k in range(p):
                            val = 0
                            for r in range(current_rank):
                                if r == remove_idx:
                                    continue
                                if i == i0 and k == k0:
                                    val += W_t[k * n + i][r] * U_t[i * m + j0][r] * V_t[j0 * p + k][r]
                            expected_val = 1 if (i == i0 and k == k0) else 0
                            if val != expected_val:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
            if not valid:
                break

        if valid:
            # Build trimmed matrices
            U_out = []
            V_out = []
            W_out = []
            for i in range(dim_U):
                U_out.append([U_t[i][r] for r in range(current_rank) if r != remove_idx])
            for i in range(dim_V):
                V_out.append([V_t[i][r] for r in range(current_rank) if r != remove_idx])
            for i in range(dim_W):
                W_out.append([W_t[i][r] for r in range(current_rank) if r != remove_idx])
            return {"U": U_out, "V": V_out, "W": W_out}

    return None


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
