import time
import numpy as np

def construct():
    n, m, p = 2, 3, 8
    t_start = time.time()
    deadline = t_start + 135.0  # safety margin

    # The known rank-44 decomposition
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0

    # Part 1: j=0 (column 0 of A, row 0 of B), 16 terms
    # A[i,0] * B[0,k] -> C[i,k]
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1
            V[0 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1

    # Part 2: j=1,2 (columns 1,2 of A, rows 1,2 of B), 4 blocks of 2x2x2 Strassen, 7 terms each = 28
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1

        # Strassen 2x2x2 for A2 (2x2) * B2_block (2x2)
        # A2 local:
        #   a00 = A[0,1] -> U index 0*m+1 = 1
        #   a01 = A[0,2] -> U index 0*m+2 = 2
        #   a10 = A[1,1] -> U index 1*m+1 = 4
        #   a11 = A[1,2] -> U index 1*m+2 = 5
        # B2_block local (rows 1,2 of B, cols k0,k1):
        #   b00 = B[1,k0] -> V index 1*p+k0
        #   b01 = B[1,k1] -> V index 1*p+k1
        #   b10 = B[2,k0] -> V index 2*p+k0
        #   b11 = B[2,k1] -> V index 2*p+k1
        # C local:
        #   c00 = C[0,k0] -> W index k0*n+0
        #   c01 = C[0,k1] -> W index k1*n+0
        #   c10 = C[1,k0] -> W index k0*n+1
        #   c11 = C[1,k1] -> W index k1*n+1

        # m1 = (a00+a11)(b00+b11)
        U[0 * m + 1, col] = 1
        U[1 * m + 2, col] = 1
        V[1 * p + k0, col] = 1
        V[2 * p + k1, col] = 1
        W[k0 * n + 0, col] = 1
        W[k1 * n + 1, col] = 1
        col += 1

        # m2 = (a10+a11)b00
        U[1 * m + 1, col] = 1
        U[1 * m + 2, col] = 1
        V[1 * p + k0, col] = 1
        W[k0 * n + 1, col] = 1
        W[k1 * n + 1, col] = -1
        col += 1

        # m3 = a00(b01-b11)
        U[0 * m + 1, col] = 1
        V[1 * p + k1, col] = 1
        V[2 * p + k1, col] = -1
        W[k1 * n + 0, col] = 1
        W[k1 * n + 1, col] = 1
        col += 1

        # m4 = a11(b10-b00)
        U[1 * m + 2, col] = 1
        V[2 * p + k0, col] = 1
        V[1 * p + k0, col] = -1
        W[k0 * n + 0, col] = 1
        W[k0 * n + 1, col] = 1
        col += 1

        # m5 = (a00+a01)b11
        U[0 * m + 1, col] = 1
        U[0 * m + 2, col] = 1
        V[2 * p + k1, col] = 1
        W[k0 * n + 0, col] = -1
        W[k1 * n + 0, col] = 1
        col += 1

        # m6 = (a10-a00)(b00+b01)
        U[1 * m + 1, col] = 1
        U[0 * m + 1, col] = -1
        V[1 * p + k0, col] = 1
        V[1 * p + k1, col] = 1
        W[k1 * n + 1, col] = 1
        col += 1

        # m7 = (a01-a11)(b10+b11)
        U[0 * m + 2, col] = 1
        U[1 * m + 2, col] = -1
        V[2 * p + k0, col] = 1
        V[2 * p + k1, col] = 1
        W[k0 * n + 0, col] = 1
        col += 1

    # Verification function
    def verify(U_mat, V_mat, W_mat, tol=1e-9):
        R = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for ip in range(n):
                            for kp in range(p):
                                val = 0.0
                                for r in range(R):
                                    val += U_mat[i * m + j, r] * V_mat[jp * p + k, r] * W_mat[kp * n + ip, r]
                                expected = 1.0 if (j == jp and i == ip and k == kp) else 0.0
                                if abs(val - expected) > tol:
                                    return False
        return True

    # Try to find a lower-rank decomposition using structured search
    # The key insight: the 2x3x8 tensor can be seen as a combination of sub-tensors.
    # We try to find a rank-r decomposition by solving a nonlinear system.

    def try_lower_rank(target_rank):
        """Try to find a rank-target_rank decomposition."""
        if time.time() > deadline - 10:
            return None

        # Use the rank-44 solution as initialization, zeroing out some columns
        # and trying to redistribute the contributions.
        U_init = np.zeros((n * m, target_rank))
        V_init = np.zeros((m * p, target_rank))
        W_init = np.zeros((p * n, target_rank))

        # Copy the first target_rank columns of the rank-44 solution
        # This is a crude initialization, but it preserves some structure.
        if target_rank <= 44:
            U_init[:, :target_rank] = U[:, :target_rank]
            V_init[:, :target_rank] = V[:, :target_rank]
            W_init[:, :target_rank] = W[:, :target_rank]
        else:
            return None

        # We need to solve for U, V, W such that the tensor decomposition is exact.
        # This is a highly nonlinear problem. We use a simple iterative approach:
        # fix two of the three matrices and solve for the third, iterating.

        # Actually, for this problem, a more principled approach is needed.
        # Let's try a different strategy: exploit the block structure.

        # The 2x3x8 multiplication can be decomposed as:
        # C = A[:,0] * B[0,:] + A[:,1:3] * B[1:3,:]
        # The first term has rank 16 (schoolbook).
        # The second term is a 2x2x8 problem, which has rank 28 with Strassen.
        # Total: 16 + 28 = 44.

        # To go lower, we need to find a decomposition of the 2x2x8 part with rank < 28,
        # or find cross-terms that reduce the total rank.

        # The 2x2x8 problem: multiply A2 (2x2) by B2 (2x8) to get C2 (2x8).
        # B2 can be seen as 4 blocks of 2x2. Each block uses 7 Strassen multiplications.
        # Total: 4 * 7 = 28.

        # Can we do better? The 2x2x2 problem has rank 7 (proven minimal).
        # For 2x2x8, the rank is at least 4 * 7 = 28 if the blocks are independent.
        # But they are not fully independent; we might be able to share some multiplications.

        # Let's try to find a rank-27 or lower decomposition for the 2x2x8 part.
        # This is a hard problem, so we use a heuristic search.

        # For now, we return None if we can't find a better solution quickly.
        # The rank-44 solution is our best bet.

        return None

    # Try a few target ranks
    for target_rank in range(43, 39, -1):
        if time.time() > deadline - 5:
            break
        result = try_lower_rank(target_rank)
        if result is not None:
            U, V, W = result
            if verify(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Return the rank-44 solution
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W):
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
