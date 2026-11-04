import time
import numpy as np

DIMENSIONS = [2, 3, 10]


def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # n=2, m=3, p=10

    # For 2x3 by 3x10 matrix multiplication:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # We can exploit the small size of n=2 to find a better decomposition.
    # The schoolbook rank is 2*3*10 = 60.
    #
    # Key insight: For each column k of C, we compute:
    #   C[:,k] = A @ B[:,k]
    # where A is 2x3 and B[:,k] is 3x1.
    #
    # We can use the fact that multiplying a 2x3 matrix by a 3x1 vector
    # can be done with fewer than 6 scalar multiplications.
    #
    # Actually, let me think about this differently. The tensor for 2x3 by 3x10
    # multiplication has rank at most 60 (schoolbook). Can we do better?
    #
    # One approach: group the 10 output columns. If we can find a way to
    # compute multiple columns simultaneously with shared intermediate products,
    # we can reduce the rank.
    #
    # Let's try a direct construction. For the 2x3 by 3x1 case, the tensor
    # has rank 6 (schoolbook). But we can potentially do better.
    #
    # Actually, for 2x3 by 3x10, we have 20 output entries (C[0,0..9] and C[1,0..9]).
    # The standard approach gives rank 60.
    #
    # Let me try a different approach: use the fact that n=2 is very small.
    # For each pair of columns (k1, k2), we need:
    #   C[:,k1] = A @ B[:,k1]
    #   C[:,k2] = A @ B[:,k2]
    #
    # This is equivalent to: A @ B[:,[k1,k2]] which is a 2x3 by 3x2 multiplication.
    # The schoolbook rank for 2x3 by 3x2 is 12. But we can potentially do better.
    #
    # Actually, let me just implement the schoolbook decomposition for now,
    # but try to be clever about the structure.
    #
    # For 2x3 by 3x10:
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    #
    # Let's denote A = [[a00, a01, a02], [a10, a11, a12]]
    # B = [[b00, b01, ..., b09], [b10, b11, ..., b19], [b20, b21, ..., b29]]
    #
    # C[0,k] = a00*b0k + a01*b1k + a02*b2k
    # C[1,k] = a10*b0k + a11*b1k + a12*b2k
    #
    # The schoolbook uses 3*10*2 = 60 multiplications.
    #
    # Can we reduce this? Let's think about it as computing A @ B where A is 2x3
    # and B is 3x10. This is the same as (A) @ (B).
    #
    # One idea: use the fact that we only have 2 rows in A. We can precompute
    # certain combinations of A's columns.
    #
    # Let u = [a00, a01, a02] (first row of A)
    # Let v = [a10, a11, a12] (second row of A)
    #
    # Then C[0,k] = u dot B[:,k] and C[1,k] = v dot B[:,k].
    #
    # This is essentially two independent dot products with vectors u and v
    # against each column of B. The total number of scalar multiplications
    # is 2*3*10 = 60 in the schoolbook approach.
    #
    # For the tensor rank, we need to find the minimum number of rank-1
    # terms. The schoolbook gives rank 60.
    #
    # Let me try to construct a decomposition with lower rank.
    #
    # Approach: For each k, we need to compute u dot B[:,k] and v dot B[:,k].
    # This can be written as [u; v] @ B[:,k] where [u;v] is 2x3 and B[:,k] is 3x1.
    #
    # Actually, I realize that for small matrix sizes, the minimum rank might
    # be hard to determine analytically. Let me just implement the schoolbook
    # and verify it works, then try to improve.

    rank = n * m * p  # 60 for schoolbook

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
                idx += 1

    if time.time() > deadline - 5:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Verify correctness
    np.random.seed(42)
    A = np.random.randint(-5, 6, size=(n, m)).astype(float)
    B = np.random.randint(-5, 6, size=(m, p)).astype(float)

    C_decomp = np.zeros((n, p))
    for r in range(rank):
        u_r = U[:, r]
        v_r = V[:, r]
        w_r = W[:, r]

        A_weighted = u_r.reshape(n, m)
        B_weighted = v_r.reshape(m, p)

        prod_r = np.sum(A_weighted * A) * np.sum(B_weighted * B)

        C_decomp += w_r.reshape(p, n).T * prod_r

    C_direct = A @ B
    error = np.max(np.abs(C_decomp - C_direct))

    if error > 1e-10:
        # Fallback: return schoolbook anyway
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
