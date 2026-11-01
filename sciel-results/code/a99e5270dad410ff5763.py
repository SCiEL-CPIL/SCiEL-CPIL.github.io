import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix
    by a 2x12 matrix.

    We use the known rank-42 decomposition based on applying Strassen's algorithm
    independently to each 2x2 block of columns in the right-hand matrix. Since the
    right-hand matrix is 2x12, it can be viewed as 6 blocks of 2x2 matrices.
    Each 2x2 x 2x2 multiplication has rank 7 (Strassen's algorithm), so the total
    rank is 6 * 7 = 42.

    The decomposition tensors U, V, W have shapes:
      U: (n*m, rank) = (4, 42)
      V: (m*p, rank) = (24, 42)
      W: (p*n, rank) = (24, 42)

    Entries are integers (in {-1, 0, 1}), which are trivially exact half-integers.

    Trace convention: output factor W is indexed as W[k*n + i, r] where the
    product C = A * B gives C[i, k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    summed appropriately over j.
    """
    n, m, p = DIMENSIONS
    assert n == 2 and m == 2 and p == 12

    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen's 7 multiplication terms for 2x2 x 2x2:
    # For matrices X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]:
    #   M1 = (x00 + x11) * (y00 + y11)
    #   M2 = (x10 + x11) * y00
    #   M3 = x00 * (y01 - y11)
    #   M4 = x11 * (y10 - y00)
    #   M5 = (x00 + x01) * y11
    #   M6 = (x10 - x00) * (y00 + y01)
    #   M7 = (x01 - x11) * (y10 + y11)
    #
    # Result C = X*Y:
    #   c00 = M1 + M4 - M5 + M7
    #   c01 = M3 + M5
    #   c10 = M2 + M4
    #   c11 = M1 - M2 + M3 + M6

    terms = [
        # term 0: M1 = (x00 + x11) * (y00 + y11)
        {'U': {(0, 0): 1, (1, 1): 1},
         'V': {(0, 0): 1, (1, 1): 1},
         'W': {(0, 0): 1, (1, 1): 1}},
        # term 1: M2 = (x10 + x11) * y00
        {'U': {(1, 0): 1, (1, 1): 1},
         'V': {(0, 0): 1},
         'W': {(0, 1): 1, (1, 1): -1}},
        # term 2: M3 = x00 * (y01 - y11)
        {'U': {(0, 0): 1},
         'V': {(0, 1): 1, (1, 1): -1},
         'W': {(1, 0): 1, (1, 1): 1}},
        # term 3: M4 = x11 * (y10 - y00)
        {'U': {(1, 1): 1},
         'V': {(1, 0): 1, (0, 0): -1},
         'W': {(0, 0): 1, (0, 1): 1}},
        # term 4: M5 = (x00 + x01) * y11
        {'U': {(0, 0): 1, (0, 1): 1},
         'V': {(1, 1): 1},
         'W': {(0, 0): -1, (1, 0): 1}},
        # term 5: M6 = (x10 - x00) * (y00 + y01)
        {'U': {(1, 0): 1, (0, 0): -1},
         'V': {(0, 0): 1, (0, 1): 1},
         'W': {(1, 1): 1}},
        # term 6: M7 = (x01 - x11) * (y10 + y11)
        {'U': {(0, 1): 1, (1, 1): -1},
         'V': {(1, 0): 1, (1, 1): 1},
         'W': {(0, 0): 1}},
    ]

    num_blocks = p // 2  # 6 blocks of 2 columns each
    for b in range(num_blocks):
        for t_idx in range(7):
            r = b * 7 + t_idx
            term = terms[t_idx]
            for (i, j), val in term['U'].items():
                U[i * m + j][r] = val
            for (j, k_loc), val in term['V'].items():
                k_glob = b * 2 + k_loc
                V[j * p + k_glob][r] = val
            for (k_loc, i), val in term['W'].items():
                k_glob = b * 2 + k_loc
                W[k_glob * n + i][r] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END


if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    rank = U.shape[1]

    # Verify: random test
    rng = np.random.default_rng(42)
    A = rng.integers(-5, 5, size=(2, 2))
    B = rng.integers(-5, 5, size=(2, 12))
    C_expected = A @ B

    # Compute via tensor decomposition
    # C[i,k] = sum_{j,r} A[i,j] * B[j,k] * (contribution from rank-r)
    # More precisely: C[i,k] = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]?
    # Actually the decomposition is: C = sum_r (col_i_of_U_r @ col_of_V_r @ W_r)
    # Let me just verify by direct multiplication using the rank-1 terms.
    # Each rank-r term contributes: for each (i,j,k), the product term is:
    #   A[i,j] * B[j,k] where the coefficients come from U, V, W
    # The tensor is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal delta structure for matrix multiplication.

    # Reconstruct C from decomposition
    C_recon = np.zeros((2, 12), dtype=int)
    for r in range(rank):
        for i in range(2):
            for j in range(2):
                for k in range(12):
                    term = U[i * 2 + j, r] * V[j * 12 + k, r] * W[k * 2 + i, r]
                    C_recon[i, k] += A[i, j] * B[j, k] * term

    # Actually that's not right. The standard bilinear form is:
    # C[i,k] = sum_j A[i,j] B[j,k], and the rank-r decomposition expresses this as:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    # Wait, that's not standard either. Let me think about this more carefully.
    #
    # The matrix multiplication tensor T_{ijk} (where C_{ik} = sum_j A_{ij} B_{jk})
    # has a rank-r decomposition if we can write:
    # T_{ijk} = sum_r U^r_{ij} V^r_{jk} W^r_{ki}  ... but that's a 3-tensor with indices (i,j,k).
    #
    # Actually the standard form: the bilinear map (A,B) -> A*B has rank r if
    # C_{ik} = sum_{r=1}^{rank} (sum_j U_{ij}^{(r)} A_{ij}) (sum_j V_{jk}^{(r)} B_{jk})
    # where W factors are implicit (or W=1 always). But the task uses a 3-factor form.
    #
    # Given the task's indexing: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    # The decomposition means: C[i,k] = sum_r (U[:,r]^T * A_vec) * (V[:,r]^T * B_vec) * W[:,r][k*n+i]
    # where A_vec = A.flatten() in row-major (i*m+j), B_vec = B.flatten() in row-major (j*p+k).
    #
    # Let me just verify numerically.
    C_recon2 = np.zeros((2, 12))
    for r in range(rank):
        # For each r, compute a scalar a_r = sum_{i,j} U[i*2+j, r] * A[i,j]
        #           and a scalar b_r = sum_{j,k} V[j*12+k, r] * B[j,k]
        # Then C[i,k] += a_r * b_r * W[k*2+i, r]
        a_r = sum(U[i * 2 + j, r] * A[i, j] for i in range(2) for j in range(2))
        b_r = sum(V[j * 12 + k, r] * B[j, k] for j in range(2) for k in range(12))
        for i in range(2):
            for k in range(12):
                C_recon2[i, k] += a_r * b_r * W[k * 2 + i, r]

    assert np.allclose(C_recon2, C_expected), f"Mismatch!\n{C_recon2}\n{C_expected}"
    print(f"OK: rank={rank} verified.")
