"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 150.0 - 5.0  # safety margin

    n, m, p = DIMENSIONS

    # Block decomposition: split p=10 into 5 blocks of size 2.
    # Each block computes a 2x3 @ 3x2 product.
    # Per block: Strassen rank-7 for the 2x2 core (cols 0,1 of A; rows 0,1 of B)
    #            + rank-4 outer product for the third column of A / row of B.
    # Total rank = 5 * (7 + 4) = 55.

    num_blocks = 5
    block_size = 2
    rank_per_block = 11
    total_rank = num_blocks * rank_per_block

    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)

    # Strassen rank-7 for 2x2x2.
    # A: i in {0,1}, j in {0,1}; B: j in {0,1}, k in {0,1}; C: i in {0,1}, k in {0,1}
    # W index convention: k*n + i  (i.e., W[k, i] -> W_flat[k*n + i])
    #
    # Standard Strassen for 2x2:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # Each term: list of (U_index, coeff), (V_index, coeff), (W_index, coeff)
    # U index = i*m + j ; V index = j*p + k ; W index = k*n + i
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11) -> C00 +=, C11 -=
        {
            'U': [(0 * m + 0, 1), (1 * m + 1, 1)],
            'V': [(0 * p + 0, 1), (1 * p + 1, 1)],
            'W': [(0 * n + 0, 1), (1 * n + 1, -1)]
        },
        # M2 = (A10+A11)B00 -> C10 +=, C11 -=
        {
            'U': [(1 * m + 0, 1), (1 * m + 1, 1)],
            'V': [(0 * p + 0, 1)],
            'W': [(0 * n + 1, 1), (1 * n + 1, -1)]
        },
        # M3 = A00(B01-B11) -> C01 +=, C11 +=
        {
            'U': [(0 * m + 0, 1)],
            'V': [(0 * p + 1, 1), (1 * p + 1, -1)],
            'W': [(1 * n + 0, 1), (1 * n + 1, 1)]
        },
        # M4 = A11(B10-B00) -> C00 +=, C10 +=
        {
            'U': [(1 * m + 1, 1)],
            'V': [(1 * p + 0, 1), (0 * p + 0, -1)],
            'W': [(0 * n + 0, 1), (0 * n + 1, 1)]
        },
        # M5 = (A00+A01)B11 -> C00 -=, C01 +=
        {
            'U': [(0 * m + 0, 1), (0 * m + 1, 1)],
            'V': [(1 * p + 1, 1)],
            'W': [(0 * n + 0, -1), (1 * n + 0, 1)]
        },
        # M6 = (A10-A00)(B00+B01) -> C11 +=
        {
            'U': [(1 * m + 0, 1), (0 * m + 0, -1)],
            'V': [(0 * p + 0, 1), (0 * p + 1, 1)],
            'W': [(1 * n + 1, 1)]
        },
        # M7 = (A01-A11)(B10+B11) -> C00 +=
        {
            'U': [(0 * m + 1, 1), (1 * m + 1, -1)],
            'V': [(1 * p + 0, 1), (1 * p + 1, 1)],
            'W': [(0 * n + 0, 1)]
        }
    ]

    # Outer product part (4 terms) for A[:, 2] and B[2, :]
    # C[i, k] += A[i, 2] * B[2, k]
    outer_terms = []
    for i in range(2):
        for k in range(2):
            outer_terms.append({
                'U': [(i * m + 2, 1)],
                'V': [(2 * p + k, 1)],
                'W': [(k * n + i, 1)]
            })

    all_terms = strassen_terms + outer_terms

    col = 0
    for b in range(num_blocks):
        for term in all_terms:
            for u_idx, coeff in term['U']:
                U[u_idx, col] += coeff
            for v_idx, coeff in term['V']:
                # v_idx is in the local 3x2 block; map k_local to k_global
                # v_idx = j*p + k_local  where k_local in {0,1}
                j = v_idx // p
                k_local = v_idx % p
                k_global = b * block_size + k_local
                V[j * p + k_global, col] += coeff
            for w_idx, coeff in term['W']:
                # w_idx = k_local * n + i
                k_local = w_idx // n
                i = w_idx % n
                k_global = b * block_size + k_local
                W[k_global * n + i, col] += coeff
            col += 1

    # Verify the decomposition exactly (with deadline guard).
    # The multiplication tensor T[i,j,k] = 1 if C[i,k] = sum_j A[i,j]*B[j,k]
    # Decomposition: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all (i,j,k)
    # We verify by checking: for each r, the rank-1 tensor U_r ⊗ V_r ⊗ W_r
    # and summing them equals the multiplication tensor.

    if time.time() < DEADLINE:
        # Build the full multiplication tensor T of shape (n*m, m*p, p*n)
        # T[i,j,k] where i in [0,n*m), j in [0,m*p), k in [0,p*n)
        # T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute
        # to the (i,k) entry of C, i.e., if the j-index of A matches the j-index of B.
        # A index: i = a_i * m + a_j  (a_i in [0,n), a_j in [0,m))
        # B index: j = b_j * p + b_k  (b_j in [0,m), b_k in [0,p))
        # C index: k = c_k * n + c_i  (c_k in [0,p), c_i in [0,n))
        # Contribution exists if a_j == b_j, a_i == c_i, and b_k == c_k.
        T = np.zeros((n * m, m * p, p * n), dtype=float)
        for a_i in range(n):
            for a_j in range(m):
                for b_j in range(m):
                    if a_j != b_j:
                        continue
                    for b_k in range(p):
                        T[a_i * m + a_j, b_j * p + b_k, b_k * n + a_i] = 1.0

        # Compute the decomposition tensor D[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
        D = np.einsum('ijr,jkr,kir->ijk', U, V, W)

        # Check exactness
        err = np.max(np.abs(D - T))
        if err > 1e-9:
            # Fallback: return the schoolbook decomposition (always valid, rank 60)
            rank_sb = n * m * p
            U_sb = np.zeros((n * m, rank_sb), dtype=float)
            V_sb = np.zeros((m * p, rank_sb), dtype=float)
            W_sb = np.zeros((p * n, rank_sb), dtype=float)
            col = 0
            for a_i in range(n):
                for a_j in range(m):
                    for b_k in range(p):
                        U_sb[a_i * m + a_j, col] = 1.0
                        V_sb[a_j * p + b_k, col] = 1.0
                        W_sb[b_k * n + a_i, col] = 1.0
                        col += 1
            return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
