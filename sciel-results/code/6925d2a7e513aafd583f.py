import numpy as np
import time

def construct():
    n, m, p = 6, 6, 6
    deadline = time.time() + 110.0

    # Build target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(p):
            for k in range(m):
                T[i * m + k, k * p + j, j * n + i] = 1.0

    # Trivial decomposition (rank 216) as fallback
    rank_triv = n * m * p
    U_triv = np.zeros((n * m, rank_triv), dtype=float)
    V_triv = np.zeros((m * p, rank_triv), dtype=float)
    W_triv = np.zeros((p * n, rank_triv), dtype=float)
    idx = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U_triv[i * m + k, idx] = 1.0
                V_triv[k * p + j, idx] = 1.0
                W_triv[j * n + i, idx] = 1.0
                idx += 1

    # Block-Strassen with block size 2: 3x3 blocks, rank 189
    bs = 2
    nb = n // bs  # 3

    # Strassen 2x2 definitions (7 multiplications)
    strassen_defs = [
        {'A': [(0, 1.0), (3, 1.0)], 'B': [(0, 1.0), (3, 1.0)]},
        {'A': [(2, 1.0), (3, 1.0)], 'B': [(0, 1.0)]},
        {'A': [(0, 1.0)], 'B': [(1, 1.0), (3, -1.0)]},
        {'A': [(3, 1.0)], 'B': [(2, 1.0), (0, -1.0)]},
        {'A': [(0, 1.0), (1, 1.0)], 'B': [(3, 1.0)]},
        {'A': [(2, 1.0), (0, -1.0)], 'B': [(0, 1.0), (1, 1.0)]},
        {'A': [(1, 1.0), (3, -1.0)], 'B': [(2, 1.0), (3, 1.0)]},
    ]
    recon_coeffs = [
        [(0, 1.0), (3, 1.0), (4, -1.0), (6, 1.0)],
        [(2, 1.0), (4, 1.0)],
        [(1, 1.0), (3, 1.0)],
        [(0, 1.0), (1, -1.0), (2, 1.0), (5, 1.0)],
    ]

    rank_bs = 189
    U2 = np.zeros((n * m, rank_bs), dtype=float)
    V2 = np.zeros((m * p, rank_bs), dtype=float)
    W2 = np.zeros((p * n, rank_bs), dtype=float)

    r_idx = 0
    for i_b in range(nb):
        for k_b in range(nb):
            for j_b in range(nb):
                for s in range(7):
                    s_def = strassen_defs[s]
                    for block_idx, coeff_a in s_def['A']:
                        ar = block_idx // 2
                        ac = block_idx % 2
                        i_glob = i_b * bs + ar
                        k_glob = k_b * bs + ac
                        U2[i_glob * m + k_glob, r_idx] += coeff_a
                    for block_idx, coeff_b in s_def['B']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        k_glob = k_b * bs + br
                        j_glob = j_b * bs + bc
                        V2[k_glob * p + j_glob, r_idx] += coeff_b
                    for c_block_idx in range(4):
                        gamma = 0.0
                        for (s_idx, g) in recon_coeffs[c_block_idx]:
                            if s_idx == s:
                                gamma = g
                                break
                        if gamma != 0.0:
                            cr = c_block_idx // 2
                            cc = c_block_idx % 2
                            i_glob = i_b * bs + cr
                            j_glob = j_b * bs + cc
                            W2[j_glob * n + i_glob, r_idx] += gamma
                    r_idx += 1

    # Verify Block-Strassen
    T_recon = np.einsum('ir,jr,kr->ijk', U2, V2, W2)
    if not np.allclose(T_recon, T, atol=1e-9):
        return {"U": U_triv.tolist(), "V": V_triv.tolist(), "W": W_triv.tolist()}

    # Rank reduction: find and remove linearly dependent columns
    # The key insight: columns of U, V, W that are linearly dependent
    # can be removed if we can redistribute their contributions.
    # 
    # A cleaner approach: use the fact that the tensor has symmetries.
    # For the 6x6 case with 3x3 blocks of 2x2, there are inherent
    # linear dependencies because the Strassen formulas for different
    # block positions produce correlated factors.
    #
    # Strategy: Compute the effective rank by checking if we can
    # express some columns as linear combinations of others.
    # This is equivalent to finding the rank of the "flattened" tensor.
    #
    # Practical approach: Try to reduce rank by checking linear
    # independence of the 189 rank-1 terms.
    #
    # For each rank-1 term r, we have vectors u_r, v_r, w_r.
    # The term is redundant if there exist coefficients c_r such that
    # sum_r c_r * (u_r ⊗ v_r ⊗ w_r) = 0 and c_r != 0 for some r.
    #
    # This is hard in general. Instead, let's try a simpler approach:
    # Check if any pair of rank-1 terms are linearly dependent
    # (i.e., one is a scalar multiple of the other in all three factors).

    # Check for exact duplicates or scalar multiples
    # Build a hash of each term's support pattern
    reduced_U = [U2[:, r] for r in range(rank_bs)]
    reduced_V = [V2[:, r] for r in range(rank_bs)]
    reduced_W = [W2[:, r] for r in range(rank_bs)]

    # Find terms that are exact negatives or zero
    keep = [True] * rank_bs
    for r in range(rank_bs):
        if np.allclose(U2[:, r], 0) and np.allclose(V2[:, r], 0) and np.allclose(W2[:, r], 0):
            keep[r] = False

    # More sophisticated: check if term r can be expressed as combination of others
    # This is expensive, so let's try a different approach.
    #
    # Actually, for this specific structure, let's analyze the dependencies.
    # The 27 block positions each contribute 7 Strassen terms.
    # The U factor for block (i_b, k_b) and Strassen step s depends only on
    # (i_b, k_b, s). Similarly V depends on (k_b, j_b, s) and W on (i_b, j_b, s).
    #
    # Key observation: For fixed s, the U factors across different (i_b, k_b)
    # are independent (different rows). Same for V and W.
    # So there are no obvious linear dependencies between different blocks.
    #
    # However, within the same (i_b, k_b, j_b) block, the 7 Strassen terms
    # might have dependencies. Let's check.

    # For a single (i_b, k_b, j_b) block, check if the 7 terms are independent
    # by examining the 7x7 matrix of inner products.
    # Actually, the 7 terms are linearly independent by construction of Strassen.
    #
    # Let me try a different approach: use numerical rank detection on
    # the combined factor matrices.
    #
    # The tensor T = sum_r u_r ⊗ v_r ⊗ w_r has rank <= 189.
    # The actual rank is the minimum number of terms needed.
    #
    # For 6x6 matrix multiplication, the known minimum rank is 216 (trivial)
    # but with Strassen-like methods we can do better.
    #
    # Actually, let me just try to reduce by removing terms that are
    # linearly dependent in a weaker sense: if two terms have the same
    # support pattern in U and V, they might be combinable.

    # Let's try: group terms by their U and V support patterns
    from collections import defaultdict
    groups = defaultdict(list)
    for r in range(rank_bs):
        u_key = tuple(np.nonzero(U2[:, r])[0])
        v_key = tuple(np.nonzero(V2[:, r])[0])
        groups[(u_key, v_key)].append(r)

    # For groups with multiple terms, check if they can be merged
    new_U = []
    new_V = []
    new_W = []
    for key, r_list in groups.items():
        if len(r_list) == 1:
            r = r_list[0]
            new_U.append(U2[:, r])
            new_V.append(V2[:, r])
            new_W.append(W2[:, r])
        else:
            # Multiple terms with same U and V support
            # Check if W vectors are linearly independent
            W_sub = np.column_stack([W2[:, r] for r in r_list])
            rank_w = np.linalg.matrix_rank(W_sub, tol=1e-9)
            if rank_w < len(r_list):
                # Linearly dependent W vectors; can reduce
                # Use SVD to find independent subset
                U_sv, s_vals, Vt = np.linalg.svd(W_sub, full_matrices=False)
                # Keep only rank_w terms
                for k in range(rank_w):
                    # Reconstruct the term using the SVD
                    # W_combined = sum_k s_k * U_k ⊗ Vt_k
                    # But we need to express as sum of rank-1 terms
                    # with the same U and V factors
                    # This is tricky; let's just keep all for now
                    pass
            # For now, keep all terms (conservative)
            for r in r_list:
                new_U.append(U2[:, r])
                new_V.append(V2[:, r])
                new_W.append(W2[:, r])

    # The grouping approach didn't help much. Let me try a more direct
    # numerical rank reduction.
    #
    # Alternative: check if the overall tensor rank is less than 189
    # by computing the rank of the matricizations.
    # T has shape (36, 36, 36). The rank is <= min(rank(T_12), rank(T_13), rank(T_23))
    # where T_12 is the 36x(36*36) matricization, etc.

    T_12 = T.reshape(n * m, m * p * p * n)
    T_13 = T.reshape(n * m, p * n * m * p)
    T_23 = T.reshape(m * p, p * n * n * m)

    r12 = np.linalg.matrix_rank(T_12, tol=1e-9)
    r13 = np.linalg.matrix_rank(T_13, tol=1e-9)
    r23 = np.linalg.matrix_rank(T_23, tol=1e-9)

    # The tensor rank is at least max(r12, r13, r23) and at most min(...)
    # Actually, tensor rank >= max of matricization ranks
    # And tensor rank <= min of matricization ranks is NOT true in general
    # But tensor rank >= max(r12, r13, r23)

    # For our case, let's just return the best we have
    # The Block-Strassen gives rank 189 which is better than 216

    if time.time() > deadline - 5:
        return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}

    # Try one more thing: check if we can do better with a mixed approach
    # Use Strassen for some blocks and schoolbook for others
    # But this is complex; let's just return the 189-rank solution

    return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
