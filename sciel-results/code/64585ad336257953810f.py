import time
import sys
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = 145.0

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    def verify(U_mat, V_mat, W_mat, tol=1e-8):
        r = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = np.dot(U_mat[a], V_mat[b] * W_mat[c])
                    if abs(s - 1.0) > tol:
                        return False
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    i_w = c % n
                    k_w = c // n
                    if i_a == i_w and j_a == j_b and k_b == k_w:
                        continue
                    s = np.dot(U_mat[a], V_mat[b] * W_mat[c])
                    if abs(s) > tol:
                        return False
        return True

    def add_strassen_block_standard(U, V, W, i_start, j_start, k_start, offset):
        a00 = i_start * m + j_start
        a01 = i_start * m + j_start + 1
        a10 = (i_start + 1) * m + j_start
        a11 = (i_start + 1) * m + j_start + 1
        b00 = j_start * p + k_start
        b01 = j_start * p + k_start + 1
        b10 = (j_start + 1) * p + k_start
        b11 = (j_start + 1) * p + k_start + 1
        w00 = k_start * n + i_start
        w01 = (k_start + 1) * n + i_start
        w10 = k_start * n + i_start + 1
        w11 = (k_start + 1) * n + i_start + 1

        U[a00, offset] += 1.0; U[a11, offset] += 1.0
        V[b00, offset] += 1.0; V[b11, offset] += 1.0
        W[w00, offset] += 1.0; W[w11, offset] += 1.0

        U[a10, offset+1] += 1.0; U[a11, offset+1] += 1.0
        V[b00, offset+1] += 1.0
        W[w10, offset+1] += 1.0; W[w11, offset+1] -= 1.0

        U[a00, offset+2] += 1.0
        V[b01, offset+2] += 1.0; V[b11, offset+2] -= 1.0
        W[w01, offset+2] += 1.0; W[w11, offset+2] += 1.0

        U[a11, offset+3] += 1.0
        V[b10, offset+3] += 1.0; V[b00, offset+3] -= 1.0
        W[w00, offset+3] += 1.0; W[w10, offset+3] += 1.0

        U[a00, offset+4] += 1.0; U[a01, offset+4] += 1.0
        V[b11, offset+4] += 1.0
        W[w00, offset+4] -= 1.0; W[w01, offset+4] += 1.0

        U[a10, offset+5] += 1.0; U[a00, offset+5] -= 1.0
        V[b00, offset+5] += 1.0; V[b01, offset+5] += 1.0
        W[w11, offset+5] += 1.0

        U[a01, offset+6] += 1.0; U[a11, offset+6] -= 1.0
        V[b10, offset+6] += 1.0; V[b11, offset+6] += 1.0
        W[w00, offset+6] += 1.0

    # Build rank-42 Strassen decomposition
    rank_init = 42
    U = np.zeros((nm, rank_init), dtype=float)
    V = np.zeros((mp, rank_init), dtype=float)
    W = np.zeros((pn, rank_init), dtype=float)

    offset = 0
    for i_start in range(0, n, 2):
        for j_start in range(0, m, 2):
            for k_start in range(0, p, 2):
                add_strassen_block_standard(U, V, W, i_start, j_start, k_start, offset)
                offset += 7

    if verify(U, V, W):
        best_rank = rank_init
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    else:
        # Fallback schoolbook
        rank0 = n * m * p
        U = np.zeros((nm, rank0), dtype=float)
        V = np.zeros((mp, rank0), dtype=float)
        W = np.zeros((pn, rank0), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * (m * p) + j * p + k
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Attempt rank reduction via SVD-based null space analysis
    # The tensor T[i,j,k] = 1 if i,k in range, j matches
    # Each rank-1 term is U[:,t] ⊗ V[:,t] ⊗ W[:,t]
    # Try to find linear dependencies

    # Use numerical rank detection on the coefficient structure
    # For each pair of terms, check if one can be expressed as linear combination
    def try_reduce_rank(U_curr, V_curr, W_curr, max_rank, tol=1e-10):
        r = U_curr.shape[1]
        if r <= max_rank:
            return U_curr, V_curr, W_curr, r

        # Try greedy elimination: for each column, check if it's in span of others
        # This is expensive, so use SVD-based approach
        # Build the 3-tensor as a matrix and check rank
        # Reshape: for fixed a, the tensor is V[:,t] * W[:,t] outer product
        # Actually, let's use the fact that if we fix a, the contribution is sum_t U[a,t] * V[:,t] * W[:,t]
        # This is rank-1 in the (b,c) space for each t

        # Alternative: try to find if any column is redundant
        # Use the verification tensor approach
        # Build full tensor T of shape (nm, mp, pn)
        # T[a,b,c] = sum_t U[a,t] * V[b,t] * W[c,t]

        # This is too large: 8*24*12 = 2304 entries, but we only care about specific ones
        # Let's use a different approach: check numerical rank of the set of vectors

        # For each term t, define vector f_t in R^(nm*mp*pn) but sparse
        # Instead, check if the set {U[:,t] ⊗ V[:,t] ⊗ W[:,t]} is linearly independent

        # Practical approach: use random projections to check rank
        # Project to a lower-dimensional space and check rank there

        # Try: for each column t, see if removing it and adjusting others still works
        # This is essentially checking if the column is in the span of others

        # Use SVD on the coefficient matrices
        # If U has rank < r, we can reduce
        U_svd = np.linalg.svd(U_curr, full_matrices=False)
        V_svd = np.linalg.svd(V_curr, full_matrices=False)
        W_svd = np.linalg.svd(W_curr, full_matrices=False)

        rank_U = np.sum(U_svd[1] > tol)
        rank_V = np.sum(V_svd[1] > tol)
        rank_W = np.sum(W_svd[1] > tol)

        min_rank = min(rank_U, rank_V, rank_W)
        if min_rank < r:
            # Can potentially reduce to min_rank
            # But need to ensure the tensor decomposition still holds
            # This is not straightforward, so just return current
            pass

        return U_curr, V_curr, W_curr, r

    # Try random perturbations and SVD reduction
    rng = np.random.RandomState(42)

    # Attempt: add small random noise, then SVD truncate
    for attempt in range(50):
        if time.time() > deadline - 5:
            break

        U_try = best_U.copy()
        V_try = best_V.copy()
        W_try = best_W.copy()

        # Add small random perturbation
        noise_scale = 1e-6
        U_try += rng.randn(nm, rank_init) * noise_scale
        V_try += rng.randn(mp, rank_init) * noise_scale
        W_try += rng.randn(pn, rank_init) * noise_scale

        # Check if still valid
        if verify(U_try, V_try, W_try, tol=1e-6):
            # Try to reduce rank via SVD
            U_s, _, _ = np.linalg.svd(U_try, full_matrices=False)
            V_s, _, _ = np.linalg.svd(V_try, full_matrices=False)
            W_s, _, _ = np.linalg.svd(W_try, full_matrices=False)

            for target_rank in [41, 40, 39, 38, 37, 36]:
                if target_rank < best_rank:
                    # Truncate to target_rank
                    U_trunc = U_try[:, :target_rank]
                    V_trunc = V_try[:, :target_rank]
                    W_trunc = W_try[:, :target_rank]
                    if verify(U_trunc, V_trunc, W_trunc, tol=1e-6):
                        best_rank = target_rank
                        best_U, best_V, best_W = U_trunc.copy(), V_trunc.copy(), W_trunc.copy()
                        break

    # Try structured rank reduction: remove one term at a time and solve
    for remove_idx in range(rank_init):
        if time.time() > deadline - 3:
            break

        U_sub = np.delete(best_U, remove_idx, axis=1)
        V_sub = np.delete(best_V, remove_idx, axis=1)
        W_sub = np.delete(best_W, remove_idx, axis=1)

        if verify(U_sub, V_sub, W_sub, tol=1e-8):
            best_rank = best_U.shape[1] - 1
            best_U, best_V, best_W = U_sub, V_sub, W_sub

    # Final verification
    if not verify(best_U, best_V, best_W, tol=1e-8):
        # Reset to original rank-42
        U = np.zeros((nm, 42), dtype=float)
        V = np.zeros((mp, 42), dtype=float)
        W = np.zeros((pn, 42), dtype=float)
        offset = 0
        for i_start in range(0, n, 2):
            for j_start in range(0, m, 2):
                for k_start in range(0, p, 2):
                    add_strassen_block_standard(U, V, W, i_start, j_start, k_start, offset)
                    offset += 7
        best_U, best_V, best_W = U, V, W
        best_rank = 42

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
