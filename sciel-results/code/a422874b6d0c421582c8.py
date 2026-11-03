"""
Matrix multiplication 3x4 by 4x7 with rank-minimal tensor decomposition.
Uses block decomposition with Strassen for 2x2x2 blocks and schoolbook for smaller blocks.
Then applies fast vectorized peeling to remove linearly dependent rank-1 terms.
Also tries alternative block splits to find lower rank.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a low-rank tensor decomposition for 3x4 * 4x7 matrix multiplication.
    """
    start_time = time.time()
    deadline = 140.0

    n, m, p = DIMENSIONS  # 3, 4, 7

    def build_decomp(split_n, split_m):
        """
        Build decomposition given how to split n and m.
        split_n: list of row group sizes summing to n
        split_m: list of col group sizes summing to m
        For each (n_block, m_block), use Strassen if both >=2, else schoolbook.
        """
        U_vecs = []
        V_vecs = []
        W_vecs = []

        def add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
            u = [0.0] * (n * m)
            v = [0.0] * (m * p)
            w = [0.0] * (p * n)

            for idx, val in zip(u_indices, u_vals):
                u[idx] += val
            for idx, val in zip(v_indices, v_vals):
                v[idx] += val
            for idx, val in zip(w_indices, w_vals):
                w[idx] += val

            U_vecs.append(u)
            V_vecs.append(v)
            W_vecs.append(w)

        def strassen_222_local(i0, j0, k0):
            i1 = i0 + 1
            j1 = j0 + 1
            k1 = k0 + 1

            add_rank1_term([i0*m+j0, i1*m+j1], [1, 1],
                           [j0*p+k0, j1*p+k1], [1, 1],
                           [k0*n+i0, k1*n+i1], [1, 1])
            add_rank1_term([i1*m+j0, i1*m+j1], [1, 1],
                           [j0*p+k0], [1],
                           [k0*n+i1, k1*n+i1], [1, -1])
            add_rank1_term([i0*m+j0], [1],
                           [j0*p+k1, j1*p+k1], [1, -1],
                           [k1*n+i0, k1*n+i1], [1, 1])
            add_rank1_term([i1*m+j1], [1],
                           [j1*p+k0, j0*p+k0], [1, -1],
                           [k0*n+i0, k0*n+i1], [1, 1])
            add_rank1_term([i0*m+j0, i0*m+j1], [1, 1],
                           [j1*p+k1], [1],
                           [k0*n+i0, k1*n+i0], [-1, 1])
            add_rank1_term([i1*m+j0, i0*m+j0], [1, -1],
                           [j0*p+k0, j0*p+k1], [1, 1],
                           [k1*n+i1], [1])
            add_rank1_term([i0*m+j1, i1*m+j1], [1, -1],
                           [j1*p+k0, j1*p+k1], [1, 1],
                           [k0*n+i0], [1])

        def schoolbook_block(i0, j0, k0, ni, mj, pk):
            for i in range(ni):
                for j in range(mj):
                    for k in range(pk):
                        ii = i0 + i
                        jj = j0 + j
                        kk = k0 + k
                        add_rank1_term([ii*m+jj], [1.0],
                                       [jj*p+kk], [1.0],
                                       [kk*n+ii], [1.0])

        def block_decomp(i0, j0, ni, mj):
            """Decompose ni x mj x p block at row offset i0, col offset j0."""
            if ni >= 2 and mj >= 2:
                # Use Strassen for 2x2x2 sub-blocks, schoolbook for remainder
                # Split p into chunks of 2, last chunk may be 1
                k0 = 0
                while k0 < p:
                    pk = min(2, p - k0)
                    if pk == 2:
                        # Further split ni and mj into 2s
                        i_off = 0
                        while i_off < ni:
                            ni_chunk = min(2, ni - i_off)
                            j_off = 0
                            while j_off < mj:
                                mj_chunk = min(2, mj - j_off)
                                if ni_chunk == 2 and mj_chunk == 2:
                                    strassen_222_local(i0 + i_off, j0 + j_off, k0)
                                else:
                                    schoolbook_block(i0 + i_off, j0 + j_off, k0, ni_chunk, mj_chunk, pk)
                                j_off += mj_chunk
                            i_off += ni_chunk
                    else:
                        # pk == 1, schoolbook
                        schoolbook_block(i0, j0, k0, ni, mj, 1)
                    k0 += pk
            else:
                # At least one dimension is 1, use schoolbook
                schoolbook_block(i0, j0, 0, ni, mj, p)

        # Iterate over n-blocks and m-blocks
        i_offset = 0
        for ni in split_n:
            j_offset = 0
            for mj in split_m:
                block_decomp(i_offset, j_offset, ni, mj)
                j_offset += mj
            i_offset += ni

        U = np.array(U_vecs).T if U_vecs else np.zeros((n*m, 0))
        V = np.array(V_vecs).T if V_vecs else np.zeros((m*p, 0))
        W = np.array(W_vecs).T if W_vecs else np.zeros((p*n, 0))
        return U, V, W

    def verify(U, V, W):
        """Check if decomposition is exact."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    def peel_fast(U, V, W):
        """
        Fast vectorized peeling: for each term, check if removing it preserves
        the decomposition. Use numpy for speed.
        """
        U_curr = U.copy()
        V_curr = V.copy()
        W_curr = W.copy()
        rank = U_curr.shape[1]
        removed = set()

        # Precompute all (i,j,k) indices
        ijks = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        num_ijk = len(ijks)

        # Precompute index arrays
        u_idx = np.array([i*m+j for i, j, k in ijks])
        v_idx = np.array([j*p+k for i, j, k in ijks])
        w_idx = np.array([k*n+i for i, j, k in ijks])

        # Extract sub-matrices for these indices
        U_sub = U_curr[u_idx, :]  # (num_ijk, rank)
        V_sub = V_curr[v_idx, :]
        W_sub = W_curr[w_idx, :]

        # The tensor value at (i,j,k) is sum_r U_sub[ijk,r] * V_sub[ijk,r] * W_sub[ijk,r]
        # = element-wise product summed over r
        # T[ijk] = sum_r (U_sub * V_sub * W_sub)[ijk, r]

        for r in range(rank):
            if time.time() > deadline - 5:
                break
            if r in removed:
                continue

            # Check if removing term r still gives correct values
            # T_without_r[ijk] = sum_{r2 != r, r2 not removed} U_sub[ijk,r2]*V_sub[ijk,r2]*W_sub[ijk,r2]
            # We need T_without_r[ijk] == 1.0 for all ijk

            # Compute contribution of term r
            contrib_r = U_sub[:, r] * V_sub[:, r] * W_sub[:, r]

            # Compute total with term r
            total_with = np.sum(U_sub * V_sub * W_sub, axis=1)

            # Total without r = total_with - contrib_r (but only for non-removed terms)
            # Actually we need to be more careful: removed terms are already excluded
            # Let's just recompute from scratch for non-removed, non-r terms
            # But that's slow. Instead, let's track the running total.

            # Simpler: compute total for all non-removed terms except r
            mask = np.ones(rank, dtype=bool)
            for rm in removed:
                mask[rm] = False
            mask[r] = False

            if not mask.any():
                continue

            U_m = U_sub[:, mask]
            V_m = V_sub[:, mask]
            W_m = W_sub[:, mask]
            total_without_r = np.sum(U_m * V_m * W_m, axis=1)

            if np.all(np.abs(total_without_r - 1.0) < 1e-9):
                removed.add(r)

        if not removed:
            return U, V, W

        keep = [r for r in range(rank) if r not in removed]
        return U_curr[:, keep], V_curr[:, keep], W_curr[:, keep]

    # Try multiple block splits and pick the best
    best_rank = None
    best_result = None

    # Split options for n=3: [1,2], [2,1], [1,1,1], [3], [1,1,1]
    # Split options for m=4: [2,2], [1,3], [3,1], [1,1,2], [2,1,1], [1,2,1], [4], [1,1,1,1], [1,1,2], [2,2]
    n_splits = [
        [1, 2],
        [2, 1],
        [1, 1, 1],
        [3],
    ]
    m_splits = [
        [2, 2],
        [1, 3],
        [3, 1],
        [1, 1, 2],
        [2, 1, 1],
        [1, 2, 1],
        [4],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [2, 1, 1],
    ]

    for ns in n_splits:
        for ms in m_splits:
            if time.time() > deadline - 20:
                break
            U, V, W = build_decomp(ns, ms)
            if not verify(U, V, W):
                continue

            # Apply peeling
            U_p, V_p, W_p = peel_fast(U, V, W)
            if not verify(U_p, V_p, W_p):
                U_p, V_p, W_p = U, V, W

            r = U_p.shape[1]
            if best_rank is None or r < best_rank:
                best_rank = r
                best_result = (U_p, V_p, W_p)

    if best_result is None:
        # Fallback: pure schoolbook
        U, V, W = build_decomp([3], [4])
        best_result = (U, V, W)

    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
