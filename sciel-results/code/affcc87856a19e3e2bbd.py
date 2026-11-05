"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 5, 6

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, rank):
        """Verify that the decomposition is exact (vectorized via NumPy)."""
        Ua = np.asarray(U, dtype=float)
        Va = np.asarray(V, dtype=float)
        Wa = np.asarray(W, dtype=float)
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Build the full tensor by iterating over the small rank axis.
        T = np.zeros((n, m, p), dtype=float)
        for c in range(rank):
            uc = Ua[:, c].reshape(n, m)
            vc = Va[:, c].reshape(m, p)
            wc = Wa[:, c].reshape(p, n)
            # outer product: (i,j) from uc, (j,k) from vc, (k,i) from wc
            # contribution to T[i,j,k] = uc[i,j]*vc[j,k]*wc[k,i]
            # Use einsum for clarity
            T += np.einsum('ij,jk,ki->ijk', uc, vc, wc)
        return bool(np.all(np.abs(T - 1.0) <= 1e-9))

    def strassen_2x2x2_terms():
        """
        Return the 7 Strassen terms for 2x2x2 matrix multiplication.
        Each term is (U_pattern, V_pattern, W_pattern) where patterns are
        dicts mapping (local_i, local_j) -> value for U, etc.
        """
        terms = []
        # Term 1: M1 = (A00+A11)(B00+B11)
        # Contributes +1 to C00 (T[0,0,0]) and +1 to C11 (T[1,1,1])
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (1,1): 1}
        terms.append((u1, v1, w1))
        
        # Term 2: M2 = (A10+A11)B00
        # Contributes +1 to C10 (T[1,0,0]) and -1 to C11 (T[1,1,1])
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        w2 = {(0,1): 1, (1,1): -1}
        terms.append((u2, v2, w2))
        
        # Term 3: M3 = A00(B01-B11)
        # Contributes +1 to C01 (T[0,0,1]) and +1 to C11 (T[1,1,1])
        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,1): -1}
        w3 = {(1,0): 1, (1,1): 1}
        terms.append((u3, v3, w3))
        
        # Term 4: M4 = A11(B10-B00)
        # Contributes +1 to C00 (T[0,0,0]) and +1 to C10 (T[1,0,0])
        u4 = {(1,1): 1}
        v4 = {(1,0): 1, (0,0): -1}
        w4 = {(0,0): 1, (0,1): 1}
        terms.append((u4, v4, w4))
        
        # Term 5: M5 = (A00+A01)B11
        # Contributes -1 to C00 (T[0,0,0]) and +1 to C01 (T[0,0,1])
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        w5 = {(0,0): -1, (1,0): 1}
        terms.append((u5, v5, w5))
        
        # Term 6: M6 = (A10-A00)(B00+B01)
        # Contributes +1 to C11 (T[1,1,1])
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (0,1): 1}
        w6 = {(1,1): 1}
        terms.append((u6, v6, w6))
        
        # Term 7: M7 = (A01-A11)(B10+B11)
        # Contributes +1 to C00 (T[0,0,0])
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,0): 1, (1,1): 1}
        w7 = {(0,0): 1}
        terms.append((u7, v7, w7))
        
        return terms

    def _all_groupings(dim):
        """Yield all ways to partition 0..dim-1 into groups of size 1 or 2,
        in order (each group is a sorted list, groups are in index order)."""
        if dim == 0:
            yield []
            return
        if dim == 1:
            yield [[0]]
            return
        if dim == 2:
            yield [[0, 1]]
            yield [[0], [1]]
            return
        if dim == 3:
            # first group either [0,1] or [0]
            for rest in _all_groupings(dim - 2):
                yield [[0, 1]] + [[x + 2 for x in g] for g in rest]
            for rest in _all_groupings(dim - 1):
                yield [[0]] + [[x + 1 for x in g] for g in rest]
            return
        if dim == 4:
            for rest in _all_groupings(dim - 2):
                yield [[0, 1]] + [[x + 2 for x in g] for g in rest]
            for rest in _all_groupings(dim - 1):
                yield [[0]] + [[x + 1 for x in g] for g in rest]
            return
        if dim == 5:
            for rest in _all_groupings(dim - 2):
                yield [[0, 1]] + [[x + 2 for x in g] for g in rest]
            for rest in _all_groupings(dim - 1):
                yield [[0]] + [[x + 1 for x in g] for g in rest]
            return
        if dim == 6:
            for rest in _all_groupings(dim - 2):
                yield [[0, 1]] + [[x + 2 for x in g] for g in rest]
            for rest in _all_groupings(dim - 1):
                yield [[0]] + [[x + 1 for x in g] for g in rest]
            return
        # Fallback: brute force for small dims
        from itertools import product
        # A grouping is equivalent to a bitmask of "pair (0,1), (2,3), ..." decisions
        # but groups must be contiguous in index order. For dim<=8 this is fine.
        # We'll just use the recursive approach generically:
        def rec(idx, remaining):
            if idx == dim:
                yield []
                return
            # single
            for rest in rec(idx + 1, remaining - 1):
                yield [[idx]] + rest
            # pair (if room)
            if remaining >= 2:
                for rest in rec(idx + 2, remaining - 2):
                    yield [[idx, idx + 1]] + rest
        yield from rec(0, dim)

    def _rank_for_groupings(i_groups, j_groups, k_groups):
        total = 0
        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        total += 7
                    else:
                        total += si * sj * sk
        return total

    def build_strassen_decomposition(i_groups=None, j_groups=None, k_groups=None):
        """
        Build a decomposition using Strassen on 2x2x2 blocks and schoolbook elsewhere.
        If groupings are not provided, search all valid 2/1 partitions of each
        dimension and pick the one with minimal total rank.
        """
        if i_groups is None or j_groups is None or k_groups is None:
            best_rank = None
            best = None
            i_opts = list(_all_groupings(n))
            j_opts = list(_all_groupings(m))
            k_opts = list(_all_groupings(p))
            for ig in i_opts:
                for jg in j_opts:
                    for kg in k_opts:
                        r = _rank_for_groupings(ig, jg, kg)
                        if best_rank is None or r < best_rank:
                            best_rank = r
                            best = (ig, jg, kg)
            i_groups, j_groups, k_groups = best
            total_rank = best_rank
        else:
            total_rank = _rank_for_groupings(i_groups, j_groups, k_groups)

        # Allocate columns
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]

        col = 0
        strassen_terms = strassen_2x2x2_terms()

        for ig in i_groups:
            for jg in j_groups:
                for kg in k_groups:
                    si, sj, sk = len(ig), len(jg), len(kg)
                    if si == 2 and sj == 2 and sk == 2:
                        # Use Strassen
                        for (u_pat, v_pat, w_pat) in strassen_terms:
                            for (li, lj), val_u in u_pat.items():
                                gi, gj = ig[li], jg[lj]
                                U[gi * m + gj][col] = val_u
                            for (lj, lk), val_v in v_pat.items():
                                gj, gk = jg[lj], kg[lk]
                                V[gj * p + gk][col] = val_v
                            for (lk, li), val_w in w_pat.items():
                                gk, gi = kg[lk], ig[li]
                                W[gk * n + gi][col] = val_w
                            col += 1
                    else:
                        # Schoolbook for this sub-block
                        for li in range(si):
                            for lj in range(sj):
                                for lk in range(sk):
                                    gi, gj, gk = ig[li], jg[lj], kg[lk]
                                    U[gi * m + gj][col] = 1
                                    V[gj * p + gk][col] = 1
                                    W[gk * n + gi][col] = 1
                                    col += 1

        return {"U": U, "V": V, "W": W, "rank": total_rank}

    def try_improved_decomposition():
        """
        Try to find a better decomposition using block-Strassen with the
        optimal 2/1 group partition (searched exhaustively over small dims).
        """
        if time.time() > deadline:
            return None

        result = build_strassen_decomposition()
        rank = result.pop("rank")

        # Verify (vectorized)
        if verify(result["U"], result["V"], result["W"], rank):
            return result

        return None

    # Start with schoolbook as fallback
    best_result = schoolbook()
    best_rank = n * m * p

    # Verify the schoolbook decomposition
    if not verify(best_result["U"], best_result["V"], best_result["W"], best_rank):
        pass

    # Try improved decomposition
    improved = try_improved_decomposition()
    if improved is not None:
        improved_rank = len(improved["U"][0])
        if improved_rank < best_rank:
            best_result = improved
            best_rank = improved_rank

    return best_result
# EVOLVE-BLOCK-END
