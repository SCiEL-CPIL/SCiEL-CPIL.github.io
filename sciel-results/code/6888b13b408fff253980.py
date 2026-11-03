"""Matrix multiplication 3x4x7 tensor decomposition - rank minimization."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]


def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2x2.
    U: (4,7), V: (4,7), W: (4,7)
    T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
    where i = i1*2+i2, j = j1*2+j2, k = k1*2+k2
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)

    # c=0: M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # c=1: M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1

    # c=2: M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1

    # c=3: M4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1

    # c=4: M5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1

    # c=5: M6 = (c-d)(f+h)
    U[2, 5] = 1; U[3, 5] = -1
    V[1, 5] = 1; V[3, 5] = 1
    W[3, 5] = 1

    # c=6: M7 = (b-d)(e+g)
    U[1, 6] = 1; U[3, 6] = -1
    V[0, 6] = 1; V[2, 6] = 1
    W[0, 6] = 1

    return U, V, W


def _generate_partitions(n):
    """Generate all ways to partition range(n) into blocks of size 1 or 2.
    Returns a list of partitions, where each partition is a list of tuples of indices.
    """
    results = []
    used = [False] * n

    def backtrack(remaining, current):
        if remaining == 0:
            results.append(list(current))
            return
        # Find the smallest unused index
        first = -1
        for i in range(n):
            if not used[i]:
                first = i
                break
        if first == -1:
            return

        # Option 1: first as singleton
        used[first] = True
        current.append((first,))
        backtrack(remaining - 1, current)
        current.pop()
        used[first] = False

        # Option 2: pair first with another unused index
        for j in range(first + 1, n):
            if not used[j]:
                used[first] = True
                used[j] = True
                current.append((first, j))
                backtrack(remaining - 2, current)
                current.pop()
                used[first] = False
                used[j] = False

    backtrack(n, [])
    return results


def _estimate_rank(nb, mb, pb):
    """Estimate the rank for a given combination of partitions."""
    rank = 0
    for b_n in nb:
        for b_m in mb:
            for b_p in pb:
                sn, sm, sp = len(b_n), len(b_m), len(b_p)
                if sn == 2 and sm == 2 and sp == 2:
                    rank += 7
                else:
                    rank += sn * sm * sp
    return rank


def _build_decomposition(n, m, p, nb, mb, pb):
    """Build a decomposition using Strassen for 2x2x2 blocks and schoolbook otherwise.
    
    Returns (U, V, W, rank) or None if invalid.
    """
    total_rank = _estimate_rank(nb, mb, pb)
    if total_rank >= n * m * p:
        return None

    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)

    col = 0
    U_s, V_s, W_s = _strassen_2x2x2()

    for b_n in nb:
        for b_m in mb:
            for b_p in pb:
                sn, sm, sp = len(b_n), len(b_m), len(b_p)
                if sn == 2 and sm == 2 and sp == 2:
                    # Strassen block
                    for ci in range(2):
                        for cj in range(2):
                            for ck in range(2):
                                i = b_n[ci]
                                j = b_m[cj]
                                k = b_p[ck]
                                for c in range(7):
                                    U[i * m + j, col + c] = U_s[ci * 2 + cj, c]
                                    V[j * p + k, col + c] = V_s[cj * 2 + ck, c]
                                    W[k * n + i, col + c] = W_s[ck * 2 + ci, c]
                    col += 7
                else:
                    # Schoolbook block
                    for ci in range(sn):
                        for cj in range(sm):
                            for ck in range(sp):
                                i = b_n[ci]
                                j = b_m[cj]
                                k = b_p[ck]
                                U[i * m + j, col] = 1
                                V[j * p + k, col] = 1
                                W[k * n + i, col] = 1
                                col += 1

    if col != total_rank:
        return None

    return U, V, W, total_rank


def _check_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    T_recon = np.einsum('ic,jc,kc->ijk', U, V, W)
    return np.allclose(T_recon, T, atol=tol)


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill

    # Schoolbook baseline
    rank_sb = n * m * p  # 84
    U_sb = np.zeros((n * m, rank_sb), dtype=float)
    V_sb = np.zeros((m * p, rank_sb), dtype=float)
    W_sb = np.zeros((p * n, rank_sb), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb

    # Generate all partitions
    n_parts = _generate_partitions(n)
    m_parts = _generate_partitions(m)
    p_parts = _generate_partitions(p)

    # Sort each list of partitions: prefer more 2-blocks (more Strassen potential)
    def _part_score(parts):
        num_2 = sum(1 for b in parts if len(b) == 2)
        return -num_2

    n_parts.sort(key=_part_score)
    m_parts.sort(key=_part_score)
    p_parts.sort(key=_part_score)

    # Pre-compute all (nb, mb, pb) combinations with their estimated ranks
    # and sort by rank ascending to find minimum first
    combos = []
    for nb in n_parts:
        for mb in m_parts:
            for pb in p_parts:
                r = _estimate_rank(nb, mb, pb)
                if r < best_rank:
                    combos.append((r, nb, mb, pb))

    combos.sort(key=lambda x: x[0])

    # Try combinations in order of estimated rank
    for r_est, nb, mb, pb in combos:
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break

        if r_est >= best_rank:
            break  # No further improvement possible

        result = _build_decomposition(n, m, p, nb, mb, pb)
        if result is not None:
            U, V, W, rank = result
            if rank < best_rank:
                # Verify
                if _check_decomposition(U, V, W, n, m, p):
                    best_U, best_V, best_W = U, V, W
                    best_rank = rank
                    # Continue searching for even better
                else:
                    # Shouldn't happen with correct construction, but skip
                    pass

    # Final verification
    if _check_decomposition(best_U, best_V, best_W, n, m, p):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
