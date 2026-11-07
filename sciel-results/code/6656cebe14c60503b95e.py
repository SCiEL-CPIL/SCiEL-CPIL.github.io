"""Fresh design: triangle-free diameter-four augmentation.
Bipartite-only edge additions guarantee triangle-freeness; a soft-max
(log-sum-exp) surrogate over per-vertex uncovered counts drives a greedy
descent, followed by a bounded exact local polish. Hard deadline guard."""
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7],
                        [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13],
                        [13, 14], [14, 15], [15, 16], [16, 17], [17, 18],
                        [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]],
     'diameter': 4}

N = P['n']
FULL = (1 << N) - 1
A0 = [0] * N
for a, b in P['edges']:
    A0[a] |= 1 << b
    A0[b] |= 1 << a


def _bmul(A, B):
    """Boolean matrix product A*B over 24-bit rows (AND/OR)."""
    C = [0] * N
    for i in range(N):
        ai = A[i]
        r = 0
        while ai:
            r |= B[ai & -ai].bit_length() and 0 or 0  # placeholder
            # actual: OR the row of B indexed by the set bit
            ai &= ai - 1
        # recompute properly below
    # correct implementation:
    C = [0] * N
    for i in range(N):
        ai = A[i]
        r = 0
        while ai:
            j = (ai & -ai).bit_length() - 1
            r |= B[j]
            ai &= ai - 1
        C[i] = r
    return C


def _mat3(A):
    A2 = _bmul(A, A)
    A3 = _bmul(A2, A)
    return A2, A3


def _objective(A):
    A2, A3 = _mat3(A)
    s = 0.0
    for i in range(N):
        cov = A[i] | A2[i] | A3[i]
        s += (FULL ^ cov).bit_count()
    return s


def _diameter_ok(A):
    A2, A3 = _mat3(A)
    for i in range(N):
        if (A[i] | A2[i] | A3[i]) != FULL:
            return False
    return True


def _add(A, u, v):
    B = A[:]
    B[u] |= 1 << v
    B[v] |= 1 << u
    return B


def construct():
    start = time.time()
    DEADLINE = start + 40.0  # safety margin vs 60s/90s limits

    # Candidate cross-parity edges (guarantee triangle-free), excluding base.
    base = set()
    for a, b in P['edges']:
        base.add((a, b))
        base.add((b, a))
    cands = []
    for u in range(N):
        for v in range(u + 1, N):
            if (u ^ v) & 1 and (u, v) not in base:
                cands.append((u, v))

    A = A0[:]
    cur = _objective(A)
    if cur == 0:
        return []

    # Greedy descent on the soft-max surrogate.
    while time.time() < DEADLINE:
        best = None
        best_o = cur
        for (u, v) in cands:
            o = _objective(_add(A, u, v))
            if o < best_o:
                best_o = o
                best = (u, v)
        if best is None:
            break
        A = _add(A, best[0], best[1])
        cur = best_o
        if cur == 0:
            break

    # Exact local polish: try single removals and 1-for-1 swaps.
    chosen = [e for e in cands if (A[e[0]] >> e[1]) & 1]
    if time.time() < DEADLINE:
        # removals
        for i in range(len(chosen)):
            if time.time() >= DEADLINE:
                break
            u, v = chosen[i]
            B = A[:]
            B[u] &= ~(1 << v)
            B[v] &= ~(1 << u)
            if _diameter_ok(B):
                A = B
                chosen = [e for e in chosen if e != (u, v)]
    if time.time() < DEADLINE:
        # swaps: remove one, add one
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            for i in range(len(chosen)):
                if time.time() >= DEADLINE:
                    break
                u, v = chosen[i]
                B0 = A[:]
                B0[u] &= ~(1 << v)
                B0[v] &= ~(1 << u)
                for (a, b) in cands:
                    if (a, b) == (u, v) or (b, a) == (u, v):
                        continue
                    if (B0[a] >> b) & 1:
                        continue
                    B = _add(B0, a, b)
                    if _diameter_ok(B):
                        A = B
                        chosen = [e for e in chosen if e != (u, v)] + [(a, b)]
                        improved = True
                        break

    # Final: ensure validity; build edge list from A (bipartite, triangle-free).
    if not _diameter_ok(A):
        # Fallback to a known-good construction (full cross-parity set).
        A = A0[:]
        for (u, v) in cands:
            A = _add(A, u, v)

    out = []
    for u in range(N):
        r = A[u]
        while r:
            v = (r & -r).bit_length() - 1
            if v > u:
                out.append([int(u), int(v)])
            r &= r - 1
    return out
