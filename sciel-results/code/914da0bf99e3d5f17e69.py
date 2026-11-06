import math
import time
import random


def construct():
    start_time = time.time()
    # Evaluator kills at 90s wall-clock; keep a comfortable safety margin.
    deadline = start_time + 75.0

    max_m = 128

    # Precompute gcd for even moduli (2..128 step 2).
    evens = list(range(2, max_m + 1, 2))
    n_even = len(evens)
    gcds = [[0] * n_even for _ in range(n_even)]
    for i in range(n_even):
        mi = evens[i]
        row = gcds[i]
        for j in range(i, n_even):
            g = math.gcd(mi, evens[j])
            row[j] = g
            gcds[j][i] = g

    # Ordering: descending modulus (largest first) for a strong forward-checking order.
    order = list(range(n_even - 1, -1, -1))  # indices into evens, largest modulus first

    def valid_residues(k, chosen):
        """Return list of residues r in [0, m) for modulus evens[k] that are disjoint
        from every already-chosen class. chosen is a list of (r, m) pairs."""
        m = evens[k]
        half = m // 2
        forbidden = bytearray(half)  # forbidden[a] for a in 0..half-1
        for (r_i, m_i) in chosen:
            g = gcds[k][m_i_index[m_i]]
            # r is invalid if r % g == r_i % g.  r = 2a, so 2a % g == r_i % g
            # <=> a % (g/2) == (r_i % g) // 2   (since g even and r_i%g even).
            h = g // 2
            b = (r_i % g) // 2
            for a in range(b, half, h):
                forbidden[a] = 1
        res = []
        for a in range(half):
            if not forbidden[a]:
                res.append(2 * a)
        return res

    # Map modulus value -> index in evens (for gcd lookup).
    m_i_index = {m: i for i, m in enumerate(evens)}

    # ---------- Fast deterministic greedy (descending + MRV) ----------
    def greedy_deterministic():
        chosen = []
        remaining = list(order)
        while remaining:
            # MRV: pick the remaining modulus with the fewest valid residues.
            best_k = None
            best_valid = None
            best_count = None
            for k in remaining:
                vr = valid_residues(k, chosen)
                c = len(vr)
                if c == 0:
                    return None  # dead end
                if best_count is None or c < best_count:
                    best_count = c
                    best_k = k
                    best_valid = vr
                    if c == 1:
                        break
            chosen.append((best_valid[0], evens[best_k]))
            remaining.remove(best_k)
        return chosen

    result = greedy_deterministic()
    if result is not None and len(result) == n_even:
        # Success: 64 pairwise-disjoint classes with distinct moduli.
        return [[r, m] for (r, m) in result]

    # ---------- Lightweight backtracking (only if greedy failed) ----------
    def backtrack(remaining, chosen):
        if time.time() > deadline - 3.0:
            return None
        if not remaining:
            return chosen
        best_k = None
        best_valid = None
        best_count = None
        for k in remaining:
            vr = valid_residues(k, chosen)
            c = len(vr)
            if c == 0:
                return None
            if best_count is None or c < best_count:
                best_count = c
                best_k = k
                best_valid = vr
                if c == 1:
                    break
        remaining.remove(best_k)
        for r in best_valid:
            res = backtrack(remaining, chosen + [(r, evens[best_k])])
            if res is not None:
                return res
        remaining.append(best_k)
        return None

    remaining = list(order)
    bt = backtrack(remaining, [])
    if bt is not None and len(bt) == n_even:
        return [[r, m] for (r, m) in bt]

    # ---------- Fallback: randomized greedy to guarantee a valid nonempty result ----------
    best = []
    for _ in range(20000):
        if time.time() > deadline - 1.0:
            break
        chosen = []
        shuffled = evens[:]
        random.shuffle(shuffled)
        for m in shuffled:
            k = m_i_index[m]
            vr = valid_residues(k, chosen)
            if vr:
                chosen.append((random.choice(vr), m))
        if len(chosen) > len(best):
            best = chosen
            if len(best) == n_even:
                break

    return [[r, m] for (r, m) in best]
