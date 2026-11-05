import time

N = 40
_DEADLINE = time.time() + 78.0  # safety margin vs 90s kill


def _compute_covered(S, max_val):
    """Bitmask of pairwise differences (1..max_val-2) covered by set S."""
    s = sorted(S)
    mask = 0
    n = len(s)
    for i in range(n):
        si = s[i]
        for j in range(i + 1, n):
            mask |= 1 << (s[j] - si - 1)
    return mask


def _count_bits(mask):
    return bin(mask).count('1')


def _find_diff_basis(target_size, max_val, seed=42, budget=25.0):
    """Find S of size target_size from 1..max_val-1 covering diffs 1..max_val-2.
    Must include 1 and max_val-1. Returns sorted list or None."""
    t0 = time.time()
    target_mask = (1 << (max_val - 2)) - 1  # bits 0..(max_val-3) -> diffs 1..(max_val-2)
    fixed = {1, max_val - 1}
    remaining = target_size - len(fixed)
    candidates = [x for x in range(2, max_val - 1)]

    def covered_of(S):
        return _compute_covered(S, max_val)

    def done(S):
        return (covered_of(S) & target_mask) == target_mask

    import random
    rng = random.Random(seed)

    best = None
    best_score = -1

    for attempt in range(20000):
        if time.time() - t0 > budget or time.time() > _DEADLINE - 3:
            break
        S = set(fixed)
        if remaining > 0:
            S.update(rng.sample(candidates, remaining))
        cov = covered_of(S)
        score = _count_bits(cov & target_mask)
        if score > best_score:
            best_score = score
            best = set(S)
        if done(S):
            return sorted(S)

        # Greedy local search
        for _ in range(40):
            if time.time() - t0 > budget or time.time() > _DEADLINE - 3:
                break
            s_list = sorted(S)
            cov = covered_of(S)
            score = _count_bits(cov & target_mask)
            if done(S):
                return sorted(S)
            improved = False
            for idx in range(len(s_list)):
                old = s_list[idx]
                if old in fixed:
                    continue
                for nv in candidates:
                    if nv in S and nv != old:
                        continue
                    S.remove(old)
                    S.add(nv)
                    ncov = covered_of(S)
                    nscore = _count_bits(ncov & target_mask)
                    if nscore > score:
                        score = nscore
                        cov = ncov
                        improved = True
                        break
                    else:
                        S.remove(nv)
                        S.add(old)
                if done(S):
                    return sorted(S)
            if not improved:
                break

    if best is not None and done(best):
        return sorted(best)
    return None


def _build_and_validate(S):
    """Build ladder + chords (0,s) for s in S. Return (edges, cycles) or None."""
    S = set(S)
    edges = set()
    # Top path 0-1-...-19
    for i in range(19):
        edges.add((i, i + 1))
    # Bottom path 20-21-...-39
    for i in range(20, 39):
        edges.add((i, i + 1))
    # Rungs
    for i in range(20):
        edges.add((i, i + 20))
    # Chords
    for s in S:
        a, b = 0, s
        if a > b:
            a, b = b, a
        edges.add((a, b))

    def edge_ok(u, v):
        a, b = (u, v) if u <= v else (v, u)
        return (a, b) in edges

    cycles = []
    # L = 3: triangle 0,1,2,0
    if not (edge_ok(0, 1) and edge_ok(1, 2) and edge_ok(2, 0)):
        return None
    cycles.append([0, 1, 2])

    # L = 4..39
    for L in range(4, N):
        if L % 2 == 0:
            m = L // 2 - 1
            cyc = list(range(0, m + 1)) + list(range(m + 20, 19, -1))
        else:
            m = (L - 1) // 2
            cyc = list(range(0, m + 1)) + list(range(m + 20, 20, -1))
            if not edge_ok(0, 21):
                # Try difference-basis form: 0, u, u+1, ..., v, 0 with v-u = L-2
                found = False
                for u in sorted(S):
                    v = u + (L - 2)
                    if v in S:
                        cyc = [0] + list(range(u, v + 1))
                        found = True
                        break
                if not found:
                    return None
        # Validate
        for i in range(len(cyc)):
            if not edge_ok(cyc[i], cyc[(i + 1) % len(cyc)]):
                return None
        cycles.append(cyc)

    # L = 40: Hamiltonian 0..19, 39..20
    cyc = list(range(0, 20)) + list(range(39, 19, -1))
    for i in range(len(cyc)):
        if not edge_ok(cyc[i], cyc[(i + 1) % len(cyc)]):
            return None
    cycles.append(cyc)

    edge_list = sorted(edges)
    return edge_list, cycles


def construct():
    t0 = time.time()
    # Search for minimal S. Prefer S containing 1 and 21 (21 covers all odd L via ladder).
    # Try sizes 2..8. For each, prefer sets that include 21.
    best_result = None
    best_key = None

    for size in range(2, 9):
        if time.time() > _DEADLINE - 5:
            break
        # First try with 21 forced (better for odd coverage via ladder)
        for seed in (42, 123, 999, 7, 555):
            if time.time() > _DEADLINE - 5:
                break
            S = _find_diff_basis(size, N, seed=seed, budget=6.0)
            if S is None:
                continue
            if 21 in S:
                res = _build_and_validate(S)
                if res is not None:
                    edges, cycles = res
                    key = (len(edges), 0)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_result = (edges, cycles)
                        break
        if best_result is not None and best_key[0] <= 58 + 1:
            break
        # Then try without forcing 21
        for seed in (42, 123, 999):
            if time.time() > _DEADLINE - 5:
                break
            S = _find_diff_basis(size, N, seed=seed, budget=4.0)
            if S is None:
                continue
            res = _build_and_validate(S)
            if res is not None:
                edges, cycles = res
                key = (len(edges), 1)
                if best_key is None or key < best_key:
                    best_key = key
                    best_result = (edges, cycles)

    if best_result is None:
        # Fallback: ladder with all needed chords (0,1) and (0,21)
        S = {1, 21}
        res = _build_and_validate(S)
        if res is None:
            # Ultimate fallback: star + path
            edges = [(0, i) for i in range(1, N)] + [(i, i + 1) for i in range(1, N - 1)]
            cycles = [list(range(L)) for L in range(3, N + 1)]
            return {'edges': edges, 'cycles': cycles}
        best_result = res

    edges, cycles = best_result
    return {'edges': edges, 'cycles': cycles}
