import time
import random

N = 16
NUM_PTS = N * N
FULL_MASK = (1 << NUM_PTS) - 1

# Precompute geometry
POINTS = [(x, y) for y in range(N) for x in range(N)]

# Precompute squared distances
dist = [[0] * NUM_PTS for _ in range(NUM_PTS)]
for i in range(NUM_PTS):
    x1, y1 = POINTS[i]
    for j in range(i + 1, NUM_PTS):
        x2, y2 = POINTS[j]
        d = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
        dist[i][j] = d
        dist[j][i] = d

# Iso[i][j] = bitmask of k such that triangle i,j,k is isosceles
Iso = [[0] * NUM_PTS for _ in range(NUM_PTS)]
for i in range(NUM_PTS):
    for j in range(i + 1, NUM_PTS):
        d_ij = dist[i][j]
        mask = 0
        for k in range(NUM_PTS):
            if k == i or k == j:
                continue
            d_ik = dist[i][k]
            d_jk = dist[j][k]
            if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                mask |= (1 << k)
        Iso[i][j] = mask
        Iso[j][i] = mask


def blocked_of(S):
    """Union of isosceles-blocks for all pairs in S."""
    m = 0
    n = len(S)
    for a in range(n):
        pi = S[a]
        for b in range(a + 1, n):
            m |= Iso[pi][S[b]]
    return m


def is_valid(S):
    """No three points of S form an isosceles triangle."""
    sm = 0
    for p in S:
        sm |= (1 << p)
    return (sm & blocked_of(S)) == 0


def is_maximal(S):
    """S is a no-isosceles set that cannot be extended."""
    sm = 0
    for p in S:
        sm |= (1 << p)
    if (sm & blocked_of(S)) != 0:
        return False
    return (sm | blocked_of(S)) == FULL_MASK


def greedy_maximal(rng):
    """Directly build a *small* maximal set via set-cover style picking."""
    S = []
    sm = 0
    # Start from a random point to diversify
    first = rng.randrange(NUM_PTS)
    S.append(first)
    sm |= (1 << first)
    while True:
        if time.time() > DEADLINE:
            break
        blk = blocked_of(S)
        uncovered = FULL_MASK & ~(sm | blk)
        if uncovered == 0:
            break
        # Pick the point covering the most uncovered points
        best_p = -1
        best_cov = -1
        # scan all points (256) -- cheap
        for p in range(NUM_PTS):
            if (sm >> p) & 1:
                continue
            # coverage added by p = union of Iso[p][s] for s in S
            add = 0
            for s in S:
                add |= Iso[p][s]
            cov = (add & uncovered).bit_count()
            if cov > best_cov:
                best_cov = cov
                best_p = p
        if best_p < 0:
            break
        S.append(best_p)
        sm |= (1 << best_p)
    return S


def local_search(S, deadline):
    """Minimize |S| while keeping it maximal. Removals + 1-for-1 + 2-for-1 swaps."""
    S = S[:]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(S)
        # --- Removals ---
        for i in range(n):
            T = S[:i] + S[i + 1:]
            if is_maximal(T):
                S = T
                improved = True
                break
        if improved:
            continue
        # --- 1-for-1 swaps ---
        sm = 0
        for p in S:
            sm |= (1 << p)
        cand = [p for p in range(NUM_PTS) if not (sm >> p) & 1]
        random.shuffle(cand)
        done = False
        for i in range(len(S)):
            if time.time() > deadline:
                break
            q = S[i]
            T0 = S[:i] + S[i + 1:]
            for y in cand[:40]:
                T = T0 + [y]
                if is_maximal(T):
                    S = T
                    improved = True
                    done = True
                    break
            if done:
                break
        if improved:
            continue
        # --- 2-for-1 swaps (remove two, add one) ---
        sm = 0
        for p in S:
            sm |= (1 << p)
        cand = [p for p in range(NUM_PTS) if not (sm >> p) & 1]
        random.shuffle(cand)
        done = False
        for i in range(len(S)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(S)):
                T0 = S[:i] + S[i + 1:j] + S[j + 1:]
                for y in cand[:25]:
                    T = T0 + [y]
                    if is_maximal(T):
                        S = T
                        improved = True
                        done = True
                        break
                if done:
                    break
            if done:
                break
    return S


def construct():
    global DEADLINE
    DEADLINE = time.time() + 285

    best_set = None
    best_len = float('inf')

    # Phase 1: many diverse greedy maximal constructions
    for seed in range(400):
        if time.time() > DEADLINE:
            break
        rng = random.Random(seed)
        S = greedy_maximal(rng)
        if is_maximal(S) and len(S) < best_len:
            best_len = len(S)
            best_set = S[:]

    if best_set is None:
        # Fallback: build any maximal set
        S = []
        sm = 0
        for p in range(NUM_PTS):
            if (sm >> p) & 1:
                continue
            blk = blocked_of(S)
            if (blk >> p) & 1:
                continue
            S.append(p)
            sm |= (1 << p)
        best_set = S

    # Phase 2: local search from the best (and a few other good seeds)
    if time.time() < DEADLINE:
        S = local_search(best_set, DEADLINE)
        if is_maximal(S) and len(S) < best_len:
            best_len = len(S)
            best_set = S[:]

    # Phase 3: occasional fresh restarts if time remains
    seed = 400
    while time.time() < DEADLINE - 5:
        rng = random.Random(seed)
        S = greedy_maximal(rng)
        if is_maximal(S) and len(S) < best_len:
            S = local_search(S, DEADLINE)
            if is_maximal(S) and len(S) < best_len:
                best_len = len(S)
                best_set = S[:]
        seed += 1

    if best_set is None or not is_maximal(best_set):
        # Final safety: rebuild a guaranteed maximal set
        S = []
        sm = 0
        for p in range(NUM_PTS):
            if (sm >> p) & 1:
                continue
            blk = blocked_of(S)
            if (blk >> p) & 1:
                continue
            S.append(p)
            sm |= (1 << p)
        best_set = S

    return [POINTS[i] for i in best_set]
