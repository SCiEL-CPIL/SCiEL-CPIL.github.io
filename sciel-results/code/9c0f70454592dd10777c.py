"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 84.0

    # ---------- structural base sets ----------
    # A: upper block, internally sum-free (min pair sum 131 > 128)
    A = set(range(65, 129))
    # B: lower block, internally sum-free (min pair sum 35 > 32)
    B = set(range(17, 33))
    # gap: only place a forbidden triple can be created (its pair sums reach A)
    GAP = list(range(33, 65))

    def is_valid_addition(s, candidate):
        """Check if adding candidate to s keeps the set valid."""
        s_with = s | {candidate}
        for a in s:
            if a == candidate:
                continue
            for b in s:
                if b <= a:
                    continue
                if (a + b) in s_with and (a + candidate) in s_with and (b + candidate) in s_with:
                    return False
        return True

    def greedy_fill(base, order, deadline):
        """Add elements from order into base while valid."""
        S = set(base)
        for x in order:
            if time.time() > deadline:
                break
            if x not in S and is_valid_addition(S, x):
                S.add(x)
        return S

    def swap_improve(S, deadline):
        """Try to remove one element and add two others to grow the set."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in sorted(S):
                if time.time() > deadline:
                    break
                S_remove = S - {x}
                for y in GAP:
                    if y in S:
                        continue
                    if not is_valid_addition(S_remove, y):
                        continue
                    S_temp = S_remove | {y}
                    for z in GAP:
                        if z == y or z in S_temp:
                            continue
                        if is_valid_addition(S_temp, z):
                            S = S_temp | {z}
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return S

    def complement_pass(S, deadline):
        """Try removing a single element, then greedily refilling the gap.
        Removing one blocker can unlock several previously-forbidden elements."""
        best = S
        for x in sorted(S):
            if time.time() > deadline:
                break
            S2 = S - {x}
            S2 = greedy_fill(S2, GAP, deadline)
            if len(S2) > len(best):
                best = S2
        return best

    def local_search(base, order, deadline):
        S = greedy_fill(base, order, deadline)
        S = swap_improve(S, deadline)
        S = complement_pass(S, deadline)
        S = swap_improve(S, deadline)
        return S

    # ---------- build a pool of candidate base sets ----------
    bases = [A | B]

    # (1) shrink A from the bottom: dropping some of the lowest A elements
    #     lets some gap elements (whose pair sums hit those A values) in.
    for drop in range(1, 21):
        bases.append(set(range(65 + drop, 129)) | B)

    # (2) extend B downward: extra small elements are sum-free with everything.
    for extra in range(1, 17):
        bases.append(A | set(range(17 - extra, 33)))

    # (3) drop a few highest B elements (their pair sums land in the gap).
    for drop in range(1, 6):
        bases.append(A | set(range(17, 33 - drop)))

    # (4) a few random bases.
    rng = random.Random(7)
    for _ in range(8):
        lo = rng.randint(65, 75)
        hi = rng.randint(10, 20)
        bases.append(set(range(lo, 129)) | set(range(hi, 33)))

    # deduplicate
    seen = set()
    uniq_bases = []
    for b in bases:
        key = frozenset(b)
        if key not in seen:
            seen.add(key)
            uniq_bases.append(b)
    bases = uniq_bases

    # ---------- search ----------
    best_set = None
    best_size = -1

    def consider(S):
        nonlocal best_set, best_size
        if S is not None and len(S) > best_size:
            best_size = len(S)
            best_set = S

    # deterministic orders
    det_orders = [
        list(range(33, 65)),
        list(range(64, 32, -1)),
        list(range(33, 49)) + list(range(64, 48, -1)),
        list(range(64, 48, -1)) + list(range(33, 49)),
    ]

    # always process the natural base first with the natural order
    consider(local_search(bases[0], det_orders[0], deadline))

    rng2 = random.Random(2024)

    # Phase A: every base with the deterministic orders (bounded time).
    for bi, base in enumerate(bases):
        for oi, order in enumerate(det_orders):
            if time.time() > deadline:
                break
            consider(local_search(base, order, deadline))

    # Phase B: random orders for the most promising bases.
    promising = [bases[0]]
    for b in bases[1:]:
        if len(b) >= 80:
            promising.append(b)

    for base in promising:
        for _ in range(30):
            if time.time() > deadline:
                break
            order = GAP[:]
            rng2.shuffle(order)
            consider(local_search(base, order, deadline))

    # Phase C: more random bases if time remains.
    while time.time() < deadline:
        lo = rng2.randint(65, 78)
        hi = rng2.randint(8, 22)
        base = set(range(lo, 129)) | set(range(hi, 33))
        order = GAP[:]
        rng2.shuffle(order)
        consider(local_search(base, order, deadline))

    if best_set is None:
        best_set = A | B

    result = [int(x) for x in sorted(best_set)]
    return result
# EVOLVE-BLOCK-END
