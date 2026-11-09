import time
import random
from fractions import Fraction

N = 256
TIME_LIMIT = 80.0  # safety margin before 90s kill

# Precompute AP pairs: ap_pairs[x] = list of (a,b) such that {x,a,b} is a 3-AP, a,b != x
ap_pairs = [[] for _ in range(N + 1)]

for x in range(1, N + 1):
    pairs = set()
    # x first: x, x+d, x+2d
    for d in range(1, (N - x) // 2 + 1):
        y = x + d
        z = x + 2 * d
        if z <= N:
            pairs.add((y, z))
    # x middle: x-d, x, x+d
    for d in range(1, min(x - 1, N - x) + 1):
        y = x - d
        z = x + d
        if y >= 1:
            pairs.add((y, z))
    # x last: x-2d, x-d, x
    for d in range(1, x // 2 + 1):
        y = x - 2 * d
        z = x - d
        if y >= 1:
            pairs.add((y, z))
    ap_pairs[x] = list(pairs)


def can_add(bitset, x):
    for a, b in ap_pairs[x]:
        if (bitset >> (a - 1)) & 1 and (bitset >> (b - 1)) & 1:
            return False
    return True


recip_frac = [Fraction(0, 1)] * (N + 1)
recip_float = [0.0] * (N + 1)
for i in range(1, N + 1):
    recip_frac[i] = Fraction(1, i)
    recip_float[i] = 1.0 / i


def greedy(order):
    bitset = 0
    for x in order:
        if can_add(bitset, x):
            bitset |= (1 << (x - 1))
    return bitset


def bitset_score(bitset):
    s = 0.0
    for x in range(1, N + 1):
        if (bitset >> (x - 1)) & 1:
            s += recip_float[x]
    return s


def bitset_to_list(bitset):
    return [x for x in range(1, N + 1) if (bitset >> (x - 1)) & 1]


def polish(bitset, deadline):
    """Add any element that can be added without creating a 3-AP."""
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            if not ((bitset >> (x - 1)) & 1) and can_add(bitset, x):
                bitset |= (1 << (x - 1))
                changed = True
    return bitset


def local_search(bitset, deadline):
    """1-for-1 and 1-for-2 swaps to maximize reciprocal sum."""
    best = bitset
    best_score = bitset_score(bitset)
    improved = True
    iters = 0
    while improved and time.time() < deadline and iters < 40:
        improved = False
        iters += 1
        selected = [x for x in range(1, N + 1) if (best >> (x - 1)) & 1]
        unselected = [x for x in range(1, N + 1) if not (best >> (x - 1)) & 1]
        # Try removing each selected (largest first = least recip first), add best single
        for rem in sorted(selected, key=lambda x: recip_float[x]):
            if time.time() > deadline:
                return best
            trial = best & ~(1 << (rem - 1))
            base = best_score - recip_float[rem]
            # 1-for-1
            for add in unselected:
                if can_add(trial, add):
                    ns = base + recip_float[add]
                    if ns > best_score + 1e-12:
                        best = trial | (1 << (add - 1))
                        best_score = ns
                        improved = True
                        break
            # 1-for-2: replace rem with two smaller elements
            if not improved:
                best_two = None
                best_two_score = base
                for a in unselected:
                    if time.time() > deadline:
                        break
                    t1 = trial | (1 << (a - 1))
                    for b in unselected:
                        if b <= a:
                            continue
                        if can_add(t1, b):
                            ns = base + recip_float[a] + recip_float[b]
                            if ns > best_two_score:
                                best_two_score = ns
                                best_two = (a, b)
                if best_two is not None and best_two_score > best_score + 1e-12:
                    a, b = best_two
                    best = trial | (1 << (a - 1)) | (1 << (b - 1))
                    best_score = best_two_score
                    improved = True
    return best


def construct():
    start = time.time()
    deadline = start + TIME_LIMIT
    rng = random.Random(12345)

    candidates = []

    # Sorted greedy (smallest first)
    candidates.append(greedy(list(range(1, N + 1))))

    # Multi-start randomized greedy
    for _ in range(60):
        if time.time() > deadline - 10:
            break
        order = list(range(1, N + 1))
        noisy = [ (1.0 / x) * (0.4 + rng.random()) for x in order ]
        order.sort(key=lambda x: noisy[x - 1], reverse=True)
        candidates.append(greedy(order))

    # Pick best initial
    best = max(candidates, key=bitset_score)

    # Polish then local search
    best = polish(best, deadline)
    best = local_search(best, deadline)

    # Final polish
    best = polish(best, deadline)

    return bitset_to_list(best)
