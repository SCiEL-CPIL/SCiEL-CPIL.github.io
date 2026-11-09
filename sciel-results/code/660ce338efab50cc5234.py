"""Size-efficient sorting networks on 16 wires: construct a small verified network."""
import time

N = 16
DEADLINE = time.time() + 80.0  # explicit wall-clock safety margin (limit is 90s)


def _odd_even_merge(a, b):
    """Odd-even merge of two sorted lists (each a list of wire indices)."""
    if not a:
        return list(b)
    if not b:
        return list(a)
    if len(a) == 1:
        return a + b
    if len(b) == 1:
        return a + b
    even_a = a[::2]
    odd_a = a[1::2]
    even_b = b[::2]
    odd_b = b[1::2]
    even_merged = _odd_even_merge(even_a, even_b)
    odd_merged = _odd_even_merge(odd_a, odd_b)
    out = []
    for i in range(len(even_merged) + len(odd_merged)):
        if i < len(even_merged):
            out.append(even_merged[i])
        if i < len(odd_merged):
            out.append(odd_merged[i])
    return out


def _build_odd_even_merge_sort():
    """Construct odd-even merge sort network as a list of comparator pairs."""
    comps = []

    def sort(k):
        if k == 1:
            return [0]
        half = k // 2
        left = sort(half)
        right = [h + half for h in sort(half)]
        merged = _odd_even_merge(left, right)
        # comparator for each adjacent pair in merged order
        for i in range(k - 1):
            a, b = merged[i], merged[i + 1]
            if a > b:
                a, b = b, a
            comps.append((a, b))
        return merged

    sort(N)
    return comps


def _simulate(comps, x):
    """Apply comparator network to 16-bit mask x; return resulting mask."""
    for a, b in comps:
        va = (x >> a) & 1
        vb = (x >> b) & 1
        if va == 0 and vb == 1:
            x ^= (1 << a) | (1 << b)
    return x


def _is_sorted(x):
    # sorted means all 1s are in the low bits: x == (1<<k)-1 for some k
    return x == 0 or (x & (x + 1)) == 0


def _verify_all(comps):
    """Full zero-one principle check over all 2**N inputs."""
    for x in range(1 << N):
        if not _is_sorted(_simulate(comps, x)):
            return False
    return True


def _random_sample(comps, samples=20000):
    """Randomized zero-one check; returns failing input or None."""
    import random
    rng = random.Random(12345)
    for _ in range(samples):
        x = rng.getrandbits(N)
        if not _is_sorted(_simulate(comps, x)):
            return x
    return None


def construct():
    # Build a strong base network.
    comps = _build_odd_even_merge_sort()

    # First ensure the base network is correct (it should be).
    if not _verify_all(comps):
        # Fallback: odd-even transposition (guaranteed correct, 120 comps).
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]

    # Prune redundant comparators: try removing each, keep if still correct.
    # Iterate to a fixed point, respecting the deadline.
    i = 0
    while i < len(comps) and time.time() < DEADLINE:
        c = comps[i]
        candidate = comps[:i] + comps[i + 1:]
        # Quick randomized pre-check before expensive full verification.
        fail = _random_sample(candidate, samples=20000)
        if fail is None:
            # Full verification to be safe.
            if _verify_all(candidate):
                comps = candidate
                continue  # do not advance i; re-examine same position
        i += 1

    return comps


# EVOLVE-BLOCK-START
# (construct defined above; kept as the entry point.)
# EVOLVE-BLOCK-END
