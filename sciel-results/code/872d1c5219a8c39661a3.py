"""Fresh design: Batcher bitonic seed + NumPy-verified randomized local search."""
import time
import random
import numpy as np

N = 18
DEADLINE = 78.0  # safety margin before the 90 s kill

# ---- Fast NumPy verifier: checks all 2**N Boolean inputs at once ----
def _apply(net, A):
    """A: (M, N) uint8 matrix of inputs; returns sorted (min,max) wires."""
    for (i, j) in net:
        ai = A[:, i]
        aj = A[:, j]
        A[:, i] = np.minimum(ai, aj)
        A[:, j] = np.maximum(ai, aj)
    return A

def _is_valid(net):
    M = 1 << N
    idx = np.arange(M, dtype=np.uint8)
    A = np.frombuffer(idx.tobytes(), dtype=np.uint8).reshape(M, N)
    B = _apply(net, A)
    return bool((B == np.sort(B, axis=1)).all())

def _make_bitonic(seed):
    """Build a Batcher bitonic merge network for N wires (0-indexed pairs)."""
    rng = random.Random(seed)

    def merge(lo, hi, step):
        if step == 1:
            return
        for i in range(lo, hi, 2 * step):
            for j in range(step):
                yield (i + j, i + step + j)
        for i in range(lo, hi, 2 * step):
            yield from merge(i, i + step, step // 2)
            yield from merge(i + step, i + 2 * step, step // 2)

    def bitonic(lo, hi, step):
        if step == 1:
            return
        for i in range(lo, hi, 2 * step):
            for j in range(step):
                yield (i + j, i + step + j)
        yield from bitonic(lo, hi, step // 2)
        yield from bitonic(hi - step, hi, step // 2)

    # Build a full bitonic (ascending) network for N wires.
    # Use the standard construction: repeatedly merge to create bitonic then sort.
    # Simplest correct construction: use the "bitonic sort" recursive structure.
    def bitonic_sort(lo, hi, direction):
        if hi - lo <= 1:
            return
        mid = (lo + hi) // 2
        bitonic_sort(lo, mid, 1)
        bitonic_sort(mid, hi, -1)
        # merge [lo, hi) which is now bitonic
        def merge_bitonic(a, b, step):
            if step == 1:
                for k in range(a, b):
                    yield (k, k + step) if k + step < hi else None
            for i in range(a, b, 2 * step):
                for j in range(step):
                    if i + step + j < hi:
                        yield (i + j, i + step + j)
                if step > 1:
                    for p in merge_bitonic(i, i + step, step // 2):
                        yield p
                    for p in merge_bitonic(i + step, min(i + 2 * step, hi), step // 2):
                        yield p
        for c in merge_bitonic(lo, hi, (hi - lo) // 2):
            if c is not None:
                yield c

    net = []
    for c in bitonic_sort(0, N, 1):
        net.append((c[0], c[1]))
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for c in net:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def _initial_network():
    """Try several Batcher seeds and keep the smallest valid one."""
    best = None
    for s in range(0, 40):
        net = _make_bitonic(s)
        if not net:
            continue
        if _is_valid(net):
            if best is None or len(net) < len(best):
                best = net
    if best is None:
        # Fallback: odd-even transposition (known valid)
        best = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    return list(best)


def construct():
    t0 = time.time()
    rng = random.Random(12345)

    net = _initial_network()
    if not _is_valid(net):
        net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    best = list(net)
    best_len = len(best)

    # Precompute all possible comparator pairs for replace/move targets
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    n_pairs = len(all_pairs)

    # Local search: iterate moves until deadline
    while time.time() < DEADLINE:
        # Choose a random move type
        r = rng.random()
        moved = False
        if r < 0.55 and len(net) > 1:
            # Remove a random comparator
            k = rng.randrange(len(net))
            cand = net[:k] + net[k + 1:]
            if _is_valid(cand):
                net = cand
                moved = True
        elif r < 0.75 and len(net) >= 2:
            # Swap two comparators
            a, b = rng.sample(range(len(net)), 2)
            cand = net[:]
            cand[a], cand[b] = net[b], net[a]
            if _is_valid(cand):
                net = cand
                moved = True
        elif r < 0.9:
            # Move a comparator to a new position
            k = rng.randrange(len(net))
            pos = rng.randrange(len(net) + 1)
            c = net[k]
            cand = net[:k] + net[k + 1:]
            cand.insert(pos, c)
            if _is_valid(cand):
                net = cand
                moved = True
        else:
            # Replace a comparator with a random one
            k = rng.randrange(len(net))
            newc = all_pairs[rng.randrange(n_pairs)]
            cand = net[:]
            cand[k] = newc
            if _is_valid(cand):
                net = cand
                moved = True

        if moved and len(net) < best_len:
            best = list(net)
            best_len = len(net)

    return best
