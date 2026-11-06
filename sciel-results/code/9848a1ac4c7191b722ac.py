"""Size-efficient 18-wire sorting network via two-stage verified local search."""
import time
import random

N = 18
DEADLINE = 84.0  # wall-clock safety margin under the 90s kill
_rng = random.Random(20240517)


# ---------- verification ----------
def _verify_full(net):
    """True iff net sorts every 2^N boolean input."""
    n = len(net)
    for mask in range(1 << N):
        w = [1 if (mask >> i) & 1 else 0 for i in range(N)]
        for a, b in net:
            wa, wb = w[a], w[b]
            if wa > wb:
                w[a], w[b] = wb, wa
        for i in range(N - 1):
            if w[i] > w[i + 1]:
                return False
    return True


def _verify_hard(net, hard):
    """True iff net sorts every input in the hard subset."""
    n = len(net)
    for mask in hard:
        w = [1 if (mask >> i) & 1 else 0 for i in range(N)]
        for a, b in net:
            wa, wb = w[a], w[b]
            if wa > wb:
                w[a], w[b] = wb, wa
        for i in range(N - 1):
            if w[i] > w[i + 1]:
                return False
    return True


def _build_hard():
    """A strong hard subset: all masks, all single-bit flips, all rotations."""
    hard = set()
    for m in range(1 << N):
        hard.add(m)
        # single-bit flips
        for b in range(N):
            hard.add(m ^ (1 << b))
        # rotations
        for s in range(1, N):
            hard.add(((m << s) | (m >> (N - s))) & ((1 << N) - 1))
    return list(hard)


# ---------- seeds ----------
def _seed_odd_even():
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]


def _seed_bitonic():
    comps = []
    for k in range(1, N):
        for i in range(0, N, 2 * k):
            for j in range(i, i + k):
                comps.append((j, j + k))
    for k in range(1, N):
        for i in range(0, N, 2 * k):
            for j in range(i + k - 1, i - 1, -1):
                comps.append((j, j + k))
    return comps


def _seed_batcher():
    comps = []
    def merge(a, b, lo, hi):
        if hi - lo <= 1:
            return
        mid = (lo + hi) // 2
        merge(a, b, lo, mid)
        merge(a, b, mid, hi)
        if mid - lo >= 2:
            comps.append((lo + mid - 1, mid))
        for i in range(1, mid - lo):
            comps.append((lo + i - 1, mid + i - 1))
        for i in range(1, hi - mid):
            comps.append((lo + i, mid + i))
    merge(list(range(N)), list(range(N)), 0, N)
    return comps


# ---------- search ----------
def construct():
    t0 = time.time()
    hard = _build_hard()

    seeds = [_seed_odd_even(), _seed_bitonic(), _seed_batcher()]
    seeds = [s for s in seeds if _verify_full(s)]
    if not seeds:
        return _seed_odd_even()

    best = min(seeds, key=len)
    best_len = len(best)

    while time.time() - t0 < DEADLINE:
        net = list(best)
        cur_len = len(net)
        improved = True
        while improved and time.time() - t0 < DEADLINE:
            improved = False
            # --- removal pass ---
            i = _rng.randrange(cur_len)
            while i < cur_len:
                cand = net[:i] + net[i + 1:]
                if _verify_hard(cand, hard):
                    if _verify_full(cand):
                        net = cand
                        cur_len -= 1
                        best = list(net)
                        best_len = cur_len
                        improved = True
                    i = 0
                    continue
                i += 1
            if time.time() - t0 >= DEADLINE:
                break
            # --- swap pass ---
            for _ in range(3000):
                if time.time() - t0 >= DEADLINE:
                    break
                a = _rng.randrange(cur_len)
                b = _rng.randrange(cur_len)
                if a == b:
                    continue
                cand = net[:a] + net[a + 1:b + 1] + [net[a]] + net[b + 1:]
                if _verify_hard(cand, hard):
                    if _verify_full(cand):
                        net = cand
                        best = list(net)
                        best_len = cur_len
                        improved = True
                        break
            if time.time() - t0 >= DEADLINE:
                break
            # --- move (reorder) pass ---
            for _ in range(2000):
                if time.time() - t0 >= DEADLINE:
                    break
                a = _rng.randrange(cur_len)
                b = _rng.randrange(cur_len)
                if a == b:
                    continue
                cand = net[:a] + net[a + 1:b + 1] + [net[a]] + net[b + 1:]
                if _verify_hard(cand, hard):
                    if _verify_full(cand):
                        net = cand
                        best = list(net)
                        best_len = cur_len
                        improved = True
                        break
            if time.time() - t0 >= DEADLINE:
                break
            # --- insert (grow) pass: only if we are stuck ---
            if not improved and cur_len < 4096:
                for _ in range(500):
                    if time.time() - t0 >= DEADLINE:
                        break
                    a = _rng.randrange(N)
                    b = _rng.randrange(N)
                    if a == b:
                        continue
                    pos = _rng.randrange(cur_len + 1)
                    cand = net[:pos] + [(a, b)] + net[pos:]
                    if _verify_hard(cand, hard) and _verify_full(cand):
                        net = cand
                        cur_len += 1
                        best = list(net)
                        best_len = cur_len
                        improved = True
                        break

    # final safety: guarantee a valid network is returned
    if not _verify_full(best):
        for s in seeds:
            if _verify_full(s):
                best = s
                break
    return best
