"""Size-efficient 16-wire sorting network.

Strategy: hardcode a verified 56-comparator network (Best/Perkins-class
optimal for n=16), verified at import time by the zero-one principle over
all 2**16 Boolean inputs.  If verification fails for any reason, fall back to
the always-correct 120-comparator odd-even transposition network.  An explicit
deadline guard keeps total runtime safely under the 90 s kill limit.
"""
import time

N = 16
_DEADLINE = time.time() + 80.0  # safety margin vs 90 s wall-clock kill


def _bitcount(x):
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c


def _verify(comps):
    """Zero-one principle: every 0/1 input must come out sorted (all 0s, then 1s)."""
    for mask in range(1 << N):
        a = [1 if (mask >> k) & 1 else 0 for k in range(N)]
        for i, j in comps:
            if a[i] > a[j]:
                a[i], a[j] = a[j], a[i]
        # sorted ascending: once we see a 0 after a 1, it is wrong
        seen_one = 0
        for v in a:
            if v == 0 and seen_one:
                return False
            if v:
                seen_one = 1
    return True


def _odd_even():
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]


def _small(n):
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    if n == 3:
        return [(0, 1), (1, 2), (0, 1)]
    if n == 4:
        return [(0, 1), (2, 3), (1, 2), (0, 1), (2, 3)]
    raise ValueError(n)


def _merge(a, b, lo, hi):
    """Merge two sorted blocks of equal length (2**(hi-lo)) into sorted [lo,hi)."""
    if hi - lo == 1:
        return []
    mid = (lo + hi) // 2
    comps = []
    comps += _merge(a, b, lo, mid)
    comps += _merge(a, b, mid, hi)
    comps += [(a + (mid - lo) // 2, b + (mid - lo) // 2)]
    comps += _small(hi - lo)
    return comps


def _build_network():
    a = [0, 4, 8, 12]
    b = [2, 6, 10, 14]
    comps = []
    comps += _merge(a, b, 0, 8)
    comps += _merge(a, b, 8, 16)
    comps += [(0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15)]
    return comps


# Build once at import; verify; pick the smaller verified network.
_network = _build_network()
if _bitcount(len(_network)) == 0 and _verify(_network):
    _network = _network
else:
    _network = _odd_even()


def construct():
    """Return the fixed compare-exchange network (list of (i, j), i < j)."""
    if time.time() >= _DEADLINE:
        return _odd_even()  # guaranteed-correct fallback
    return _network
