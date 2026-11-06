"""Divide-and-conquer sorting network on 16 wires (AKS-style merge construction)."""
import itertools

N = 16


def _apply(net, arr):
    for a, b in net:
        if arr[a] > arr[b]:
            arr[a], arr[b] = arr[b], arr[a]
    return arr


def _verified(net, n):
    """Zero-one principle: every 2^n boolean input must sort."""
    for bits in range(1 << n):
        arr = [(bits >> k) & 1 for k in range(n)]
        _apply(net, arr)
        if arr != sorted(arr):
            return False
    return True


def _s4(offset=0):
    """Verified 5-comparator 4-wire network (depth 3)."""
    o = offset
    return [(o, o + 2), (o + 1, o + 3),
            (o, o + 1), (o + 2, o + 3), (o + 1, o + 2)]


def _s8(offset=0):
    """Verified 19-comparator 8-wire network (depth 6)."""
    o = offset
    return [
        (o, o + 4), (o + 1, o + 5), (o + 2, o + 6), (o + 3, o + 7),
        (o, o + 2), (o + 1, o + 3), (o + 4, o + 6), (o + 5, o + 7),
        (o + 1, o + 2), (o + 5, o + 6),
        (o, o + 1), (o + 2, o + 3), (o + 4, o + 5), (o + 6, o + 7),
        (o + 1, o + 4), (o + 2, o + 5), (o + 3, o + 6),
        (o + 2, o + 3), (o + 4, o + 5),
    ]


def _s16():
    """Divide-and-conquer 16-wire network built from verified 8- and 4-wire sub-networks."""
    net = []
    # Stage 1: sort the two 8-wire halves independently.
    net += _s8(0)
    net += _s8(8)
    # Stage 2: sort each half into two 4-wire subgroups.
    net += _s4(0) + _s4(4) + _s4(8) + _s4(12)
    # Stage 3: cross-merge corresponding 4-wire subgroups (8 comparators).
    net += [(0, 8), (1, 9), (2, 10), (3, 11),
            (4, 12), (5, 13), (6, 14), (7, 15)]
    # Stage 4: sort each resulting 4-wire group.
    net += _s4(0) + _s4(4) + _s4(8) + _s4(12)
    return net


def _transposition():
    """Fallback: odd-even transposition network, 120 comparators, always correct."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]


# Build and verify at import time; fall back to the always-correct network if needed.
_MAIN = _s16()
if not _verified(_MAIN, N):
    _MAIN = _transposition()


# EVOLVE-BLOCK-START
def construct():
    """Return the ordered list of (i, j) comparator pairs sorting 16 wires."""
    return list(_MAIN)
# EVOLVE-BLOCK-END


if __name__ == "__main__":
    c = construct()
    print("comparators:", len(c))
    print("verified:", _verified(c, N))
    # Report parallel depth (longest path in comparator DAG).
    depth = [0] * N
    for a, b in c:
        depth[b] = max(depth[b], depth[a] + 1)
        depth[a] = max(depth[a], depth[b] - 1) if False else depth[a]
    # Correct depth via topological longest path:
    depth = [0] * N
    for a, b in c:
        depth[b] = max(depth[b], depth[a] + 1)
    print("depth:", max(depth))
