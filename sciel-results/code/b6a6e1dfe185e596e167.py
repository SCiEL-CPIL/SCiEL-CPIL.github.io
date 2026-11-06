"""Size-efficient sorting network on 18 wires.

Searches for a minimal compare-exchange network that sorts all 2^18 Boolean
inputs.  Verification is fully vectorized with NumPy (all 262,144 inputs at
once), so each candidate can be checked in a few milliseconds, permitting a
large number of redundant-comparator removals and local-search moves within
the wall-clock budget.
"""
import time
import random
import numpy as np

N = 18
DEADLINE = 90.0          # evaluator wall-clock limit
SAFETY = 5.0             # stop this many seconds early
TOTAL_INPUTS = 1 << N

# Precompute the full Boolean input matrix: row k is the binary pattern of k.
# bits[j] = (k >> j) & 1  ->  shape (262144, 18) of uint8.
def _build_inputs():
    k = np.arange(TOTAL_INPUTS, dtype=np.uint32)
    bits = np.zeros((TOTAL_INPUTS, N), dtype=np.uint8)
    for j in range(N):
        bits[:, j] = (k >> j) & 1
    return bits

INP = _build_inputs()


def apply_net(net, bits):
    """Apply a comparator list to the full Boolean matrix (in place copy)."""
    b = bits.copy()
    for (i, j) in net:
        bi = b[:, i]
        bj = b[:, j]
        b[:, i] = np.minimum(bi, bj)
        b[:, j] = np.maximum(bi, bj)
    return b


def is_sorted(b):
    """True if every row is non-decreasing along the wires."""
    return bool(np.all(b[:, :-1] <= b[:, 1:]))


def verify(net):
    """Correct iff the sorted output equals the per-row sorted input."""
    b = apply_net(net, INP)
    # A correct sorting network must sort each row; for Boolean inputs the
    # sorted row is all-zeros then all-ones, i.e. non-decreasing.
    return is_sorted(b)


def odd_even_transposition():
    """Classic odd-even transposition network (correct, 153 comparators)."""
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net


def redundant_indices(net):
    """Return the set of comparator indices whose removal keeps the network
    sorting all 2^18 Boolean inputs.  Vectorized over the candidates."""
    m = len(net)
    if m == 0:
        return set()
    # Precompute the state after each prefix (state[k] = output after net[:k]).
    # We need the input to comparator k, which is state[k].
    states = [None] * (m + 1)
    b = INP.copy()
    states[0] = b
    for k in range(m):
        i, j = net[k]
        bi = b[:, i]
        bj = b[:, j]
        b[:, i] = np.minimum(bi, bj)
        b[:, j] = np.maximum(bi, bj)
        states[k + 1] = b

    # For each candidate k, simulate skipping it:
    #   state_before = states[k]
    #   then apply net[k+1:]
    # We do this by, for each k, copying states[k] and applying the suffix.
    # To keep it fast, process candidates in a loop but the per-candidate work
    # is a vectorized suffix application.  m is at most ~153, suffix length
    # shrinks, so total work is manageable.
    red = set()
    for k in range(m):
        b = states[k].copy()
        i, j = net[k]
        # skip comparator k; apply the rest
        for (ii, jj) in net[k + 1:]:
            bi = b[:, ii]
            bj = b[:, jj]
            b[:, ii] = np.minimum(bi, bj)
            b[:, jj] = np.maximum(bi, bj)
        if is_sorted(b):
            red.add(k)
    return red


def greedy_remove(net):
    """Repeatedly remove any redundant comparator until none remain."""
    net = list(net)
    changed = True
    while changed:
        changed = False
        red = redundant_indices(net)
        if not red:
            break
        # remove one (the last index to keep earlier structure; order is
        # arbitrary among redundant ones)
        k = max(red)
        net.pop(k)
        changed = True
    return net


def local_search(net, deadline, best_net, best_len, rng):
    """Try to improve `net` by adding a random comparator then removing
    redundancies.  Returns (improved_net, improved_len) or None."""
    while time.time() < deadline:
        base = list(net)
        base_len = len(base)
        # add a random comparator
        i = rng.randrange(N)
        j = rng.randrange(N)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        cand = base + [(i, j)]
        # also try inserting it at a random position (order matters)
        pos = rng.randrange(len(cand) + 1)
        cand = base[:pos] + [(i, j)] + base[pos:]
        # remove redundancies
        cand = greedy_remove(cand)
        if len(cand) < base_len and verify(cand):
            return cand
    return None


def reorder_and_trim(net, deadline, rng):
    """Attempt to move a comparator later (which can expose redundancy in the
    earlier part) and trim.  Returns improved net or None."""
    m = len(net)
    while time.time() < deadline and m > 1:
        k = rng.randrange(m - 1)
        cand = list(net)
        # move comparator k one step later
        cand[k], cand[k + 1] = cand[k + 1], cand[k]
        cand = greedy_remove(cand)
        if len(cand) < m and verify(cand):
            return cand
    return None


def construct():
    start = time.time()
    deadline = start + (DEADLINE - SAFETY)
    rng = random.Random(12345)

    # Start from a correct seed network and trim it aggressively.
    net = odd_even_transposition()
    assert verify(net), "seed network must be correct"
    net = greedy_remove(net)
    assert verify(net), "greedy removal must preserve correctness"
    best_net = list(net)
    best_len = len(net)

    # Local search: add-then-trim and reorder-then-trim.
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # add-then-trim
        res = local_search(best_net, deadline, best_net, best_len, rng)
        if res is not None:
            best_net = res
            best_len = len(res)
            improved = True
            continue
        # reorder-then-trim
        res = reorder_and_trim(best_net, deadline, rng)
        if res is not None:
            best_net = res
            best_len = len(res)
            improved = True

    # Final safety verification (should always hold).
    if not verify(best_net):
        # Fallback to a known-correct network if something went wrong.
        best_net = odd_even_transposition()

    return list(best_net)


# Provide a convenient depth report (not required by the interface, but the
# contract mentions parallel depth is reported separately).
def depth(net):
    """Compute the parallel depth (longest path) of the network."""
    dist = [0] * N
    for (i, j) in net:
        d = max(dist[i], dist[j]) + 1
        dist[i] = max(dist[i], d)
        dist[j] = max(dist[j], d)
    return max(dist) if N else 0
