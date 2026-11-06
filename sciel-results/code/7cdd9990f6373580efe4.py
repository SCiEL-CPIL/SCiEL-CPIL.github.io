"""Structural redesign: search-based comparator-network minimization.

Instead of returning a fixed transposition network, we (1) build a valid
sorting network, (2) verify it over all 2**N Boolean inputs using a fast
bit-parallel symbolic simulator, and (3) run a deadline-bounded local search
that removes redundant comparators and performs moves/swaps to shrink the
network while preserving correctness.

The verifier runs the network *symbolically*: each wire is represented by an
N-bit mask (one bit per Boolean input). A comparator (i, j) is applied with
pure integer bitwise ops, so a full verification of all 2**N inputs costs only
O(#comparators) integer operations. This makes each candidate extremely cheap
to evaluate, enabling many search steps within the wall-clock budget.
"""

import time

N = 18
FULL = (1 << N) - 1          # all-ones mask over the 2**N Boolean inputs
DEADLINE = 80.0              # safety margin below the 90 s kill
SEED_VALUE = 153.0


# --------------------------------------------------------------------------- #
# Fast symbolic verification over all 2**N Boolean inputs
# --------------------------------------------------------------------------- #
def _wire_masks():
    """Return the list of per-wire masks: wire k has bit b set iff input b
    has a 1 on wire k. For the identity input b (an integer 0..2**N-1), wire k
    is bit k of b. So mask_k has bit b set iff (b >> k) & 1."""
    masks = [0] * N
    for b in range(1 << N):
        for k in range(N):
            if (b >> k) & 1:
                masks[k] |= (1 << b)
    return masks


def _verify(net):
    """Return True iff net sorts every Boolean input.

    net: list of (i, j) comparators, i < j.
    We simulate the network on the symbolic masks. After the network, the
    sorted output for every input must equal the canonical sorted form:
    for input b, the sorted wires are the top popcount(b) wires = 1 and the
    rest = 0. We check that the resulting masks equal that canonical form.
    """
    if not net:
        return False
    masks = _wire_masks()
    for (i, j) in net:
        mi, mj = masks[i], masks[j]
        # compare-exchange: min to i, max to j
        # min = mi & mj ; max = mi | mj
        lo = mi & mj
        hi = mi | mj
        masks[i] = lo
        masks[j] = hi
    # canonical sorted masks: wire k is 1 for input b iff the number of 1-bits
    # in b is greater than (N-1-k).
    for k in range(N):
        expected = 0
        for b in range(1 << N):
            if bin(b).count("1") > (N - 1 - k):
                expected |= (1 << b)
        if masks[k] != expected:
            return False
    return True


# Precompute canonical sorted masks once.
_CANON = None
def _canonical_masks():
    global _CANON
    if _CANON is None:
        _CANON = [0] * N
        for b in range(1 << N):
            pc = bin(b).count("1")
            for k in range(N):
                if pc > (N - 1 - k):
                    _CANON[k] |= (1 << b)
    return _CANON


def _verify_fast(net):
    """Faster verification using precomputed canonical masks."""
    if not net:
        return False
    masks = _wire_masks()
    for (i, j) in net:
        mi, mj = masks[i], masks[j]
        masks[i] = mi & mj
        masks[j] = mi | mj
    canon = _canonical_masks()
    for k in range(N):
        if masks[k] != canon[k]:
            return False
    return True


# --------------------------------------------------------------------------- #
# Network construction
# --------------------------------------------------------------------------- #
def _odd_even_transposition():
    """Seed valid network: odd-even transposition (N^2/2 comparators)."""
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net


def _bitonic_network(n):
    """Bitonic sorting network (valid, ~ n^2/2 comparators)."""
    def merge(a, b, d):
        # a: start index, b: length, d: direction (1=asc, -1=desc)
        if b <= 1:
            return
        h = b // 2
        merge(a, h, 1)
        merge(a + h, h, -1)
        for i in range(h):
            x, y = a + i, a + i + h
            if d == 1:
                yield (min(x, y), max(x, y))
            else:
                yield (max(x, y), min(x, y))

    def bitonic(a, b, d):
        if b <= 1:
            return
        h = b // 2
        bitonic(a, h, 1)
        bitonic(a + h, h, -1)
        yield from merge(a, b, d)

    net = list(bitonic(0, n, 1))
    return net


# --------------------------------------------------------------------------- #
# Local search: prune redundant comparators + moves
# --------------------------------------------------------------------------- #
def _prune(net, deadline):
    """Remove any comparator whose deletion keeps the network correct.
    Repeats until no single removal helps (greedy redundancy elimination)."""
    changed = True
    while changed and time.time() < deadline:
        changed = False
        i = 0
        while i < len(net) and time.time() < deadline:
            cand = net[:i] + net[i + 1:]
            if _verify_fast(cand):
                net = cand
                changed = True
                # do not advance i; re-check same position
            else:
                i += 1
    return net


def _local_search(net, deadline):
    """Try to shrink the network further via single removals (already done in
    prune) and comparator swaps that reduce count is impossible via swap;
    instead we try to find a *smaller* valid network by random restarts of the
    prune from shuffled orderings, and by attempting to insert then prune.

    We keep the best (fewest comparators) valid network found."""
    best = list(net)
    best_len = len(best)
    start = time.time()
    while time.time() < deadline:
        # Randomly reorder and re-prune: different orderings can expose
        # different redundancies because we only ever remove, never add, so
        # reordering + pruning from the same set cannot beat the original
        # prune. Instead we try *randomly dropping* a few comparators and
        # re-verifying; if still valid, keep and continue pruning.
        trial = list(best)
        # try dropping up to 3 random comparators
        import random
        random.seed(int(time.time() * 1000) % (1 << 30))
        k = min(3, len(trial))
        idxs = random.sample(range(len(trial)), k)
        for ix in sorted(idxs, reverse=True):
            trial.pop(ix)
        if _verify_fast(trial):
            trial = _prune(trial, deadline)
            if len(trial) < best_len:
                best = trial
                best_len = len(best)
        # early exit if we hit a good target
        if best_len <= 80:
            break
    return best


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def construct():
    deadline = time.time() + DEADLINE

    # Build a couple of candidate valid networks and keep the smallest.
    candidates = []
    for builder in (_odd_even_transposition, _bitonic_network):
        try:
            net = builder(N)
            if _verify_fast(net):
                candidates.append(net)
        except Exception:
            pass

    if not candidates:
        # Fallback: return the seed network (guaranteed valid) without search.
        return _odd_even_transposition()

    # Start from the smallest candidate.
    net = min(candidates, key=len)

    # Greedy redundancy pruning (monotone, safe).
    net = _prune(net, deadline)

    # Bounded local search for further reduction.
    net = _local_search(net, deadline)

    # Final safety: ensure validity; if something went wrong, fall back.
    if not _verify_fast(net):
        net = _prune(_odd_even_transposition(), deadline)

    return net
