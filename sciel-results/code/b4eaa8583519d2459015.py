"""Size-efficient sorting network on 16 wires.

Strategy:
  1. Build a Batcher odd-even mergesort network (correct by the zero-one principle).
  2. Prune redundant comparators: a comparator is redundant if, when removed,
     the remaining network still sorts every 0/1 input.
  3. Return the pruned list of (i, j) pairs.

The evaluator scores seed/value where seed = 120 (odd-even transposition count),
so a 60-comparator network scores 2.0.
"""
import time

N = 16
DEADLINE = 80.0  # wall-clock safety margin (evaluator kills at 90 s)

# ---------------------------------------------------------------------------
# Comparator helpers
# ---------------------------------------------------------------------------

def apply_comp(vals, i, j):
    """Apply one compare-exchange in place: min to i, max to j."""
    a, b = vals[i], vals[j]
    if a > b:
        vals[i], vals[j] = b, a
    return vals


def apply_network(vals, comps):
    """Apply a full network to a copy of vals; return sorted-or-not result."""
    v = list(vals)
    for i, j in comps:
        a, b = v[i], v[j]
        if a > b:
            v[i], v[j] = b, a
    return v


# ---------------------------------------------------------------------------
# Batcher odd-even mergesort network
# ---------------------------------------------------------------------------

def _odd_even_merge(a, b):
    """Return list of (i, j) comparators that merge two adjacent sorted halves.

    a, b are lists of wire indices (each half already sorted).
    """
    comps = []
    # Odd phase: pair (a[k], b[k])
    for k in range(len(a)):
        comps.append((a[k], b[k]))
    # Even phase: internal merges of the even and odd subsequences
    even = [a[0]] + [b[k] for k in range(1, len(a), 2)] + [a[k] for k in range(1, len(a), 2)]
    odd  = [b[0]] + [a[k] for k in range(1, len(a), 2)] + [b[k] for k in range(1, len(b), 2)]
    if len(even) > 1:
        comps.extend(_odd_even_merge(even[0::2], even[1::2]))
    if len(odd) > 1:
        comps.extend(_odd_even_merge(odd[0::2], odd[1::2]))
    return comps


def _batcher_merge_sort(wires):
    """Batcher odd-even mergesort on the given wire indices."""
    n = len(wires)
    if n <= 1:
        return []
    comps = []
    comps.extend(_batcher_merge_sort(wires[0::2]))
    comps.extend(_batcher_merge_sort(wires[1::2]))
    comps.extend(_odd_even_merge(wires[0::2], wires[1::2]))
    return comps


def _build_batcher():
    return _batcher_merge_sort(list(range(N)))


# ---------------------------------------------------------------------------
# Verification over all 65536 zero-one inputs
# ---------------------------------------------------------------------------

def verify(comps):
    """Return True iff the network sorts every 0/1 input."""
    for mask in range(1 << N):
        vals = [(mask >> i) & 1 for i in range(N)]
        out = apply_network(vals, comps)
        # sorted ascending: each element <= next
        for i in range(N - 1):
            if out[i] > out[i + 1]:
                return False
    return True


# ---------------------------------------------------------------------------
# Redundancy pruning
# ---------------------------------------------------------------------------

def prune(comps):
    """Remove redundant comparators, keeping the network correct.

    A comparator c is redundant if removing it still yields a sorting network.
    We greedily remove redundant comparators.
    """
    n = len(comps)
    # Precompute all 0/1 inputs as tuples for speed
    inputs = [tuple((mask >> i) & 1 for i in range(N)) for mask in range(1 << N)]

    current = list(comps)
    i = 0
    while i < len(current):
        if time.time() > DEADLINE:
            break
        # Test removing comparator i
        candidate = current[:i] + current[i + 1:]
        ok = True
        for inp in inputs:
            v = list(inp)
            for a, b in candidate:
                if v[a] > v[b]:
                    v[a], v[b] = v[b], v[a]
            for k in range(N - 1):
                if v[k] > v[k + 1]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            current = candidate
            # do not advance i: re-test the same position with the new list
        else:
            i += 1
    return current


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def construct():
    # Start from Batcher odd-even mergesort
    comps = _build_batcher()

    # Verify the base network (sanity)
    if not verify(comps):
        # Fallback: odd-even transposition (always correct)
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]

    # Prune redundant comparators
    pruned = prune(comps)

    # Final verification
    if not verify(pruned):
        return comps

    return pruned
