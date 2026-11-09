"""Size-efficient sorting network on 16 wires.
Strategy: start from Batcher odd-even merge sort (known-good, ~119 comparators),
then iteratively remove redundant comparators via a fast bit-parallel zero-one
verification. Keep the best valid network found within the time budget."""
import time
import numpy as np

N = 16
TOTAL_ZERO_ONE = 1 << N  # 65536

# Deadline guard: hard-kill safety margin before the 90s wall-clock limit.
T_START = time.time()
DEADLINE = T_START + 82.0  # 8s safety margin

# Pre-compute initial wire states as Python ints (65536-bit each).
# Wire k contains bit k of each input index.
_MASKS = []
for _k in range(N):
    _m = 0
    for _v in range(TOTAL_ZERO_ONE):
        if (_v >> _k) & 1:
            _m |= (1 << _v)
    _MASKS.append(_m)

# Pre-compute sorted target: for each input, the sorted output has
# (TOTAL_ZERO_ONE - popcount) zeros then popcount ones.
# Wire k is 1 for inputs where k >= (TOTAL_ZERO_ONE - popcount(input)).
# Equivalently, wire k is 1 for inputs where popcount(input) > TOTAL_ZERO_ONE - 1 - k.
# Let's pre-compute the target wire masks.
_TARGET_MASKS = []
for _k in range(N):
    _m = 0
    for _v in range(TOTAL_ZERO_ONE):
        _pc = bin(_v).count('1')
        # After sorting, wire k is 1 iff _pc > TOTAL_ZERO_ONE - 1 - k
        if _pc > (TOTAL_ZERO_ONE - 1 - _k):
            _m |= (1 << _v)
    _TARGET_MASKS.append(_m)


def _verify(comp_list):
    """Return True iff the compare-exchange network sorts every 0/1 input.
    Uses bit-parallel representation: each wire is a 65536-bit Python int."""
    if len(comp_list) == 0:
        return True
    wires = [m for m in _MASKS]  # Copy of initial state
    for i, j in comp_list:
        a = wires[i]
        b = wires[j]
        wires[i] = a & b
        wires[j] = a | b
    # Check if wires match the sorted target
    for k in range(N):
        if wires[k] != _TARGET_MASKS[k]:
            return False
    return True


def _batcher_oe_merge_sort(n):
    """Batcher's odd-even merge sort network for n wires (recursive)."""
    def oe_merge(left, right):
        """Merge two sorted halves (each of length h) into one sorted block."""
        comps = []
        h = len(left)
        if h == 1:
            # Odd-even transposition merge of two 1-wire halves.
            comps.append((left[0], right[0]))
            return comps
        # Split each half into even/odd indexed wires.
        le = [left[k] for k in range(0, h, 2)]
        lo = [left[k] for k in range(1, h, 2)]
        re = [right[k] for k in range(0, h, 2)]
        ro = [right[k] for k in range(1, h, 2)]
        # Recursively merge the even-indexed pairs and odd-indexed pairs.
        comps += oe_merge(le, re)
        comps += oe_merge(lo, ro)
        # Compare-exchange between the interleaved results.
        for k in range(1, h, 2):
            comps.append((left[k - 1], right[k - 1]))
        return comps

    def sort_block(wires):
        m = len(wires)
        if m == 1:
            return []
        h = m // 2
        left = wires[:h]
        right = wires[h:]
        comps = []
        comps += sort_block(left)
        comps += sort_block(right)
        comps += oe_merge(left, right)
        return comps

    return sort_block(list(range(n)))


def _remove_redundant(comp_list):
    """Try to drop each comparator; keep the network if it still sorts.
    Tries from the end first (more likely redundant)."""
    cur = list(comp_list)
    changed = True
    while changed and time.time() < DEADLINE:
        changed = False
        # Try from end to start
        i = len(cur) - 1
        while i >= 0 and time.time() < DEADLINE:
            trial = cur[:i] + cur[i + 1:]
            if _verify(trial):
                cur = trial
                changed = True
                # do not advance i; re-examine same index
            else:
                i -= 1
    return cur


def _remove_redundant_multi(comp_list):
    """Aggressive removal: try to remove multiple comparators per pass."""
    cur = list(comp_list)
    # First do single removals
    cur = _remove_redundant(cur)
    # Then try to remove pairs that might be jointly redundant
    while time.time() < DEADLINE:
        improved = False
        for i in range(len(cur)):
            if time.time() > DEADLINE:
                break
            for j in range(i + 1, len(cur)):
                if time.time() > DEADLINE:
                    break
                trial = cur[:i] + cur[i + 1:j] + cur[j + 1:]
                if _verify(trial):
                    cur = trial
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return cur


def construct():
    """Build the smallest verified 16-wire sorting network found in budget."""
    # Strong initial construction: Batcher odd-even merge sort.
    network = _batcher_oe_merge_sort(N)
    # Sanity: it must be valid.
    if not _verify(network):
        # Fallback to a guaranteed-valid transposition network.
        network = [(i, i + 1) for phase in range(N)
                   for i in range(phase % 2, N - 1, 2)]

    best = network
    # Iteratively strip redundant comparators.
    best = _remove_redundant(best)
    
    if time.time() < DEADLINE - 5.0:
        # Try multi-removal for further reduction
        best = _remove_redundant_multi(best)

    if time.time() < DEADLINE - 2.0:
        # Final safety: ensure the returned network is valid.
        if not _verify(best):
            best = network

    return [(int(i), int(j)) for (i, j) in best]
