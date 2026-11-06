import time
import random
import numpy as np

N = 18
NUM_BOOLS = 1 << N
SEED = 153.0
DEADLINE = 85.0  # wall-clock safety margin (evaluator kills at 90s)

# ---------------------------------------------------------------------------
# Fast verification using bitmasks: each wire is a 262144-bit integer where
# bit x is 1 iff wire is 1 on Boolean input x.
# Comparator (i, j): w_i' = w_i & w_j, w_j' = w_i | w_j.
# Sorted iff wire 0 has bit set on every input with at least one 1.
# ---------------------------------------------------------------------------
_ALL_ONES = (1 << NUM_BOOLS) - 1
_ANY_BIT = 1  # bit 0 = all-zeros input

def _initial_masks():
    """Wire i's mask: bit x set iff bit i of x is 1."""
    masks = [0] * N
    for i in range(N):
        m = 0
        # Pattern: for each block of size 2^(i+1), the upper half has bit i set
        half = 1 << i
        full = half << 1
        # Build by setting bits in [half, full) within each block of size full
        for start in range(0, NUM_BOOLS, full):
            for b in range(half):
                m |= 1 << (start + half + b)
        masks[i] = m
    return masks

def _verify(masks):
    """Check if wire 0 is 1 on every input that has at least one 1."""
    # Wire 0 must have all bits set except possibly bit 0 (all-zeros input)
    # Actually: sorted means the maximum element is on wire 0.
    # For Boolean inputs, max is 1 iff any wire is 1.
    # So wire 0 must equal OR of all wires. Since OR of all wires = _ALL_ONES & ~_ANY_BIT... 
    # Wait: on the all-zeros input, all wires are 0, so wire 0 should be 0.
    # On every other input, at least one wire is 1, so wire 0 must be 1.
    # So wire 0 should have all bits set except bit 0.
    return (masks[0] & (~_ANY_BIT)) == (_ALL_ONES & ~_ANY_BIT)

def _apply(masks, comps):
    for i, j in comps:
        a, b = masks[i], masks[j]
        masks[i] = a & b
        masks[j] = a | b
    return masks

def _verify_comps(comps):
    return _verify(_apply(_initial_masks(), comps))

def _batcher_oe_merge_sort(n, offset=0):
    """Batcher's odd-even merge sort network for n wires."""
    if n <= 1:
        return []
    comps = []
    half = n // 2
    comps.extend(_batcher_oe_merge_sort(half, offset))
    comps.extend(_batcher_oe_merge_sort(n - half, offset + half))
    comps.extend(_merge_batcher(half, n - half, offset))
    return comps

def _merge_batcher(k, m, offset):
    """Merge two sorted blocks of length k and m."""
    if k == 0 or m == 0:
        return []
    if k == 1 and m == 1:
        return [(offset, offset + 1)]
    comps = []
    k2 = (k + 1) // 2
    m2 = (m + 1) // 2
    comps.extend(_merge_batcher(k2, m2, offset))
    if k >= 2:
        comps.extend(_merge_batcher(k - k2, m - m2, offset + 1))
    for p in range(offset, offset + k + m, 2):
        if p + 1 < offset + k + m:
            comps.append((p, p + 1))
    return comps

def _greedy_remove(comps, budget):
    """Greedy removal: try removing each comparator, keep if still valid."""
    masks = _apply(_initial_masks(), comps)
    kept = []
    for idx, (i, j) in enumerate(comps):
        if time.time() > budget:
            kept.extend(comps[idx:])
            break
        a, b = masks[i], masks[j]
        # Remove comparator
        masks[i] = a | b
        masks[j] = a
        if _verify(masks):
            continue  # keep it removed
        else:
            # Restore
            masks[i] = a & b
            masks[j] = a | b
            kept.append((i, j))
    return kept

def _random_removal(comps, budget):
    """Randomized removal pass."""
    masks = _apply(_initial_masks(), comps)
    kept = []
    order = list(range(len(comps)))
    random.shuffle(order)
    for idx in order:
        if time.time() > budget:
            # Add remaining
            remaining = [comps[k] for k in range(len(comps)) if k not in set(order[:order.index(idx)])]
            # Simpler: just break and add rest
            break
        i, j = comps[idx]
        a, b = masks[i], masks[j]
        masks[i] = a | b
        masks[j] = a
        if _verify(masks):
            continue
        else:
            masks[i] = a & b
            masks[j] = a | b
            kept.append((i, j))
    return kept

def _swap_comparators(comps, budget):
    """Try swapping two comparators to see if it still sorts."""
    masks = _apply(_initial_masks(), comps)
    n = len(comps)
    if n < 2:
        return comps
    # Try a few random swaps
    for _ in range(min(50, n)):
        if time.time() > budget:
            break
        i, j = random.sample(range(n), 2)
        a, b = comps[i], comps[j]
        # Swap
        comps[i], comps[j] = b, a
        if _verify_comps(comps):
            pass  # keep swap
        else:
            # Revert
            comps[i], comps[j] = a, b
    return comps

def _add_comparator(comps, budget):
    """Try adding a random comparator (for use in more advanced search)."""
    # This is a placeholder; we focus on removal
    return comps

def construct():
    start = time.time()
    budget = start + DEADLINE

    # Seed: Batcher's odd-even merge sort
    comps = _batcher_oe_merge_sort(N)
    if not _verify_comps(comps):
        # Fallback to odd-even transposition
        comps = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                comps.append((i, i + 1))
    if not _verify_comps(comps):
        return comps

    # Phase 1: Deterministic greedy removal
    while time.time() < budget - 5.0:
        new = _greedy_remove(comps, budget - 5.0)
        if len(new) >= len(comps):
            break
        comps = new

    # Phase 2: Randomized removal passes
    while time.time() < budget - 2.0:
        new = _random_removal(comps, budget - 2.0)
        if len(new) >= len(comps):
            break
        comps = new

    # Phase 3: Try swapping comparators
    while time.time() < budget - 1.0:
        new = _swap_comparators(comps, budget - 1.0)
        if len(new) != len(comps):
            comps = new
        else:
            # No improvement from swapping, stop
            break

    # Guarantee validity
    if not _verify_comps(comps):
        comps = _batcher_oe_merge_sort(N)

    return comps
