"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 18
NUM_BOOLS = 1 << N  # 262144

def _build_ooe_network(n):
    """Odd-even transposition network for n wires."""
    comps = []
    for phase in range(n):
        start = phase % 2
        for i in range(start, n - 1, 2):
            comps.append((i, i + 1))
    return comps

# Precompute initial masks and expected final masks once
_NUM_WORDS = (1 << N) // 64  # 4096

def _precompute_masks(n=N):
    """Precompute initial wire masks and expected final masks."""
    # Initial masks: mask[i] has bit k set iff (k >> i) & 1
    init_masks = np.zeros((n, _NUM_WORDS), dtype=np.uint64)
    for i in range(n):
        # Pattern: for bit position k, set iff (k >> i) & 1
        # Build word by word
        for w in range(_NUM_WORDS):
            base = 64 * w
            word = 0
            for bit in range(64):
                k = base + bit
                if (k >> i) & 1:
                    word |= (1 << bit)
            init_masks[i, w] = word
    
    # Expected final masks: after sorting, wire i = 1 iff popcount(k) >= n - i
    # We'll store expected bit counts per wire
    from math import comb
    expected_counts = []
    for i in range(n):
        count = 0
        for j in range(n - i, n + 1):
            count += comb(n, j)
        expected_counts.append(count)
    
    return init_masks, expected_counts

_INIT_MASKS, _EXPECTED_COUNTS = _precompute_masks()

def _verify_fast(comps, n=N):
    """Fast verification using precomputed masks. Returns True if correct."""
    # Copy initial masks
    masks = _INIT_MASKS.copy()
    
    for (a, b) in comps:
        masks[a] &= masks[b]
        masks[b] = masks[a] | masks[b]
    
    # Check monotonicity: mask[i] should be subset of mask[i+1]
    for i in range(n - 1):
        if np.any(masks[i] & ~masks[i + 1]):
            return False
    
    # Check boundary: wire 0 has exactly 1 bit set (all-1s input)
    bit_count_0 = sum(bin(w).count('1') for w in masks[0])
    if bit_count_0 != 1:
        return False
    
    # Wire n-1: all except all-zeros
    bit_count_last = sum(bin(w).count('1') for w in masks[n - 1])
    if bit_count_last != (1 << n) - 1:
        return False
    
    return True

def _verify(comps, n=N):
    """Verify using bitmask method. Returns True if correct."""
    return _verify_fast(comps, n)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # 8 second safety margin
    
    n = N
    comps = _build_ooe_network(n)
    
    # Verify initial network
    if not _verify_fast(comps, n):
        return comps
    
    best = list(comps)
    best_len = len(best)
    
    # Phase 1: Greedy removal - try to remove each comparator
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(comps)):
            if time.time() > deadline:
                break
            # Try removing comparator at idx
            trial = comps[:idx] + comps[idx+1:]
            if _verify_fast(trial, n):
                comps = trial
                improved = True
                best = list(comps)
                best_len = len(comps)
                break  # Restart from beginning after a removal
    
    # Phase 2: Try swapping pairs of comparators (not just adjacent) to find new redundancies
    # Focus on swaps that might enable further removals
    for _round in range(30):
        if time.time() > deadline:
            break
        found_improvement = False
        L = len(comps)
        # Try swapping pairs that are close together (within 5 positions)
        for idx in range(L - 1):
            if time.time() > deadline:
                break
            for jdx in range(idx + 1, min(idx + 6, L)):
                if time.time() > deadline:
                    break
                # Skip if same comparator
                if comps[idx] == comps[jdx]:
                    continue
                trial = comps[:]
                trial[idx], trial[jdx] = trial[jdx], trial[idx]
                # Quick check: is the swapped network still valid?
                if _verify_fast(trial, n):
                    # Try to remove a comparator from the swapped network
                    for j in range(len(trial)):
                        if time.time() > deadline:
                            break
                        trial2 = trial[:j] + trial[j+1:]
                        if _verify_fast(trial2, n):
                            comps = trial2
                            best = list(comps)
                            best_len = len(comps)
                            found_improvement = True
                            break
                    if found_improvement:
                        break
            if found_improvement:
                break
        if not found_improvement:
            break
    
    # Phase 3: Try batch removal - remove multiple comparators at once
    # Use a more aggressive approach: try removing comparators in groups
    for _round in range(20):
        if time.time() > deadline:
            break
        found_improvement = False
        L = len(comps)
        # Try removing pairs of comparators that are far apart
        for idx in range(L - 1):
            if time.time() > deadline:
                break
            for jdx in range(idx + 1, L):
                if time.time() > deadline:
                    break
                # Try removing both
                trial = comps[:idx] + comps[idx+1:jdx] + comps[jdx+1:]
                if _verify_fast(trial, n):
                    comps = trial
                    best = list(comps)
                    best_len = len(comps)
                    found_improvement = True
                    break
            if found_improvement:
                break
        if not found_improvement:
            break
    
    return best
# EVOLVE-BLOCK-END
