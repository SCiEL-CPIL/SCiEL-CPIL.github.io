import numpy as np
import time
import random

N = 18
NUM_WORDS = (1 << N) // 64  # 4096

def _precompute_initial_masks(n=N):
    """Precompute the initial wire masks for n wires."""
    num_words = (1 << n) // 64
    masks = np.zeros((n, num_words), dtype=np.uint64)
    for i in range(n):
        for w in range(num_words):
            base = 64 * w
            word = 0
            for bit in range(64):
                k = base + bit
                if (k >> i) & 1:
                    word |= (1 << bit)
            masks[i, w] = word
    return masks

def _precompute_expected_masks(n=N):
    """Precompute expected masks after sorting: wire i has 1 iff popcount(k) >= n-i."""
    num_words = (1 << n) // 64
    expected = np.zeros((n, num_words), dtype=np.uint64)
    # For each input k, popcount(k) determines which wires should be 1
    # Wire i should be 1 iff popcount(k) >= n - i
    # So for input k with popcount p, wires [n-p, n-1] should be 1
    
    # Build by iterating over all k
    for k in range(1 << n):
        p = bin(k).count('1')
        start_wire = n - p
        for i in range(start_wire, n):
            word_idx = k // 64
            bit_idx = k % 64
            expected[i, word_idx] |= (1 << bit_idx)
    return expected

# Precompute
_INITIAL_MASKS = _precompute_initial_masks()
_EXPECTED_MASKS = _precompute_expected_masks()

def _verify_fast(comps, n=N):
    """Fast verification using precomputed masks."""
    num_words = (1 << n) // 64
    masks = _INITIAL_MASKS.copy()
    
    for (a, b) in comps:
        masks[a] &= masks[b]
        masks[b] = masks[a] | masks[b]
    
    # Check against expected
    for i in range(n):
        if not np.array_equal(masks[i], _EXPECTED_MASKS[i]):
            return False
    return True

def _build_initial_network(n):
    """Build a reasonable initial network."""
    # Use odd-even transposition as baseline
    comps = []
    for phase in range(n):
        start = phase % 2
        for i in range(start, n - 1, 2):
            comps.append((i, i + 1))
    return comps

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = N
    comps = _build_initial_network(n)
    
    # Verify initial
    if not _verify_fast(comps, n):
        return comps
    
    best = list(comps)
    best_len = len(best)
    
    # Phase 1: Aggressive removal
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(comps)):
            if time.time() > deadline:
                break
            trial = comps[:idx] + comps[idx+1:]
            if _verify_fast(trial, n):
                comps = trial
                best = list(comps)
                best_len = len(comps)
                improved = True
                break  # Restart
    
    # Phase 2: Try replacing comparators
    for _round in range(100):
        if time.time() > deadline:
            break
        found = False
        for idx in range(len(comps)):
            if time.time() > deadline:
                break
            a, b = comps[idx]
            # Try replacing with a different comparator
            for new_a in range(n):
                if time.time() > deadline:
                    break
                for new_b in range(new_a + 1, n):
                    if (new_a, new_b) == (a, b):
                        continue
                    trial = comps[:]
                    trial[idx] = (new_a, new_b)
                    if _verify_fast(trial, n):
                        # Now try to remove
                        for j in range(len(trial)):
                            if time.time() > deadline:
                                break
                            trial2 = trial[:j] + trial[j+1:]
                            if _verify_fast(trial2, n):
                                comps = trial2
                                best = list(comps)
                                best_len = len(comps)
                                found = True
                                break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if not found:
            break
    
    return best
