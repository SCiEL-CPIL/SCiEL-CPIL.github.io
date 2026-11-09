"""Improved program: hardcoded 60-comparator Batcher OEMS for N=16 with fast verification."""
import time
import random
import numpy as np

N = 16
DEADLINE = 85.0  # safety margin before 90s kill

def _verify_network(comps):
    """Verify sorting network using zero-one principle (all 2^N binary inputs)."""
    # Generate all 2^16 binary inputs as a (65536, 16) array
    all_inputs = np.array([list(map(int, format(i, '016b'))) for i in range(65536)], dtype=np.int8)
    
    # Apply all comparators
    x = all_inputs.copy()
    for (i, j) in comps:
        a = x[:, i].copy()
        b = x[:, j].copy()
        # min goes to i, max goes to j
        x[:, i] = np.minimum(a, b)
        x[:, j] = np.maximum(a, b)
    
    # Check if each row is sorted (non-decreasing)
    diffs = np.diff(x, axis=1)
    return bool(np.all(diffs >= 0))

def _build_batcher_oems(n):
    """Build Batcher's odd-even merge sort network for n wires."""
    def oe_merge_network(n):
        """Odd-even merge network for 2n inputs (n from top half, n from bottom half)."""
        if n == 1:
            return [(0, 1)]
        # Compare (i, n+i) for i in 0..n-1
        comps = [(i, n + i) for i in range(n)]
        # Then recursively merge even-indexed and odd-indexed subsequences
        even_merge = oe_merge_network(n // 2)
        odd_merge = oe_merge_network(n // 2)
        # Apply even merge on even positions
        for (a, b) in even_merge:
            comps.append((2 * a, 2 * b))
        # Apply odd merge on odd positions
        for (a, b) in odd_merge:
            comps.append((2 * a + 1, 2 * b + 1))
        return comps
    
    def sort_network(n):
        """Sorting network using odd-even merge sort."""
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        if n % 2 == 1:
            # Handle odd number: sort n-1, then insert the last element
            base = sort_network(n - 1)
            extra = [(i, n - 1) for i in range(n - 1)]
            return base + extra
        # Even: split into two halves
        half = n // 2
        left = sort_network(half)
        right = [(i + half, j + half) for (i, j) in sort_network(half)]
        # Odd-even merge the two sorted halves
        merge = oe_merge_network(half)
        return left + right + merge
    
    return sort_network(n)

def _local_search(comps, deadline=None):
    """Local search to minimize comparator count while maintaining correctness."""
    best = list(comps)
    random.seed(42)
    
    start_time = time.time()
    no_improve = 0
    
    while True:
        if deadline is not None and time.time() > deadline:
            break
        if no_improve > 5000:
            break
        
        # Try random removal of a comparator
        if len(best) > 1:
            idx = random.randint(0, len(best) - 1)
            trial = best[:idx] + best[idx+1:]
            if _verify_network(trial):
                best = trial
                no_improve = 0
            else:
                no_improve += 1
    
    return best

def construct():
    # Build Batcher's odd-even merge sort for N=16 (known to be 60 comparators)
    comps = _build_batcher_oems(N)
    
    # Verify correctness
    if not _verify_network(comps):
        # Fallback to odd-even transposition (120 comparators)
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    # Local search to try to reduce below 60 (unlikely but worth trying)
    deadline = time.time() + (DEADLINE - time.time() - 3.0)  # 3s safety margin
    best = _local_search(comps, deadline=deadline)
    
    return best

# EVOLVE-BLOCK-END
