"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 16
DEADLINE = 85.0  # safety margin before 90s kill

def _apply_comparators(comps, x):
    """Apply compare-exchange comparators to input vector x."""
    x = x.copy()
    for (i, j) in comps:
        a, b = x[i], x[j]
        if a > b:
            x[i], x[j] = b, a
    return x

def _verify_network(comps):
    """Verify sorting network using zero-one principle (all 2^N binary inputs)."""
    n = len(comps)
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

def _batcher_oe_merge_sort(n):
    """Batcher's odd-even merge sort network."""
    def oe_merge(A, B):
        """Merge two sorted sequences A and B (each of length n/2)."""
        if len(A) == 1:
            return A
        mid = len(A) // 2
        A1, A2 = A[:mid], A[mid:]
        B1, B2 = B[:mid], B[mid:]
        # Recursively merge
        M1 = oe_merge(A1, B1)
        M2 = oe_merge(A2, B2)
        # Cross comparators
        cross = []
        for i in range(mid):
            cross.append((M1[i], M2[mid - 1 - i]))
        return cross
    
    def build(n):
        """Build sorting network for n wires."""
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        if n % 2 != 0:
            # Handle odd: sort n-1 wires, then merge with the last
            base = build(n - 1)
            # Add comparators to merge wire n-1 into the sorted network
            extra = [(i, n - 1) for i in range(n - 1)]
            return base + extra
        # Even: split into two halves, sort each, then odd-even merge
        half = n // 2
        left = build(half)
        right = [(i + half, j + half) for (i, j) in build(half)]
        # Odd-even merge of two sorted halves
        merge = _oe_merge_two_sorted(half)
        return left + right + merge
    
    def _oe_merge_two_sorted(half):
        """Odd-even merge of two sorted halves of length `half`."""
        if half == 1:
            return [(0, 1)]
        mid = half // 2
        # First half: merge elements 0,2,4,... from left with 0,2,4,... from right
        # Second half: merge elements 1,3,5,... from left with 1,3,5,... from right
        result = []
        # Odd positions (0-indexed even indices): merge left[0],left[2],... with right[0],right[2],...
        # Actually, odd-even merge works by interleaving
        # For two sorted sequences of length h:
        # Compare (left[i], right[h-1-i]) for appropriate i
        # Standard odd-even merge:
        if half == 2:
            return [(0, 2), (1, 3), (1, 2)]
        # Recursive odd-even merge
        # Split into odd and even indexed elements
        # Left even: 0, 2, 4, ... ; Left odd: 1, 3, 5, ...
        # Right even: half, half+2, ... ; Right odd: half+1, half+3, ...
        
        # Merge even-indexed: left[0],left[2],... with right[half],right[half+2],...
        # These form two sorted sequences of length mid
        # Then merge odd-indexed: left[1],left[3],... with right[half+1],right[half+3],...
        
        # Actually, let me just use a known construction
        # For n=4: (0,2),(1,3),(1,2)
        # For n=8: (0,4),(1,5),(2,6),(3,7) then merge evens and odds
        # This is getting complex. Let me use a different approach.
        pass
    
    return build(n)

def _build_batcher(n):
    """Build Batcher's odd-even merge sort network."""
    def oe_merge_network(n):
        """Odd-even merge network for 2n inputs (n from top half, n from bottom half)."""
        if n == 1:
            return [(0, 1)]
        # Recursively merge
        # Compare (i, n+i) for i in 0..n-1
        comps = [(i, n + i) for i in range(n)]
        # Then recursively merge even-indexed and odd-indexed subsequences
        # Even indices: 0, 2, 4, ... -> map to 0, 1, 2, ...
        # Odd indices: 1, 3, 5, ... -> map to 0, 1, 2, ...
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
            # Insert element n-1 into sorted network of n-1 elements
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

def _local_search(comps, max_iters=50000, deadline=None):
    """Local search to minimize comparator count while maintaining correctness."""
    best = list(comps)
    best_count = len(best)
    random.seed(42)
    
    start_time = time.time()
    iters = 0
    no_improve = 0
    
    while iters < max_iters:
        if deadline is not None and time.time() > deadline:
            break
        
        # Try random removal of a comparator
        if len(best) > 1:
            idx = random.randint(0, len(best) - 1)
            trial = best[:idx] + best[idx+1:]
            if _verify_network(trial):
                best = trial
                best_count = len(best)
                no_improve = 0
            else:
                no_improve += 1
        
        iters += 1
        
        if no_improve > 10000:
            break
    
    return best

def construct():
    # Start with Batcher's odd-even merge sort
    comps = _build_batcher(N)
    initial_count = len(comps)
    
    if not _verify_network(comps):
        # Fallback to odd-even transposition
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    # Local search to reduce comparators
    deadline = time.time() + (DEADLINE - time.time() - 2.0)  # 2s safety margin
    best = _local_search(comps, max_iters=100000, deadline=deadline)
    
    return best

# EVOLVE-BLOCK-END
