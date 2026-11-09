import time
import random
import numpy as np

N = 16
DEADLINE = 85.0  # Safety margin before 90s kill

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
    # Generate all 2^16 binary inputs as a (65536, 16) array
    # Using a trick to generate efficiently
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

def _greedy_reduce(comps, deadline):
    """Try to remove redundant comparators greedily."""
    current = list(comps)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(current)):
            if time.time() > deadline:
                return current
            # Try removing comparator at index i
            trial = current[:i] + current[i+1:]
            if _verify_network(trial):
                current = trial
                changed = True
                break # Restart loop as indices changed
    return current

def _local_search(comps, deadline):
    """Local search to minimize comparator count."""
    best = list(comps)
    best_count = len(best)
    random.seed(42)
    
    start_time = time.time()
    no_improve = 0
    
    while time.time() < deadline:
        # Try random removal
        if len(best) > 1:
            idx = random.randint(0, len(best) - 1)
            trial = best[:idx] + best[idx+1:]
            if _verify_network(trial):
                best = trial
                best_count = len(best)
                no_improve = 0
                continue
        
        # Try moving a comparator to a different position
        if len(best) > 1:
            idx = random.randint(0, len(best) - 1)
            comp = best[idx]
            trial = best[:idx] + best[idx+1:]
            new_idx = random.randint(0, len(trial))
            trial.insert(new_idx, comp)
            if _verify_network(trial):
                # Check if it reduced count (it didn't, but might enable future removals)
                # For now, we accept moves that maintain correctness, hoping to break local minima
                # But strictly, we want to minimize count. So we only accept if count decreases?
                # No, moves keep count same. We accept moves if they allow subsequent removals.
                # Simplest: accept move if it doesn't break correctness, then try to remove again.
                best = trial
                no_improve = 0
                continue
        
        no_improve += 1
        if no_improve > 5000:
            break
            
    return best

def construct():
    # Start with Batcher's odd-even merge sort
    comps = _build_batcher(N)
    
    if not _verify_network(comps):
        # Fallback to odd-even transposition
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    # 1. Greedy reduction
    deadline_greedy = time.time() + 20.0
    comps = _greedy_reduce(comps, deadline_greedy)
    
    # 2. Local search with moves
    deadline_local = time.time() + (DEADLINE - time.time() - 5.0)
    comps = _local_search(comps, deadline_local)
    
    # 3. Final greedy sweep
    deadline_final = time.time() + (DEADLINE - time.time() - 2.0)
    comps = _greedy_reduce(comps, deadline_final)
    
    return comps
