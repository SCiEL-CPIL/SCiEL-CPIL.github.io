"""Size-efficient sorting networks on 16 wires - optimized search."""
import time
import numpy as np

N = 16
ALL_INPUTS = np.arange(65536, dtype=np.uint16)

# EVOLVE-BLOCK-START

def _batcher_odd_even_merge(n):
    """Generate Batcher odd-even merge sort network for n wires (power of 2)."""
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    half = n // 2
    left = _batcher_odd_even_merge(half)
    right = [(i + half, j + half) for (i, j) in left]
    merge = _odd_even_merge(n)
    return left + right + merge

def _odd_even_merge(n):
    """Odd-even merge network for n wires (power of 2, n >= 2)."""
    if n == 2:
        return [(0, 1)]
    half = n // 2
    odd = _odd_even_merge(half)
    even = _odd_even_merge(half)
    odd_shifted = [(i + 1, j + 1) for (i, j) in odd]
    cross = [(i, i + 1) for i in range(0, n - 1, 2)]
    return odd + even + odd_shifted + cross

def _batcher_16():
    """Full Batcher OES network for 16 wires."""
    return _batcher_odd_even_merge(16)

def _verify_network(comparators):
    """Verify sorting network using zero-one principle with numpy.
    Returns True if the network sorts all 2^16 Boolean inputs."""
    bits = ((ALL_INPUTS[:, None] >> np.arange(16, dtype=np.uint16)[None, :]) & 1).astype(np.int8)
    
    for (i, j) in comparators:
        mn = np.minimum(bits[:, i], bits[:, j])
        mx = np.maximum(bits[:, i], bits[:, j])
        bits[:, i] = mn
        bits[:, j] = mx
    
    diffs = np.diff(bits, axis=1)
    return not np.any(diffs < 0)

def _try_remove_one(comparators, idx):
    """Try removing comparator at index idx. Returns new list if valid, else None."""
    candidate = comparators[:idx] + comparators[idx+1:]
    if _verify_network(candidate):
        return candidate
    return None

def _greedy_remove(comparators, deadline):
    """Systematically try to remove comparators one at a time, scanning from start.
    Repeat until no more can be removed or deadline hit."""
    current = list(comparators)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(current)):
            if time.time() > deadline:
                break
            candidate = _try_remove_one(current, idx)
            if candidate is not None:
                current = candidate
                improved = True
                break  # Restart scan from beginning after a removal
    return current

def _try_swap(comparators, i, j):
    """Try swapping comparators at positions i and j. Returns new list if valid, else None."""
    candidate = list(comparators)
    candidate[i], candidate[j] = candidate[j], candidate[i]
    if _verify_network(candidate):
        return candidate
    return None

def _local_search(comparators, deadline):
    """Local search to minimize comparator count while maintaining correctness."""
    current = list(comparators)
    
    # Phase 1: Greedy removal
    current = _greedy_remove(current, deadline)
    
    # Phase 2: Try swaps to enable more removals
    while time.time() < deadline:
        progress = False
        for i in range(len(current)):
            if time.time() > deadline:
                break
            for j in range(i+1, len(current)):
                if time.time() > deadline:
                    break
                candidate = _try_swap(current, i, j)
                if candidate is not None:
                    current = candidate
                    progress = True
                    break
            if progress:
                break
        if not progress:
            # No swaps helped, try more removals
            new_current = _greedy_remove(current, deadline)
            if len(new_current) < len(current):
                current = new_current
                progress = True
            if not progress:
                break
    
    return current

def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    # Start with Batcher OES network
    base = _batcher_16()
    
    # Verify base network
    if not _verify_network(base):
        # Fallback to odd-even transposition
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    # Apply local search with time budget
    result = _local_search(base, deadline)
    
    # Final verification
    if not _verify_network(result):
        return base
    
    return result

# EVOLVE-BLOCK-END
