"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 16

# EVOLVE-BLOCK-START

def _verify_network_fast(comparators):
    """Verify sorting network using zero-one principle with numpy.
    Returns True if the network sorts all 2^16 Boolean inputs."""
    all_inputs = np.arange(65536, dtype=np.uint8)
    bits = ((all_inputs[:, None] >> np.arange(16)[None, :]) & 1).astype(np.int8)
    for (i, j) in comparators:
        mn = np.minimum(bits[:, i], bits[:, j])
        mx = np.maximum(bits[:, i], bits[:, j])
        bits[:, i] = mn
        bits[:, j] = mx
    diffs = np.diff(bits, axis=1)
    if np.any(diffs < 0):
        return False
    return True

def _batcher_oes(n):
    """Batcher odd-even merge sort for n wires (power of 2)."""
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    half = n // 2
    left = _batcher_oes(half)
    right = [(i + half, j + half) for (i, j) in left]
    merge = _odd_even_merge(n)
    return left + right + merge

def _odd_even_merge(n):
    """Odd-even merge for n wires (power of 2, n >= 2)."""
    if n == 2:
        return [(0, 1)]
    half = n // 2
    odd = _odd_even_merge(half)
    even = _odd_even_merge(half)
    odd_shifted = [(i + 1, j + 1) for (i, j) in odd]
    cross = [(i, i + 1) for i in range(0, n - 1, 2)]
    return odd + even + odd_shifted + cross

def _local_search(network, time_budget=80.0, seed=42):
    """
    Stochastic Local Search to minimize comparator count.
    Moves: Remove, Swap, Move, Add.
    Acceptance: Valid network. Prefer smaller size.
    """
    deadline = time.time() + time_budget
    rng = random.Random(seed)
    current = list(network)
    best = list(current)
    best_count = len(current)
    
    if not _verify_network_fast(current):
        return current

    while time.time() < deadline - 2.0:
        move_type = rng.choice(['remove', 'swap', 'move', 'add'])
        candidate = list(current)
        
        if move_type == 'remove':
            if len(candidate) <= 50: # Don't go below 50 blindly
                continue
            idx = rng.randrange(len(candidate))
            candidate.pop(idx)
            
        elif move_type == 'swap':
            if len(candidate) < 2:
                continue
            i, j = rng.sample(range(len(candidate)), 2)
            candidate[i], candidate[j] = candidate[j], candidate[i]
            
        elif move_type == 'move':
            if len(candidate) < 2:
                continue
            i = rng.randrange(len(candidate))
            j = rng.randrange(len(candidate))
            if i == j:
                continue
            comp = candidate.pop(i)
            candidate.insert(j, comp)
            
        elif move_type == 'add':
            if len(candidate) > 70: # Cap size to avoid bloat
                continue
            i, j = rng.sample(range(N), 2)
            if i > j:
                i, j = j, i
            candidate.append((i, j))
        
        # Verify candidate
        if _verify_network_fast(candidate):
            if len(candidate) < best_count:
                best = candidate
                best_count = len(candidate)
                current = candidate
            elif len(candidate) == best_count:
                # Explore same size
                if rng.random() < 0.5:
                    current = candidate
            else:
                # Accept larger with low probability to escape local minima?
                # Only if we are stuck for a while? For now, reject larger.
                pass
        else:
            # Reject invalid
            pass
            
    return best

def construct():
    """Construct a sorting network for 16 wires, minimizing comparator count."""
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Start with Batcher OES (known correct, 63 comparators)
    base = _batcher_oes(16)
    
    if not _verify_network_fast(base):
        # Fallback to odd-even transposition (120 comparators, definitely correct)
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    best = base
    best_count = len(base)
    
    # Run Local Search
    if time.time() < deadline - 5.0:
        result = _local_search(base, time_budget=min(75.0, deadline - time.time() - 5.0), seed=123)
        if _verify_network_fast(result) and len(result) < best_count:
            best = result
            best_count = len(result)
            
    # Final verification
    if not _verify_network_fast(best):
        return base
    
    return best

# EVOLVE-BLOCK-END
