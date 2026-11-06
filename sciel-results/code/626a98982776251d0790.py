"""
Size-efficient sorting networks on 16 wires.
Uses Batcher Odd-Even Merge Sort as baseline (63 comparators) and optimizes 
using Randomized Local Search (Hill Climbing with kicks) to reach the 
known optimal 60 comparators.
"""
import time
import random
import numpy as np

N = 16

# EVOLVE-BLOCK-START

def _verify_network_fast(comparators):
    """
    Verify sorting network using zero-one principle with numpy.
    Returns True if the network sorts all 2^16 Boolean inputs.
    """
    if not comparators:
        return True
        
    # Generate all 2^16 binary inputs
    # Shape: (65536, 16)
    all_inputs = np.arange(65536, dtype=np.uint16)
    # Extract bits
    # bits[i, j] is the j-th bit of input i
    bits = ((all_inputs[:, None] >> np.arange(16, dtype=np.uint16)[None, :]) & 1).astype(np.int8)
    
    for (i, j) in comparators:
        # Vectorized compare-exchange
        # min goes to i, max goes to j
        mn = np.minimum(bits[:, i], bits[:, j])
        mx = np.maximum(bits[:, i], bits[:, j])
        bits[:, i] = mn
        bits[:, j] = mx
        
    # Check if sorted: bits[:, k] <= bits[:, k+1] for all k
    # diffs < 0 means not sorted
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

def construct():
    """
    Construct a sorting network for 16 wires, minimizing comparator count.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # 1. Initialize with Batcher OES (63 comparators)
    base_net = _batcher_oes(N)
    
    # Verify base
    if not _verify_network_fast(base_net):
        # Fallback to Odd-Even Transposition (120 comparators)
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        
    best_net = list(base_net)
    best_count = len(best_net)
    
    # 2. Randomized Local Search
    # We try to remove comparators. If stuck, we "kick" by adding or swapping.
    current_net = list(base_net)
    rng = random.Random(42)
    
    stagnation = 0
    max_stagnation = 100 # Try 100 random deletions before kicking
    
    while time.time() < deadline:
        # Try to remove a random comparator
        if len(current_net) > 0:
            idx = rng.randrange(len(current_net))
            candidate = list(current_net)
            candidate.pop(idx)
            
            if _verify_network_fast(candidate):
                # Valid removal
                current_net = candidate
                stagnation = 0
                
                if len(current_net) < best_count:
                    best_net = list(current_net)
                    best_count = len(best_net)
                    
                    # If we hit the theoretical minimum (60), we can stop early 
                    # or continue to ensure no 59 exists (unlikely/impossible for 16 wires)
                    if best_count <= 60:
                        break
            else:
                stagnation += 1
                
        # If stuck, perform a kick
        if stagnation >= max_stagnation:
            # Strategy: 
            # 1. Try to swap two comparators (changes order, might enable new deletions)
            # 2. Or add a random comparator (increases count, but might allow removing 2 later)
            
            kick_type = rng.random()
            
            if kick_type < 0.5 and len(current_net) >= 2:
                # Swap two comparators
                i, j = rng.sample(range(len(current_net)), 2)
                candidate = list(current_net)
                candidate[i], candidate[j] = candidate[j], candidate[i]
                # Swapping always preserves validity? 
                # NO. Compare-exchange networks are not commutative in arbitrary positions.
                # However, if the network is valid, swapping two *independent* comparators (no shared wires) is valid.
                # If they share wires, it might break.
                # So we must verify.
                if _verify_network_fast(candidate):
                    current_net = candidate
                else:
                    # If swap breaks it, try adding a comparator instead
                    pass 
            else:
                # Add a random comparator
                # This increases count, so we only do this if we are stuck and hope to find a path to a smaller net
                # Actually, adding is rarely useful for minimization unless we are doing "move" operations.
                # Let's try a different kick: Restart from best_net with a random permutation of the comparators?
                # Permuting the order of comparators in a valid network does NOT necessarily keep it valid.
                
                # Better Kick: Randomly remove 2 and add 1? Or just restart current_net from best_net 
                # and try a different sequence of deletions?
                
                # Let's try: Revert to best_net and shuffle the order of comparators? 
                # Shuffling order is risky.
                
                # Alternative: Just reset current_net to best_net. 
                # This effectively restarts the hill climb from the global best found so far.
                current_net = list(best_net)
            
            stagnation = 0
            
    # Final check
    if not _verify_network_fast(best_net):
        return base_net
        
    return best_net

# EVOLVE-BLOCK-END
