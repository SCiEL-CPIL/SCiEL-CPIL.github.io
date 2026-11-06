"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

N = 18
TOTAL_BITS = 1 << N  # 262144

def _sorted_mask_for_count(k, n=N):
    """Return the bitmask representing a sorted sequence with k ones in n wires."""
    if k == 0:
        return 0
    if k == n:
        return (1 << n) - 1
    return ((1 << k) - 1) << (n - k)

# Precompute the sorted target for each input mask
_popcount = np.frombuffer(bytearray(1 << N), dtype=np.uint8)
for i in range(1, 1 << N):
    _popcount[i] = _popcount[i >> 1] + (i & 1)

_targets = np.empty(1 << N, dtype=np.uint32)
for k in range(N + 1):
    target = _sorted_mask_for_count(k)
    _targets[_popcount == k] = target

def _verify(network):
    """Verify that the network sorts all 262144 Boolean inputs. Returns True if valid."""
    if not network:
        return False
    inputs = np.arange(1 << N, dtype=np.uint32)
    for (i, j) in network:
        bit_i = (inputs >> i) & 1
        bit_j = (inputs >> j) & 1
        swap = (bit_i == 1) & (bit_j == 0)
        if np.any(swap):
            inputs = np.where(swap, (inputs & ~(1 << i)) | (1 << j), inputs)
            inputs = np.where(swap, (inputs & ~(1 << j)) | (1 << i), inputs)
    return np.all(inputs == _targets)

def _verify_fast(network):
    """Faster verification using bit-parallel simulation with early exit."""
    if not network:
        return False
    inputs = np.arange(1 << N, dtype=np.uint32)
    for (i, j) in network:
        mask_i = 1 << i
        mask_j = 1 << j
        # Only swap where bit_i=1 and bit_j=0
        swap_mask = (inputs & mask_i) & ~(inputs & mask_j)
        if swap_mask.any():
            # Clear bit_i and set bit_j
            inputs = (inputs & ~mask_i) | ((swap_mask >> i) << j)
            # Clear bit_j and set bit_i (but only where we swapped)
            inputs = (inputs & ~mask_j) | ((swap_mask >> j) << i)
    return np.array_equal(inputs, _targets)

def _odd_even_transposition(n=N):
    """Odd-even transposition sorting network for n wires."""
    return [(i, i + 1) for phase in range(n)
            for i in range(phase % 2, n - 1, 2)]

def _batcher_odd_even_merge(n):
    """Batcher's odd-even merge sort for n wires (n should be power of 2)."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    if n == 4:
        return [(0, 1), (2, 3), (0, 2), (1, 3)]
    if n == 8:
        return [(0,1),(2,3),(4,5),(6,7),
                (0,2),(1,3),(4,6),(5,7),
                (0,4),(1,5),(2,6),(3,7),
                (1,2),(5,6)]
    # For larger powers of 2, use recursive construction
    half = n // 2
    left = _batcher_odd_even_merge(half)
    right = [(a + half, b + half) for a, b in left]
    # Merge phase: compare (i, i+half) for i in 0..half-1, but in odd-even pattern
    merge = []
    for phase in range(half):
        for i in range(phase % 2, half, 2):
            merge.append((i, i + half))
    # After merge, recursively sort each half
    left_sort = _batcher_odd_even_merge(half)
    right_sort = [(a + half, b + half) for a, b in left_sort]
    return left + right + merge + left_sort + right_sort

def _batcher_sort_16():
    """Batcher's odd-even merge sort for 16 wires."""
    return _batcher_odd_even_merge(16)

def _merge_two_sorted_into_sorted(a_start, a_len, b_start, b_len):
    """Merge two sorted subarrays into one sorted array.
    a_start..a_start+a_len-1 and b_start..b_start+b_len-1 are sorted.
    Returns comparators to merge them into a_start..a_start+a_len+b_len-1 sorted."""
    # For simplicity, use odd-even transposition on the combined range
    # This is not optimal but correct
    total_len = a_len + b_len
    # Interleave: place a elements and b elements alternately, then sort
    # Actually, simpler: just sort the combined range
    start = min(a_start, b_start)
    return _odd_even_transposition(total_len)  # This is placeholder, not correct

def _construct_18():
    """Construct a sorting network for 18 wires using divide and conquer."""
    # Strategy: Sort 16 wires with Batcher, then merge the remaining 2 wires
    # More carefully: interleave the 2 extra wires into the 16-wire network
    
    # Approach 1: Use 16-wire Batcher + merge 2 wires
    # The 2 extra wires need to be merged into the sorted 16
    # We can think of it as merging a sorted 16 with a sorted 2
    
    # Better approach: Use known efficient networks
    # Let's use a hybrid: sort 16 wires, then insert 2 more wires
    
    # For now, use odd-even transposition as baseline
    return _odd_even_transposition(18)

def _local_search_aggressive(network, deadline=85.0):
    """Aggressive local search: try removals and 2-for-1 replacements."""
    start_time = time.time()
    current = list(network)
    best = list(current)
    best_count = len(current)
    
    # First, verify
    if not _verify_fast(current):
        return network
    
    no_improve_streak = 0
    max_no_improve = 30
    
    while time.time() < deadline - 2.0:
        improved = False
        
        # Try removing each comparator
        for idx in range(len(current)):
            if time.time() > deadline - 2.0:
                break
            trial = current[:idx] + current[idx+1:]
            if _verify_fast(trial):
                current = trial
                best = list(current)
                best_count = len(current)
                improved = True
                no_improve_streak = 0
                break
        
        if improved:
            continue
        
        # Try 2-for-1: remove two comparators, add one different one
        if len(current) > 2:
            found = False
            # Sample some pairs to try
            num_pairs_to_try = min(50, len(current) * (len(current) - 1) // 2)
            indices = list(range(len(current)))
            random.shuffle(indices)
            tried = 0
            for i in indices:
                if tried >= num_pairs_to_try or time.time() > deadline - 2.0:
                    break
                for j in indices:
                    if i == j or tried >= num_pairs_to_try or time.time() > deadline - 2.0:
                        continue
                    if i > j:
                        continue
                    # Remove comparators at i and j
                    remaining = current[:i] + current[i+1:j] + current[j+1:] if i < j else current[:j] + current[j+1:i] + current[i+1:]
                    # Try adding a random new comparator
                    for _ in range(5):
                        a = random.randint(0, N-2)
                        b = random.randint(a+1, N-1)
                        new_comp = (a, b)
                        # Insert at random position
                        pos = random.randint(0, len(remaining))
                        trial = remaining[:pos] + [new_comp] + remaining[pos:]
                        if _verify_fast(trial):
                            current = trial
                            best = list(current)
                            best_count = len(current)
                            improved = True
                            no_improve_streak = 0
                            found = True
                            break
                    if found:
                        break
                    tried += 1
            if found:
                continue
        
        no_improve_streak += 1
        if no_improve_streak >= max_no_improve:
            break
    
    return best

def _hill_climbing_with_restarts(network, deadline=85.0):
    """Hill climbing with random restarts."""
    start_time = time.time()
    best = list(network)
    best_count = len(network)
    
    # First verify the initial network
    if not _verify_fast(best):
        return network
    
    while time.time() < deadline - 3.0:
        # Do a round of local search
        result = _local_search_aggressive(best, deadline=deadline)
        if len(result) < best_count:
            best = result
            best_count = len(result)
        
        # Check if we should restart
        if time.time() > deadline - 3.0:
            break
        
        # Occasionally restart from a perturbed version
        if random.random() < 0.1 and len(best) > 100:
            # Perturb: remove a random comparator and see if still valid
            idx = random.randint(0, len(best) - 1)
            trial = best[:idx] + best[idx+1:]
            if _verify_fast(trial):
                best = trial
                best_count = len(best)
    
    return best

# EVOLVE-BLOCK-START
def construct():
    """Construct a sorting network for 18 wires."""
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s limit
    
    # Start with odd-even transposition
    network = _odd_even_transposition(18)
    
    if time.time() > deadline:
        return network
    
    # Verify the initial network
    if not _verify_fast(network):
        return network
    
    # Aggressive local search
    improved = _hill_climbing_with_restarts(network, deadline=deadline)
    
    return improved
# EVOLVE-BLOCK-END
