"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17
SEED_COMPARATORS = 136  # Number of comparators in the seed odd-even transposition network

# Precompute the input matrix and expected sorted outputs once at module load time
_NUM_INPUTS = 1 << N
_indices = np.arange(_NUM_INPUTS, dtype=np.uint32)
_bits = 1 << np.arange(N, dtype=np.uint32)
_INPUTS = (_indices[:, None] & _bits[None, :]) > 0
_SORTED_INPUTS = np.sort(_INPUTS, axis=1)

# Precompute a smaller sample set for fast validation during search
_SAMPLE_SIZE = 4096
_rng = np.random.RandomState(42)
_SAMPLE_INDICES = _rng.choice(_NUM_INPUTS, size=_SAMPLE_SIZE, replace=False)
_SAMPLE_INPUTS = _INPUTS[_SAMPLE_INDICES]
_SAMPLE_SORTED = _SORTED_INPUTS[_SAMPLE_INDICES]

def _simulate_on_comparators(comparators, input_matrix):
    """Simulate network on given input matrix using vectorized operations."""
    state = input_matrix.copy()
    for i, j in comparators:
        a = state[:, i]
        b = state[:, j]
        # min goes to i, max goes to j
        mask = a > b
        state[:, i] = np.where(mask, b, a)
        state[:, j] = np.where(mask, a, b)
    return state

def is_sorted_network_full(comparators):
    """
    Check if the network sorts all 2^17 inputs using the zero-one principle.
    Uses precomputed input matrix for speed.
    """
    if len(comparators) > 4096:
        return False
    
    state = _simulate_on_comparators(comparators, _INPUTS)
    return np.array_equal(state, _SORTED_INPUTS)

def is_sorted_network_sample(comparators):
    """
    Fast check on a sample of inputs. Not sufficient for final validation,
    but much faster for guiding search.
    """
    if len(comparators) > 4096:
        return False
    
    state = _simulate_on_comparators(comparators, _SAMPLE_INPUTS)
    return np.array_equal(state, _SAMPLE_SORTED)

# Alias for backward compatibility
def is_sorted_network(comparators):
    return is_sorted_network_full(comparators)

def random_network(num_comparators):
    """Generate a random sorting network candidate."""
    comparators = []
    for _ in range(num_comparators):
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        comparators.append((i, j))
    return comparators

def try_remove_comparator(current, idx):
    """Try removing comparator at index idx. Returns (success, new_network)."""
    candidate = current.copy()
    candidate.pop(idx)
    if is_sorted_network_sample(candidate):
        if is_sorted_network_full(candidate):
            return True, candidate
    return False, current

def odd_even_merge_sort_network(n):
    """
    Generate an odd-even merge sort network for n wires.
    Returns list of (i, j) comparators.
    """
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    # Split into two halves
    half = n // 2
    left = odd_even_merge_sort_network(half)
    right = odd_even_merge_sort_network(n - half)
    
    # Offset right half comparators
    right = [(i + half, j + half) for i, j in right]
    
    # Merge the two halves
    merged = left + right
    # Odd-even merge: interleave comparators between the two halves
    for k in range(1, n, 2):
        if k < n:
            merged.append((k - 1, k))
    
    return merged

def odd_even_merge_sort_network_v2(n):
    """
    Better odd-even merge sort network construction.
    """
    comparators = []
    
    def build(n):
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        half = n // 2
        left = build(half)
        right = build(n - half)
        right = [(i + half, j + half) for i, j in right]
        
        # Merge phase: odd-even merge
        merge_comps = []
        # We need to merge two sorted sequences of length half and n-half
        # Using the odd-even merge procedure
        for phase in range(n - 1):
            start = phase % 2
            for i in range(start, n - 1, 2):
                # Compare wire i and i+1, but only if they span the boundary
                # Actually, for odd-even merge, we compare specific pairs
                pass
        
        # Simpler: just use the standard merge network
        # For merging two sorted lists of sizes a and b, we need a+b-1 comparators
        # arranged in a specific pattern
        a = half
        b = n - half
        
        # The merge network for two sorted inputs of size a and b
        # uses a+b-1 comparators in a specific arrangement
        merge = []
        for i in range(a + b - 1):
            # This is a simplified version; real odd-even merge is more complex
            # Let's just use a simple approach
            pass
        
        return left + right + merge
    
    return build(n)

# Use a known-good small network construction
def bitonic_sort_network(n):
    """
    Generate a bitonic sort network for n wires.
    n must be a power of 2 for the standard construction.
    For n=17, we'll pad to 32 or use a hybrid approach.
    Here we return the network for the largest power of 2 <= n,
    plus additional comparators for the remaining wires.
    """
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    # For non-power-of-2, we'll use a different approach
    # Let's just build it recursively with the bitonic pattern
    comparators = []
    
    def bitonic(k, direction=1):
        """Generate bitonic merge network for k wires."""
        if k == 1:
            return []
        if k == 2:
            if direction == 1:
                return [(0, 1)]
            else:
                return [(1, 0)]  # swap
        
        half = k // 2
        # Recursively sort halves in opposite directions
        left = bitonic(half, 1)
        right = bitonic(k - half, -1)
        right = [(i + half, j + half) for i, j in right]
        
        # Merge: compare pairs (i, i+half)
        merge = []
        for i in range(half):
            j = i + half
            if direction == 1:
                merge.append((i, j))
            else:
                merge.append((j, i))
        
        return left + right + merge
    
    # For n=17, we can't use pure bitonic (needs power of 2)
    # Instead, use a layered approach
    return []

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin: finish well before 90s limit
    
    # Start with a known valid network using odd-even merge sort
    def oe_merge_sort(n):
        """Odd-even merge sort network for n elements."""
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        half = n // 2
        left = oe_merge_sort(half)
        right = oe_merge_sort(n - half)
        right = [(i + half, j + half) for i, j in right]
        
        # Odd-even merge of two sorted halves
        merge = []
        for phase in range(n - 1):
            offset = phase % 2
            for i in range(offset, n - 1, 2):
                merge.append((i, i + 1))
        
        return left + right + merge
    
    seed_network = oe_merge_sort(N)
    # Verify seed is valid
    if not is_sorted_network_full(seed_network):
        # Fallback to odd-even transposition
        seed_network = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                seed_network.append((i, i + 1))
    
    best_network = list(seed_network)
    best_len = len(best_network)
    current = list(best_network)
    
    # Helper: try to remove comparator at index idx
    def try_remove(idx):
        nonlocal current, best_network, best_len
        candidate = current.copy()
        candidate.pop(idx)
        if is_sorted_network_sample(candidate):
            if is_sorted_network_full(candidate):
                current = candidate
                best_network = candidate
                best_len = len(current)
                return True
        return False
    
    # Phase 1: Aggressive systematic deletion - multiple passes
    for pass_num in range(50):
        if time.time() > deadline - 10.0:
            break
        progress = False
        i = 0
        while i < len(current):
            if time.time() > deadline - 10.0:
                break
            if try_remove(i):
                progress = True
                # Don't increment i since list shifted
            else:
                i += 1
        if not progress:
            break
    
    # Phase 2: Random deletion with restarts
    rng_search = random.Random(12345)
    for restart in range(10):
        if time.time() > deadline - 6.0:
            break
        attempts = 0
        while time.time() < deadline - 6.0 and attempts < 500:
            attempts += 1
            if not current:
                break
            idx = rng_search.randint(0, len(current) - 1)
            if try_remove(idx):
                pass
            else:
                # Occasionally try inserting a new comparator then removing others
                if attempts % 50 == 0 and len(current) > 60:
                    # Try to find a useful insertion
                    i = rng_search.randint(0, N - 2)
                    j = rng_search.randint(i + 1, N - 1)
                    candidate = current.copy()
                    candidate.append((i, j))
                    if is_sorted_network_sample(candidate):
                        if is_sorted_network_full(candidate):
                            # Now try to remove some comparators
                            for k in range(len(candidate)):
                                if time.time() > deadline - 6.0:
                                    break
                                if try_remove(k):
                                    break
    
    # Phase 3: Local search - try moving comparators to different positions
    if time.time() < deadline - 4.0:
        for _ in range(500):
            if time.time() > deadline - 4.0:
                break
            if len(current) < 2:
                break
            idx = rng_search.randint(0, len(current) - 1)
            new_idx = rng_search.randint(0, len(current) - 1)
            if idx == new_idx:
                continue
            
            candidate = current.copy()
            comp = candidate.pop(idx)
            candidate.insert(new_idx, comp)
            
            if is_sorted_network_sample(candidate):
                if is_sorted_network_full(candidate):
                    current = candidate
                    best_network = candidate
                    best_len = len(current)
    
    # Phase 4: Try replacing comparators with different ones
    if time.time() < deadline - 3.0:
        for _ in range(300):
            if time.time() > deadline - 3.0:
                break
            if not current:
                break
            idx = rng_search.randint(0, len(current) - 1)
            i = rng_search.randint(0, N - 2)
            j = rng_search.randint(i + 1, N - 1)
            if (i, j) == current[idx]:
                continue
            
            candidate = current.copy()
            candidate[idx] = (i, j)
            if is_sorted_network_sample(candidate):
                if is_sorted_network_full(candidate):
                    current = candidate
                    best_network = candidate
                    best_len = len(current)
    
    # Phase 5: Final aggressive deletion pass
    i = 0
    while i < len(current) and time.time() < deadline - 1.0:
        if try_remove(i):
            pass
        else:
            i += 1
    
    # Final Check: Ensure we return a valid network
    if not is_sorted_network_full(best_network):
        return seed_network
        
    return best_network
# EVOLVE-BLOCK-END
