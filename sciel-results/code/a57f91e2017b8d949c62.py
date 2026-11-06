"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17
SEED_COMPARATORS = 136  # Number of comparators in the seed odd-even transposition network

# Precompute masks for fast simulation
# A comparator (i, j) sets wire i to min(x_i, x_j) and wire j to max(x_i, x_j)
# For 17 wires, we can represent state as a 17-bit integer.
# However, simulating each comparator step-by-step on 131k inputs is slow in pure Python.
# We will use NumPy vectorization over all inputs at once for speed.

# Pre-compute the indices for each wire in a flattened array
# Inputs: shape (131072, 17)
# Comparators: list of (i, j)

def simulate_network(comparators, inputs):
    """
    inputs: numpy array of shape (num_inputs, N) with values in {0, 1}
    comparators: list of (i, j)
    returns: sorted numpy array of shape (num_inputs, N)
    """
    state = inputs.copy()
    for i, j in comparators:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    return state

# Precompute the input matrix and expected sorted outputs once at module load time
_NUM_INPUTS = 1 << N
_indices = np.arange(_NUM_INPUTS, dtype=np.uint32)
_bits = 1 << np.arange(N, dtype=np.uint32)
_INPUTS = ((_indices[:, None] & _bits[None, :]) > 0).astype(np.uint8)
_SORTED_INPUTS = np.sort(_INPUTS, axis=1)

# Precompute a smaller sample set for fast validation during search
_SAMPLE_SIZE = 4096
_rng = np.random.RandomState(42)
_SAMPLE_INDICES = _rng.choice(_NUM_INPUTS, size=_SAMPLE_SIZE, replace=False)
_SAMPLE_INPUTS = _INPUTS[_SAMPLE_INDICES]
_SAMPLE_SORTED = _SORTED_INPUTS[_SAMPLE_INDICES]

def is_sorted_network_full(comparators):
    """
    Check if the network sorts all 2^17 inputs using the zero-one principle.
    Uses precomputed input matrix for speed.
    """
    if len(comparators) > 4096:
        return False
    
    # Simulate the network on the precomputed inputs
    state = _INPUTS.copy()
    for i, j in comparators:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    
    # Check if state matches sorted inputs
    return bool(np.array_equal(state, _SORTED_INPUTS))

def is_sorted_network_sample(comparators):
    """
    Fast check on a sample of inputs. Not sufficient for final validation,
    but much faster for guiding search.
    """
    if len(comparators) > 4096:
        return False
    
    state = _SAMPLE_INPUTS.copy()
    for i, j in comparators:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    
    return bool(np.array_equal(state, _SAMPLE_SORTED))

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
    deadline = start_time + 85.0  # Safety margin: finish well before 90s limit
    
    # Proper odd-even merge sort network construction
    def oe_merge_sort(n):
        """Proper odd-even merge sort network for n elements."""
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        half = n // 2
        left = oe_merge_sort(half)
        right = oe_merge_sort(n - half)
        right = [(i + half, j + half) for i, j in right]
        
        # Odd-even merge of two sorted sequences
        merge = []
        for k in range(1, n):
            if k % 2 == 1:
                for pos in range(0, n - 1, 2):
                    merge.append((pos, pos + 1))
            else:
                for pos in range(1, n - 1, 2):
                    merge.append((pos, pos + 1))
        
        return left + right + merge
    
    seed_network = oe_merge_sort(N)
    # Verify seed is valid
    if not is_sorted_network_full(seed_network):
        # Fallback: try a different construction
        # Odd-even transposition sort
        seed_network = []
        for phase in range(N):
            for i in range(phase % 2, N - 1, 2):
                seed_network.append((i, i + 1))
    
    best_network = list(seed_network)
    best_len = len(best_network)
    current = list(best_network)
    
    # Fast rejection check: simulate only first half of comparators on sample
    def fast_reject(candidate):
        """Quick check: if after first half of comparators the sample is already wrong, reject."""
        if len(candidate) > 4096:
            return False
        state = _SAMPLE_INPUTS.copy()
        mid = len(candidate) // 2
        for i, j in candidate[:mid]:
            min_val = np.minimum(state[:, i], state[:, j])
            max_val = np.maximum(state[:, i], state[:, j])
            state[:, i] = min_val
            state[:, j] = max_val
        # Check partial ordering: for each row, first half should be <= second half
        if mid < N:
            # Not enough comparators to guarantee anything, just return True
            return True
        # Check that the state is "progressing" toward sorted
        # For a correct network, after mid comparators, the distribution should be reasonable
        # Simple heuristic: check that no wire has all 1s while another has all 0s in wrong order
        col_sums = state.sum(axis=0)
        # In a sorting network, col_sums should be non-decreasing
        if not np.all(np.diff(col_sums) >= -1):  # allow small violations
            return False
        return True
    
    # Importance scoring: for each comparator, measure how often it changes the state
    def compute_importance(network, num_samples=1000):
        """Compute importance score for each comparator based on how often it changes state."""
        if not network:
            return []
        
        # Use a subset of inputs for speed
        sample_idx = np.random.RandomState(123).choice(_NUM_INPUTS, size=min(num_samples, _NUM_INPUTS), replace=False)
        sample_inputs = _INPUTS[sample_idx]
        
        state = sample_inputs.copy()
        importance = [0.0] * len(network)
        
        for idx, (i, j) in enumerate(network):
            min_val = np.minimum(state[:, i], state[:, j])
            max_val = np.maximum(state[:, i], state[:, j])
            # Count how many inputs this comparator changes
            changed = np.sum(state[:, i] != min_val)
            importance[idx] = changed
            state[:, i] = min_val
            state[:, j] = max_val
        
        return importance
    
    # Phase 1: Systematic Deletion with fast rejection
    i = 0
    while i < len(current) and time.time() < deadline - 30.0:
        candidate = current.copy()
        candidate.pop(i)
        # Fast rejection first
        if fast_reject(candidate):
            # Use fast sample validation
            if is_sorted_network_sample(candidate):
                # Confirm with full validation
                if is_sorted_network_full(candidate):
                    current = candidate
                    best_network = candidate
                    best_len = len(current)
                    continue
        i += 1
    
    # Phase 2: Importance-guided deletion - try removing least important comparators first
    if time.time() < deadline - 25.0:
        importance = compute_importance(current)
        # Sort indices by importance (ascending - try removing low importance first)
        sorted_indices = sorted(range(len(current)), key=lambda x: importance[x])
        
        for idx in sorted_indices:
            if time.time() > deadline - 25.0:
                break
            if idx >= len(current):
                continue
            candidate = current.copy()
            candidate.pop(idx)
            if fast_reject(candidate):
                if is_sorted_network_sample(candidate):
                    if is_sorted_network_full(candidate):
                        current = candidate
                        best_network = candidate
                        best_len = len(current)
                        # Recompute importance after deletion
                        importance = compute_importance(current)
                        sorted_indices = sorted(range(len(current)), key=lambda x: importance[x])
    
    # Phase 3: Multiple iterative passes of systematic deletion
    for _ in range(20):
        if time.time() > deadline - 20.0:
            break
        progress = False
        i = 0
        while i < len(current):
            if time.time() > deadline - 20.0:
                break
            candidate = current.copy()
            candidate.pop(i)
            if fast_reject(candidate):
                if is_sorted_network_sample(candidate):
                    if is_sorted_network_full(candidate):
                        current = candidate
                        best_network = candidate
                        best_len = len(current)
                        progress = True
                        continue
            i += 1
        if not progress:
            break
    
    # Phase 4: Random deletion attempts
    attempts = 0
    while time.time() < deadline - 15.0 and attempts < 300:
        attempts += 1
        if not current:
            break
        idx = random.randint(0, len(current) - 1)
        candidate = current.copy()
        candidate.pop(idx)
        if fast_reject(candidate):
            if is_sorted_network_sample(candidate):
                if is_sorted_network_full(candidate):
                    current = candidate
                    best_network = candidate
                    best_len = len(current)
    
    # Phase 5: Local search - swap and replace moves
    if time.time() < deadline - 8.0:
        for _ in range(500):
            if time.time() > deadline - 8.0:
                break
            if len(current) < 2:
                break
            idx = random.randint(0, len(current) - 1)
            
            # Try swapping with a random other position
            other_idx = random.randint(0, len(current) - 1)
            if other_idx != idx:
                candidate = current.copy()
                candidate[idx], candidate[other_idx] = candidate[other_idx], candidate[idx]
                if fast_reject(candidate):
                    if is_sorted_network_sample(candidate):
                        if is_sorted_network_full(candidate):
                            current = candidate
                            best_network = candidate
                            best_len = len(current)
            
            # Try moving comparator to a different position
            if time.time() < deadline - 8.0:
                new_idx = random.randint(0, len(current) - 1)
                if new_idx != idx:
                    candidate = current.copy()
                    comp = candidate.pop(idx)
                    candidate.insert(new_idx, comp)
                    if fast_reject(candidate):
                        if is_sorted_network_sample(candidate):
                            if is_sorted_network_full(candidate):
                                current = candidate
                                best_network = candidate
                                best_len = len(current)
            
            # Try replacing with a different comparator
            if time.time() < deadline - 8.0:
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                if (i, j) != current[idx]:
                    candidate = current.copy()
                    candidate[idx] = (i, j)
                    if fast_reject(candidate):
                        if is_sorted_network_sample(candidate):
                            if is_sorted_network_full(candidate):
                                current = candidate
                                best_network = candidate
                                best_len = len(current)
    
    # Phase 6: Final systematic pass
    i = 0
    while i < len(current) and time.time() < deadline - 3.0:
        candidate = current.copy()
        candidate.pop(i)
        if fast_reject(candidate):
            if is_sorted_network_sample(candidate):
                if is_sorted_network_full(candidate):
                    current = candidate
                    best_network = candidate
                    best_len = len(current)
                    continue
        i += 1
    
    # Final Check: Ensure we return a valid network
    if not is_sorted_network_full(best_network):
        return seed_network
        
    return best_network
# EVOLVE-BLOCK-END
