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
    
    # Shuffle-exchange based network construction
    # For N=17, we define a structured network with layers
    # Each layer has a fixed set of potential comparator positions
    # based on a shuffle/exchange pattern
    
    def perfect_shuffle_permutation(n):
        """
        Return a perfect shuffle permutation for n wires.
        For n not a power of 2, use a generalized shuffle.
        The shuffle maps wire i to wire (2*i) % (n-1) for i < n-1,
        and wire n-1 stays at n-1.
        """
        if n <= 2:
            return list(range(n))
        perm = [0] * n
        for i in range(n - 1):
            perm[i] = (2 * i) % (n - 1)
        perm[n - 1] = n - 1
        return perm
    
    def build_shuffle_exchange_network(n, num_layers):
        """
        Build a shuffle-exchange network with the given number of layers.
        Each layer applies a perfect shuffle, then comparators on
        adjacent wire pairs (after shuffling).
        """
        comparators = []
        for layer in range(num_layers):
            # Apply perfect shuffle (implicit in the wire mapping)
            # Add comparators on adjacent pairs
            for i in range(n - 1):
                comparators.append((i, i + 1))
        return comparators
    
    def build_structured_seed(n):
        """
        Build a structured seed network using shuffle-exchange patterns.
        This creates a network with good initial structure that is
        amenable to deletion-based optimization.
        """
        comparators = []
        
        # Phase 1: Odd-even transposition (N phases)
        for phase in range(n):
            for i in range(phase % 2, n - 1, 2):
                comparators.append((i, i + 1))
        
        return comparators
    
    def build_ak_style_network(n):
        """
        Build an AKS-style network for n wires.
        AKS uses recursive splitting with perfect shuffles.
        For n=17, we use a hybrid approach.
        """
        comparators = []
        
        # Recursive construction
        def ak_sort(k, offset):
            if k <= 1:
                return
            if k == 2:
                comparators.append((offset, offset + 1))
                return
            
            # Split into halves
            half = k // 2
            ak_sort(half, offset)
            ak_sort(k - half, offset + half)
            
            # Merge with shuffle pattern
            # Apply several rounds of comparators with shuffling
            for round_num in range(k):
                # Shuffle wires within this segment
                for i in range(k - 1):
                    a = offset + i
                    b = offset + i + 1
                    # Only add if not already present to avoid duplicates
                    if (a, b) not in comparators:
                        comparators.append((a, b))
        
        ak_sort(n, 0)
        return comparators
    
    # Try multiple seed constructions and pick the best
    seeds = []
    
    # Seed 1: Odd-even transposition
    seed1 = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed1.append((i, i + 1))
    seeds.append(seed1)
    
    # Seed 2: Structured shuffle-exchange
    seed2 = build_structured_seed(N)
    seeds.append(seed2)
    
    # Seed 3: Bitonic-like for 17 wires
    # Use bitonic sort on 16 wires + merge the 17th
    def bitonic_sort_16():
        """Bitonic sort network for 16 wires."""
        comps = []
        
        def bitonic(k, offset, direction=1):
            if k == 1:
                return
            if k == 2:
                if direction == 1:
                    comps.append((offset, offset + 1))
                else:
                    comps.append((offset, offset + 1))  # Same comparator
                return
            
            half = k // 2
            bitonic(half, offset, 1)
            bitonic(half, offset + half, -1)
            
            # Merge
            for i in range(half):
                a = offset + i
                b = offset + i + half
                if direction == 1:
                    comps.append((a, b))
                else:
                    comps.append((a, b))
        
        bitonic(16, 0, 1)
        return comps
    
    seed3 = bitonic_sort_16()
    # Add comparators to merge wire 16 into the sorted 16 wires
    for i in range(15, -1, -1):
        seed3.append((i, 16))
    for i in range(16):
        seed3.append((i, i + 1))
    seeds.append(seed3)
    
    # Verify and pick the shortest valid seed
    best_seed = None
    best_seed_len = float('inf')
    for seed in seeds:
        if is_sorted_network_full(seed):
            if len(seed) < best_seed_len:
                best_seed = seed
                best_seed_len = len(seed)
    
    if best_seed is None:
        # Fallback: use the first seed that works, or generate a valid one
        best_seed = seed1
        if not is_sorted_network_full(best_seed):
            # Last resort: dense network
            best_seed = []
            for i in range(N):
                for j in range(i + 1, N):
                    best_seed.append((i, j))
    
    best_network = list(best_seed)
    best_len = len(best_network)
    
    # Phase 1: Systematic Deletion with sample validation
    current = list(best_network)
    i = 0
    while i < len(current):
        if time.time() > deadline - 25.0:
            break
        candidate = current.copy()
        candidate.pop(i)
        # Use fast sample validation first
        if is_sorted_network_sample(candidate):
            # Confirm with full validation
            if is_sorted_network_full(candidate):
                current = candidate
                best_network = candidate
                best_len = len(current)
                continue
        i += 1
    
    # Phase 2: Multiple iterative passes of systematic deletion
    for _ in range(30):
        if time.time() > deadline - 20.0:
            break
        progress = False
        i = 0
        while i < len(current):
            if time.time() > deadline - 20.0:
                break
            candidate = current.copy()
            candidate.pop(i)
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
    
    # Phase 3: Random deletion attempts
    attempts = 0
    while time.time() < deadline - 12.0 and attempts < 500:
        attempts += 1
        if not current:
            break
        idx = random.randint(0, len(current) - 1)
        candidate = current.copy()
        candidate.pop(idx)
        if is_sorted_network_sample(candidate):
            if is_sorted_network_full(candidate):
                current = candidate
                best_network = candidate
                best_len = len(current)
    
    # Phase 4: Local search - swap and replace moves
    if time.time() < deadline - 5.0:
        for _ in range(800):
            if time.time() > deadline - 5.0:
                break
            if len(current) < 2:
                break
            idx = random.randint(0, len(current) - 1)
            
            # Try swapping with a random other position
            other_idx = random.randint(0, len(current) - 1)
            if other_idx != idx:
                candidate = current.copy()
                candidate[idx], candidate[other_idx] = candidate[other_idx], candidate[idx]
                if is_sorted_network_sample(candidate):
                    if is_sorted_network_full(candidate):
                        current = candidate
                        best_network = candidate
                        best_len = len(current)
            
            # Try moving comparator to a different position
            if time.time() < deadline - 5.0:
                new_idx = random.randint(0, len(current) - 1)
                if new_idx != idx:
                    candidate = current.copy()
                    comp = candidate.pop(idx)
                    candidate.insert(new_idx, comp)
                    if is_sorted_network_sample(candidate):
                        if is_sorted_network_full(candidate):
                            current = candidate
                            best_network = candidate
                            best_len = len(current)
            
            # Try replacing with a different comparator
            if time.time() < deadline - 5.0:
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                if (i, j) != current[idx]:
                    candidate = current.copy()
                    candidate[idx] = (i, j)
                    if is_sorted_network_sample(candidate):
                        if is_sorted_network_full(candidate):
                            current = candidate
                            best_network = candidate
                            best_len = len(current)
    
    # Phase 5: Final systematic pass
    i = 0
    while i < len(current) and time.time() < deadline - 2.0:
        candidate = current.copy()
        candidate.pop(i)
        if is_sorted_network_sample(candidate):
            if is_sorted_network_full(candidate):
                current = candidate
                best_network = candidate
                best_len = len(current)
                continue
        i += 1
    
    # Final Check: Ensure we return a valid network
    if not is_sorted_network_full(best_network):
        return best_seed
        
    return best_network
# EVOLVE-BLOCK-END
