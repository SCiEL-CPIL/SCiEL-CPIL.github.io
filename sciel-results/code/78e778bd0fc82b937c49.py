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
_INPUTS = (_indices[:, None] & _bits[None, :]) > 0
_SORTED_INPUTS = np.sort(_INPUTS, axis=1)

def is_sorted_network(comparators):
    """
    Check if the network sorts all 2^17 inputs using the zero-one principle.
    Uses precomputed input matrix with early exit on first failure.
    """
    if len(comparators) > 4096:
        return False
    if not comparators:
        return False
    
    # Simulate the network on the precomputed inputs
    state = _INPUTS.copy()
    for i, j in comparators:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    
    # Check if state matches sorted inputs - early exit on first mismatch
    if not np.all(state == _SORTED_INPUTS):
        return False
    return True

def random_network(num_comparators):
    """Generate a random sorting network candidate."""
    comparators = []
    for _ in range(num_comparators):
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        comparators.append((i, j))
    return comparators

def improve_network(comparators, max_iterations=100, deadline=None):
    """
    Try to improve a network by removing redundant comparators or swapping them.
    This is a heuristic search.
    """
    best = comparators
    best_len = len(best)
    
    current = list(best)
    
    for _ in range(max_iterations):
        if deadline and time.time() > deadline:
            break
            
        if not current:
            break
            
        # Randomly select an index to modify
        idx = random.randint(0, len(current) - 1)
        
        # Strategy 1: Remove
        candidate = current.copy()
        candidate.pop(idx)
        if is_sorted_network(candidate):
            current = candidate
            best = candidate
            best_len = len(current)
            continue
            
        # Strategy 2: Replace with a random comparator
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        candidate = current.copy()
        candidate[idx] = (i, j)
        if is_sorted_network(candidate):
            current = candidate
            best = candidate
            continue
            
        # Strategy 3: Swap with a nearby comparator
        if idx > 0:
            candidate = current.copy()
            candidate[idx], candidate[idx-1] = candidate[idx-1], candidate[idx]
            if is_sorted_network(candidate):
                current = candidate
                best = candidate
                continue
                
    return best

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin: finish well before 90s limit
    
    # Start with a known valid network (Odd-Even Transposition)
    seed_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_network.append((i, i + 1))
            
    best_network = list(seed_network)
    best_len = len(seed_network)
    
    def systematic_deletion_pass(network, time_budget):
        """Do a forward pass trying to remove each comparator."""
        current = list(network)
        i = 0
        while i < len(current):
            if time.time() > time_budget:
                break
            candidate = current.copy()
            candidate.pop(i)
            if is_sorted_network(candidate):
                current = candidate
                # Don't increment i; the next element shifted into position i
            else:
                i += 1
        return current
    
    def multi_pass_deletion(network, max_passes, time_budget):
        """Repeat deletion passes until no progress or time runs out."""
        current = list(network)
        for _ in range(max_passes):
            if time.time() > time_budget:
                break
            before = len(current)
            current = systematic_deletion_pass(current, time_budget)
            if len(current) == before:
                break
        return current
    
    # Phase 1: Aggressive systematic deletion with multiple passes
    # Use most of the time budget for this phase
    t1_deadline = deadline - 8.0
    current = multi_pass_deletion(best_network, max_passes=20, time_budget=t1_deadline)
    if len(current) < best_len:
        best_network = current
        best_len = len(current)
    
    # Phase 2: Random restarts with local improvement
    # Try several random perturbations and see if we can find smaller networks
    t2_deadline = deadline - 4.0
    for restart in range(30):
        if time.time() > t2_deadline:
            break
        
        # Start from current best and try random removals
        candidate = list(best_network)
        for _ in range(50):
            if time.time() > t2_deadline:
                break
            if not candidate:
                break
            idx = random.randint(0, len(candidate) - 1)
            new_candidate = candidate.copy()
            new_candidate.pop(idx)
            if is_sorted_network(new_candidate):
                candidate = new_candidate
        
        if len(candidate) < best_len:
            best_network = candidate
            best_len = len(candidate)
            # After finding improvement, do another systematic pass
            if time.time() < t2_deadline - 2.0:
                improved = systematic_deletion_pass(best_network, t2_deadline)
                if len(improved) < best_len:
                    best_network = improved
                    best_len = len(improved)
    
    # Phase 3: Final local search with replacements if time remains
    t3_deadline = deadline - 2.0
    if time.time() < t3_deadline and best_len > 65:
        improved = improve_network(best_network, max_iterations=200, deadline=t3_deadline)
        if len(improved) < best_len:
            best_network = improved
            best_len = len(improved)
    
    # Final Check: Ensure we return a valid network
    if not is_sorted_network(best_network):
        return seed_network
        
    return best_network
# EVOLVE-BLOCK-END
