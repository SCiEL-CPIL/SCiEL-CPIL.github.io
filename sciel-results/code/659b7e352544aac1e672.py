"""Sorting network synthesis for 18 wires with time-bounded local search."""
import numpy as np
import time
import random

N = 18
DEADLINE = 85.0  # safety margin before 90s kill

def _apply_comparator(state, i, j):
    """Apply compare-exchange to a single state (numpy array)."""
    if state[i] > state[j]:
        state[i], state[j] = state[j], state[i]
    return state

def _verify_network(comparators):
    """
    Verify a sorting network against all 2^N Boolean inputs using vectorized NumPy.
    Returns True if the network sorts all Boolean inputs.
    """
    if len(comparators) == 0:
        return True
    
    # All 2^N Boolean inputs as a matrix: shape (2^N, N)
    n_inputs = 1 << N
    # Generate all bit patterns
    bits = np.arange(n_inputs, dtype=np.uint8)
    # Create the matrix: each row is one input pattern
    # bits[i] in binary gives the i-th input
    input_matrix = np.zeros((n_inputs, N), dtype=np.uint8)
    for k in range(N):
        input_matrix[:, k] = (bits >> k) & 1
    
    # Apply each comparator
    for (i, j) in comparators:
        # For each input, if input[i] > input[j], swap
        # input[i] > input[j] means input[i]=1 and input[j]=0
        swap_mask = (input_matrix[:, i] == 1) & (input_matrix[:, j] == 0)
        # Apply swap where needed
        input_matrix[swap_mask, i] = 0
        input_matrix[swap_mask, j] = 1
    
    # Check if sorted: each row should be non-decreasing
    # sorted means for all k < N-1: row[k] <= row[k+1]
    for k in range(N - 1):
        if np.any(input_matrix[:, k] > input_matrix[:, k + 1]):
            return False
    return True

def _verify_network_fast(comparators, timeout=None):
    """
    Faster verification using the 0-1 principle with a more efficient approach.
    We can verify by checking that the network sorts all 2^N patterns.
    Use a more memory-efficient approach for large N.
    """
    if len(comparators) == 0:
        return True
    
    n_inputs = 1 << N
    # Use uint8 for memory efficiency
    state = np.zeros((n_inputs, N), dtype=np.uint8)
    for k in range(N):
        state[:, k] = (np.arange(n_inputs, dtype=np.uint8) >> k) & 1
    
    for (i, j) in comparators:
        swap_mask = (state[:, i] == 1) & (state[:, j] == 0)
        if np.any(swap_mask):
            state[swap_mask, i] = 0
            state[swap_mask, j] = 1
    
    # Check sorted
    diffs = np.diff(state, axis=1)
    return not np.any(diffs < 0)

def _count_inversions(state):
    """Count inversions in a state (useful for heuristic)."""
    count = 0
    for i in range(N):
        for j in range(i + 1, N):
            if state[i] > state[j]:
                count += 1
    return count

def _network_depth(comparators):
    """Compute the parallel depth of the network."""
    wire_last_layer = [-1] * N
    depth = 0
    for (i, j) in comparators:
        new_layer = max(wire_last_layer[i], wire_last_layer[j]) + 1
        if new_layer > depth:
            depth = new_layer
        wire_last_layer[i] = new_layer
        wire_last_layer[j] = new_layer
    return depth

def _remove_redundant(comparators, verify_func):
    """
    Try to remove redundant comparators one at a time.
    Returns the reduced list and whether any were removed.
    """
    removed = False
    changed = True
    while changed:
        changed = False
        for idx in range(len(comparators)):
            candidate = comparators[:idx] + comparators[idx+1:]
            if verify_func(candidate):
                comparators = candidate
                removed = True
                changed = True
                break
    return comparators, removed

def _local_search(comparators, verify_func, deadline):
    """
    Local search: try removing, swapping, and adding comparators.
    Returns the best network found.
    """
    best = list(comparators)
    best_len = len(best)
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try removing each comparator
        for idx in range(len(best)):
            if time.time() >= deadline:
                break
            candidate = best[:idx] + best[idx+1:]
            if verify_func(candidate):
                best = candidate
                best_len = len(best)
                improved = True
                break
        
        if improved:
            continue
        
        # 2. Try swapping adjacent comparators
        for idx in range(len(best) - 1):
            if time.time() >= deadline:
                break
            candidate = best[:]
            candidate[idx], candidate[idx+1] = best[idx+1], best[idx]
            if verify_func(candidate):
                # Swapping doesn't change count, but might enable further removals
                # Only accept if it leads to a different (potentially better) structure
                # Actually, same count - skip unless we want diversity
                pass
        
        # 3. Try adding a comparator then removing two (net -1)
        # This is more complex; try random additions
        if len(best) < 200:  # don't add too many
            for _ in range(50):
                if time.time() >= deadline:
                    break
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                # Find a good position to insert
                pos = random.randint(0, len(best))
                candidate = best[:pos] + [(i, j)] + best[pos:]
                # Now try to remove two comparators
                for idx1 in range(len(candidate)):
                    if time.time() >= deadline:
                        break
                    c1 = candidate[:idx1] + candidate[idx1+1:]
                    for idx2 in range(len(c1)):
                        if time.time() >= deadline:
                            break
                        c2 = c1[:idx2] + c1[idx2+1:]
                        if len(c2) < best_len and verify_func(c2):
                            best = c2
                            best_len = len(best)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            # Stuck - try a random perturbation
            if len(best) > 10:
                # Remove a random comparator and try to repair
                idx = random.randint(0, len(best) - 1)
                perturbed = best[:idx] + best[idx+1:]
                if verify_func(perturbed):
                    best = perturbed
                    best_len = len(best)
                else:
                    # Try adding back a different comparator
                    i = random.randint(0, N - 2)
                    j = random.randint(i + 1, N - 1)
                    pos = random.randint(0, len(perturbed))
                    candidate = perturbed[:pos] + [(i, j)] + perturbed[pos]
                    if verify_func(candidate):
                        best = candidate
                        best_len = len(best)
            else:
                # Small network, just break
                break
    
    return best

def construct():
    """
    Construct a sorting network for 18 wires.
    Returns a list of (i, j) comparator pairs.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Start with odd-even transposition network
    comparators = [(i, i + 1) for phase in range(N)
                   for i in range(phase % 2, N - 1, 2)]
    
    # Verify the seed
    if not _verify_network_fast(comparators):
        # Should not happen, but fallback
        return comparators
    
    # Phase 1: Remove redundant comparators
    comparators, _ = _remove_redundant(comparators, _verify_network_fast)
    
    if time.time() >= deadline - 5:
        return comparators
    
    # Phase 2: Local search
    comparators = _local_search(comparators, _verify_network_fast, deadline - 2)
    
    # Phase 3: Final cleanup - remove any remaining redundancies
    comparators, _ = _remove_redundant(comparators, _verify_network_fast)
    
    # Final verification
    if not _verify_network_fast(comparators):
        # Fallback to seed
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    return comparators
