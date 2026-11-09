"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17
_DEADLINE = 85  # seconds, safety margin before 90s kill

# EVOLVE-BLOCK-START

def _generate_initial_states():
    """
    Generate the initial state matrix for all 2^N binary inputs.
    Shape: (2^N, N)
    states[input_id, wire] is the value of wire 'wire' for 'input_id'.
    """
    num_inputs = 1 << N
    # Create an array of all input IDs
    input_ids = np.arange(num_inputs, dtype=np.uint8)
    
    # For each wire k, the value is bit k of input_id
    # We can construct this column by column or use broadcasting
    # states[i, j] = (i >> j) & 1
    # Using broadcasting:
    # input_ids has shape (2^N, 1)
    # wire_indices has shape (1, N)
    wire_indices = np.arange(N, dtype=np.uint8)
    states = (input_ids[:, np.newaxis] >> wire_indices[np.newaxis, :]) & 1
    return states.astype(np.uint8)

def _verify_network(network, initial_states):
    """
    Verify if the network sorts all inputs using vectorized simulation.
    Returns True if valid.
    """
    # Make a copy to avoid modifying the original
    states = initial_states.copy()
    
    for i, j in network:
        # Compare-exchange: min to i, max to j
        # vi = states[:, i], vj = states[:, j]
        vi = states[:, i]
        vj = states[:, j]
        
        # min and max
        min_val = np.minimum(vi, vj)
        max_val = np.maximum(vi, vj)
        
        states[:, i] = min_val
        states[:, j] = max_val
    
    # Check if all rows are sorted (non-decreasing)
    # A row is sorted if states[:, k] <= states[:, k+1] for all k
    # We can check if any adjacent pair violates this
    # diff = states[:, 1:] - states[:, :-1]
    # If any diff < 0, it's not sorted.
    # Since values are 0 or 1, diff can be -1, 0, 1.
    # We just need to check if states[:, 1:] >= states[:, :-1]
    
    # Vectorized check:
    # If any element in states[:, 1:] is less than the corresponding element in states[:, :-1], it's invalid.
    # np.any(states[:, 1:] < states[:, :-1])
    
    return not np.any(states[:, 1:] < states[:, :-1])

def _seed_network():
    """Odd-even transposition sorting network for N wires."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def construct():
    start_time = time.time()
    
    # Pre-compute initial states
    initial_states = _generate_initial_states()
    
    network = _seed_network()
    best = list(network)
    best_len = len(best)
    
    # Verify seed (should be valid)
    if not _verify_network(network, initial_states):
        return best

    # Optimization loop: Greedy removal
    # We repeatedly try to remove comparators.
    # To be efficient, we can restart the scan after each removal.
    # Or we can scan once, and if we remove one, restart.
    
    improved = True
    while improved and (time.time() - start_time < _DEADLINE):
        improved = False
        # Iterate backwards to avoid index shifting issues if we remove
        # Or just restart the loop
        for idx in range(len(network) - 1, -1, -1):
            if time.time() - start_time > _DEADLINE:
                return best
            
            # Try removing network[idx]
            candidate = network[:idx] + network[idx+1:]
            
            # Verify candidate
            if _verify_network(candidate, initial_states):
                network = candidate
                best = list(network)
                best_len = len(best)
                improved = True
                # Restart the inner loop to check if more can be removed
                # from the new, shorter network
                break
        
        # If no improvement in a full pass, we might be stuck in a local minimum.
        # We could try swapping, but removal is the primary goal for size.
        # Given time constraints, simple greedy removal is the most robust structural change
        # from the previous slow Python-loop verification.

    return best

# EVOLVE-BLOCK-END
