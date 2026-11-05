import numpy as np
import itertools
import time

N = 17

# Pre-compute inputs
# This is global to avoid re-computing in construct if called multiple times?
# The task says construct() is the entry point.
# It might be called once.
# Generating 131k inputs takes ~0.1s.

def _generate_inputs():
    # Use np.unpackbits or similar for speed?
    # np.array(list(itertools.product([0,1], repeat=N))) is fine.
    return np.array(list(itertools.product([0, 1], repeat=N)), dtype=np.int8)

_INPUTS = None

def _get_inputs():
    global _INPUTS
    if _INPUTS is None:
        _INPUTS = _generate_inputs()
    return _INPUTS

def _is_valid(network):
    inputs = _get_inputs()
    state = inputs.copy()
    for i, j in network:
        # Vectorized min/max
        state[:, i], state[:, j] = np.minimum(state[:, i], state[:, j]), np.maximum(state[:, i], state[:, j])
    # Check sorted
    # np.diff along axis 1
    return np.all(np.diff(state, axis=1) >= 0)

def _odd_even_transposition_network():
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net

def construct():
    start_time = time.time()
    TIME_LIMIT = 55.0
    
    network = _odd_even_transposition_network()
    
    # Verify seed just in case
    if not _is_valid(network):
        # Fallback to a known valid one? OET is always valid.
        return network

    improved = True
    while improved and (time.time() - start_time) < TIME_LIMIT:
        improved = False
        for idx in range(len(network)):
            # Check time inside loop
            if (time.time() - start_time) > TIME_LIMIT:
                return network
            
            # Try removing network[idx]
            candidate = network[:idx] + network[idx+1:]
            if _is_valid(candidate):
                network = candidate
                improved = True
                break # Restart scan from beginning
        
    return network
