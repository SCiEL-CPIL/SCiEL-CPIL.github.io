import numpy as np
import time

N = 16
TIME_LIMIT = 90.0
SAFETY_MARGIN = 15.0

def check_validity(network):
    if not network:
        return False
        
    # Generate all 2^16 inputs
    # Use vectorized operations
    # inputs: 0..65535
    inputs = np.arange(2**N, dtype=np.uint16)
    
    # Unpack to bits
    # bits shape: (65536, 16)
    # We can use bit shifting
    bits = np.zeros((2**N, N), dtype=np.uint8)
    for k in range(N):
        bits[:, k] = (inputs >> k) & 1
        
    # Apply comparators
    for i, j in network:
        min_val = np.minimum(bits[:, i], bits[:, j])
        max_val = np.maximum(bits[:, i], bits[:, j])
        bits[:, i] = min_val
        bits[:, j] = max_val
        
    # Check sorted
    # inv = (bits[:, :-1] == 1) & (bits[:, 1:] == 0)
    # if np.any(inv): return False
    diff = np.diff(bits.astype(np.int8), axis=1)
    if np.any(diff < 0):
        return False
        
    return True

def construct():
    # Initial network: Odd-Even Transposition
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    start_time = time.time()
    best_network = network
    
    # Greedy Removal Search
    # We perform multiple passes. In each pass, we try to remove each comparator.
    # If a removal is valid, we remove it and restart the pass.
    
    while time.time() - start_time < TIME_LIMIT - SAFETY_MARGIN:
        improved = False
        for idx in range(len(network)):
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            # Try removing network[idx]
            candidate = network[:idx] + network[idx+1:]
            
            # Optimization: If the network is small, check quickly.
            # Also, we can cache validity? No, network changes.
            
            if check_validity(candidate):
                network = candidate
                best_network = network
                improved = True
                break # Restart the for loop
        
        if not improved:
            # No single removal works. 
            # We could try swaps, but for simplicity and safety, we stop.
            # Or we can try a few random swaps to unlock removals.
            # Let's add a simple swap search if we have time.
            break
    
    return best_network
