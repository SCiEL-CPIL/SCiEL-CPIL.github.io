import numpy as np
import random
import time

N = 17
NUM_INPUTS = 2**N
# Precompute base inputs for speed
BASE_INPUTS = (np.arange(NUM_INPUTS)[:, None] // 2**np.arange(N-1, -1, -1)) % 2
BASE_INPUTS = BASE_INPUTS.astype(np.uint8)

def check_network(comparators):
    """
    Check if the given network sorts all 2^N boolean inputs.
    Returns True if valid, False otherwise.
    """
    if not comparators:
        return N <= 1
    
    inputs = BASE_INPUTS.copy()
    
    for i, j in comparators:
        if i > j:
            i, j = j, i
        min_val = np.minimum(inputs[:, i], inputs[:, j])
        max_val = np.maximum(inputs[:, i], inputs[:, j])
        inputs[:, i] = min_val
        inputs[:, j] = max_val
        
    diffs = np.diff(inputs, axis=1)
    return np.all(diffs >= 0)

def greedy_compress(network, time_limit):
    """
    Attempts to reduce the network size by greedily removing comparators.
    Returns the compressed network.
    """
    current = network[:]
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        for idx in range(len(current)):
            candidate = current[:idx] + current[idx+1:]
            if check_network(candidate):
                current = candidate
                improved = True
                break # Restart loop with new length
    return current

def construct():
    # Seed: Odd-even transposition
    seed_network = [(i, i + 1) for phase in range(N)
                    for i in range(phase % 2, N - 1, 2)]
    
    current_network = seed_network[:]
    best_network = current_network[:]
    
    start_time = time.time()
    timeout = 55  # Leave 5s margin
    
    # Initial compression
    current_network = greedy_compress(current_network, timeout)
    best_network = current_network[:]
    
    while time.time() < timeout:
        # Perturbation Strategy:
        # 1. Take current network
        # 2. Add random comparators (safe, preserves validity)
        # 3. Shuffle
        # 4. Compress
        # 5. If result is smaller than best, update
        
        # Make a copy to perturb
        perturbed = current_network[:]
        
        # Insert 1-3 random comparators
        num_inserts = random.randint(1, 3)
        for _ in range(num_inserts):
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, len(perturbed))
            perturbed.insert(pos, (i, j))
            
        # Shuffle by swapping a few times
        # The number of swaps can be proportional to size
        num_swaps = random.randint(5, 15)
        for _ in range(num_swaps):
            if len(perturbed) < 2:
                break
            idx1 = random.randint(0, len(perturbed) - 1)
            idx2 = random.randint(0, len(perturbed) - 1)
            if idx1 != idx2:
                perturbed[idx1], perturbed[idx2] = perturbed[idx2], perturbed[idx1]
                
        # Compress the perturbed network
        # We allocate a sub-time-limit for this compression
        # If we are close to timeout, skip or shorten
        remaining_time = timeout - (time.time() - start_time)
        if remaining_time < 1:
            break
            
        # Allocate at most 50% of remaining time to this compression, 
        # or a fixed small amount if remaining is large?
        # Let's just use the global timeout in greedy_compress, 
        # but break out if we find a better solution early?
        # greedy_compress will run until no more improvements or time limit.
        
        compressed = greedy_compress(perturbed, timeout)
        
        # Update best if improved
        if len(compressed) < len(best_network):
            best_network = compressed[:]
            current_network = compressed[:]
            
        # If we didn't improve, we might want to update current_network 
        # to the compressed version anyway? 
        # If compressed is same size as current, but different structure, 
        # it might be better for future perturbations.
        # However, if compressed is larger (shouldn't happen if we started 
        # from valid and compressed), we ignore.
        # Actually, compressed is always <= perturbed.
        # And perturbed is current + inserts.
        # So compressed can be larger than current?
        # Yes, if the inserts couldn't be removed.
        # In that case, we should NOT update current_network to a larger network.
        # We only update current_network if it leads to a better best_network.
        # Or, we can update current_network to the compressed version 
        # if it is strictly smaller than current_network?
        # If it's the same size, it's a different local minimum.
        # Let's update current_network to compressed if len(compressed) <= len(current_network)
        # This allows us to explore different local minima of the same size.
        
        if len(compressed) <= len(current_network):
            current_network = compressed[:]

    return best_network
