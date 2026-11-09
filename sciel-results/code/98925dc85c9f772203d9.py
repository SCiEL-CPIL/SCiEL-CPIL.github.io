import numpy as np
import random
import time

N = 17
_NUM_INPUTS = 2 ** N

# Precompute the input matrix once
def _make_input_matrix():
    return (np.arange(_NUM_INPUTS)[:, None] // 2**np.arange(N-1, -1, -1)) % 2

_INPUT_MATRIX = _make_input_matrix()

def check_network(comparators):
    """
    Check if the given network sorts all 2^N boolean inputs.
    Returns True if valid, False otherwise.
    """
    inputs = _INPUT_MATRIX.copy()
    
    for i, j in comparators:
        if i > j:
            i, j = j, i
        min_val = np.minimum(inputs[:, i], inputs[:, j])
        max_val = np.maximum(inputs[:, i], inputs[:, j])
        inputs[:, i] = min_val
        inputs[:, j] = max_val
        
    # Check if sorted: each row must be non-decreasing
    diffs = np.diff(inputs, axis=1)
    return bool(np.all(diffs >= 0))

def construct():
    # Seed: Odd-even transposition
    seed_network = [(i, i + 1) for phase in range(N)
                    for i in range(phase % 2, N - 1, 2)]
    
    current_network = seed_network[:]
    best_network = current_network[:]
    
    start_time = time.time()
    timeout = 55  # Leave 5s margin
    
    # Hill Climbing: try to remove comparators
    # Try multiple removals per iteration for efficiency
    
    no_improve_count = 0
    
    while time.time() - start_time < timeout:
        improved = False
        n_comps = len(current_network)
        
        if n_comps == 0:
            break
        
        # Try multiple random removals in this iteration
        attempts = min(5, n_comps)
        for _ in range(attempts):
            if len(current_network) == 0:
                break
            idx = random.randint(0, len(current_network) - 1)
            candidate = current_network[:idx] + current_network[idx+1:]
            
            if check_network(candidate):
                current_network = candidate
                best_network = candidate[:]
                improved = True
                no_improve_count = 0
                break
        
        if not improved:
            no_improve_count += 1
            
            # If stuck, try more aggressive strategies
            if no_improve_count > 10 and len(current_network) >= 2:
                strategy = random.random()
                
                if strategy < 0.5:
                    # Try a random swap
                    idx1 = random.randint(0, len(current_network) - 1)
                    idx2 = random.randint(0, len(current_network) - 1)
                    if idx1 != idx2:
                        candidate = current_network[:]
                        candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
                        if check_network(candidate):
                            current_network = candidate
                            best_network = candidate[:]
                            improved = True
                            no_improve_count = 0
                else:
                    # Try a move: remove from one position, insert at another
                    idx1 = random.randint(0, len(current_network) - 1)
                    idx2 = random.randint(0, len(current_network) - 1)
                    if idx1 != idx2:
                        comp = current_network[idx1]
                        candidate = current_network[:idx1] + current_network[idx1+1:]
                        # Adjust idx2 if needed
                        if idx2 > idx1:
                            idx2 -= 1
                        idx2 = min(idx2, len(candidate))
                        candidate.insert(idx2, comp)
                        if check_network(candidate):
                            current_network = candidate
                            best_network = candidate[:]
                            improved = True
                            no_improve_count = 0

    return best_network
