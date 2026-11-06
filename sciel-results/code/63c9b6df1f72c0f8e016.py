"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 18
TIME_LIMIT = 80  # safety margin for evaluator 90s limit

def _odd_even_transposition():
    """Generate the seed odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Start with the known good seed network
    best_network = _odd_even_transposition()
    best_len = len(best_network)
    
    # Pre-compute verification inputs
    all_inputs = np.arange(2**N, dtype=np.uint32)
    bits_matrix = np.zeros((2**N, N), dtype=np.uint8)
    for k in range(N):
        bits_matrix[:, k] = (all_inputs >> k) & 1
    
    # Pre-compute sorted check: a row is sorted if for all k < N-1, current[k] <= current[k+1]
    # We can just check if current[:, :-1] <= current[:, 1:] holds for all elements
    
    def _fast_verify(comparators):
        current = bits_matrix.copy()
        for i, j in comparators:
            mask = current[:, i] > current[:, j]
            if np.any(mask):
                current[mask, i], current[mask, j] = current[mask, j], current[mask, i]
        # Check sorted
        if np.any(current[:, :-1] > current[:, 1:]):
            return False
        return True

    # Verify seed just in case
    if not _fast_verify(best_network):
        return best_network # Should not happen, but safe fallback

    current_network = list(best_network)
    current_len = best_len

    # Local search: try removing comparators
    # Strategy: Repeatedly scan from end to start, try removing each comparator.
    # If removal keeps it valid, keep the removal and restart scan from end.
    # This is a greedy "remove if possible" heuristic.
    
    # Optimization: 
    # 1. Vectorized verification is the bottleneck.
    # 2. We can potentially cache intermediate states? No, too much memory.
    # 3. We can try to be smarter about which comparator to remove.
    #    Usually, redundant comparators are later in the network or in specific patterns.
    #    However, simple backward scan is robust.
    
    # To speed up, we can try to limit the search space or use a better heuristic.
    # Let's stick to the backward scan but ensure we don't exceed time.
    
    improved = True
    while improved and (time.time() < deadline - 5):
        improved = False
        # Iterate backwards to avoid index issues when removing
        for idx in range(len(current_network) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            
            # Try removing comparator at idx
            trial = current_network[:idx] + current_network[idx+1:]
            
            # Optimization: If we have removed many comparators, the network is smaller.
            # Verification cost is proportional to number of comparators.
            
            if _fast_verify(trial):
                current_network = trial
                current_len = len(current_network)
                improved = True
                # Restart scan from end to find more redundancies introduced by removal?
                # Or just continue? Restarting is safer to ensure we don't miss
                # a comparator that became redundant due to a previous removal.
                break
        
        # If we didn't break (no improvement in this pass), we are done.
        # But wait, the 'improved' flag is set to True inside the loop.
        # If we break due to time, we exit the while loop.
        
        # If we completed the inner for loop without finding a removal, improved is False, and we exit while.
    
    if current_len < best_len:
        best_network = current_network
        best_len = current_len
    
    return best_network
# EVOLVE-BLOCK-END
