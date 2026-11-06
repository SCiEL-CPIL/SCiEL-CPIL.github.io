import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 17
    min_dist = 6
    max_size = 340
    
    # Precompute popcounts using NumPy for fast indexing
    # 2**17 = 131072, this array is small and fast to create
    popcount_np = np.array([bin(i).count('1') for i in range(1 << n)], dtype=np.uint8)
    
    def greedy_search(initial_words):
        """
        Runs a NumPy-accelerated greedy search to expand the set of codewords.
        """
        C = np.array(initial_words, dtype=np.uint32)
        C_list = list(initial_words)
        
        # We will try to add words until no more can be added or time runs out
        # To avoid getting stuck in a local maximum, we can use a "best improvement" strategy
        # or simply random sampling with a "max min-distance" heuristic.
        
        while time.time() < deadline - 2.0:
            # Generate a batch of random candidates
            # We use uint32 to ensure fast XOR
            batch_size = 2000
            batch = np.random.randint(0, 1 << n, size=batch_size, dtype=np.uint32)
            
            # Compute distances from each candidate in batch to all words in C
            # Shape: (batch_size, len(C))
            # XOR operation
            xor_matrix = batch[:, None] ^ C[None, :]
            
            # Get popcounts
            # popcount_np is indexed by the XOR values
            dists = popcount_np[xor_matrix]
            
            # Find minimum distance for each candidate
            min_dists = dists.min(axis=1)
            
            # Find valid candidates (min_dist >= 6)
            valid_mask = min_dists >= min_dist
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) == 0:
                # No valid candidates in this batch.
                # This might mean we are stuck or just unlucky.
                # To improve chances, we could try a smaller batch or different range?
                # Or just continue with new random batch.
                continue
            
            # Select the best candidate: the one with the maximum minimum distance
            # This helps in spreading out the codewords
            best_idx_in_valid = valid_indices[np.argmax(min_dists[valid_indices])]
            new_word = batch[best_idx_in_valid]
            
            # Check if it's already in C (shouldn't be, but just in case)
            if new_word not in C_list:
                C = np.append(C, new_word)
                C_list.append(new_word)
                
                if len(C_list) >= max_size:
                    break
        
        return C_list

    # Strategy: Multiple restarts with different initial seeds
    # Start with 0
    best_codewords = []
    best_size = 0
    
    # We can try starting with 0, or with a few random words
    # Starting with 0 is safe.
    
    # To increase the chance of finding a larger code, we can run multiple greedy passes
    # and keep the best result.
    
    # However, greedy is deterministic given the random seed.
    # We will just run the greedy search once, but with a "restart" mechanism if it stalls?
    # Actually, the NumPy greedy is fast enough to just run it.
    
    # Let's try a few different random seeds to see if we can find a better code
    # We have 55 seconds.
    
    # Run 1: Start with 0
    current_best = greedy_search([0])
    best_size = len(current_best)
    best_codewords = current_best
    
    # If we didn't reach a good size, try restarting with random seeds?
    # But the greedy search itself uses random batches, so it's stochastic.
    # We can just run it again if time permits?
    # The function `greedy_search` runs until deadline.
    # So we only get one run.
    
    # To make it better, we can modify `greedy_search` to do **local search** or **restarts** internally.
    # But for now, let's just return the result of the single run.
    # Given the speed, it should find a maximal code quickly.
    
    return best_codewords
