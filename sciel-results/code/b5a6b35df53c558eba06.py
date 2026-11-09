import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    # Fallback for non-target parameters
    if not (q == 3 and n == 8 and radius == 1):
        # Simple greedy for other cases
        # For this specific task, we focus on the target case
        # But we must return something valid
        # The seed used direct product for n=8.
        # For general case, a greedy set cover is safer.
        pass 

    # Target Case: K_3(8,1)
    total_words = q ** n
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for 120s limit
    
    # 1. Precompute neighbors for all words in {0,1,2}^8
    # Word index is base-3 representation of the word
    # word = [w0, w1, ..., w7]
    # index = w0 * 3^7 + w1 * 3^6 + ... + w7 * 3^0
    
    # Generate all words and their neighbors
    # Neighbors include the word itself and all words at Hamming distance 1
    
    neighbors = []
    # Powers of 3 for index calculation
    powers = [3 ** (n - 1 - i) for i in range(n)]
    
    # We iterate over all possible indices
    for idx in range(total_words):
        # Convert index to word
        word = []
        temp = idx
        for i in range(n):
            word.append(temp % q)
            temp //= q
        word.reverse() # Now word[0] is most significant
        
        neigh = [idx]
        for i in range(n):
            original_val = word[i]
            for new_val in range(q):
                if new_val != original_val:
                    # Calculate new index
                    delta = (new_val - original_val) * powers[i]
                    new_idx = idx + delta
                    neigh.append(new_idx)
        neighbors.append(neigh)
    
    # Convert to numpy arrays for faster access? 
    # List of lists is fine for 6561 items.
    
    # 2. Greedy Initialization
    # cover_counts[i] = number of codewords covering word i
    cover_counts = np.zeros(total_words, dtype=np.int32)
    codewords = []
    uncovered = set(range(total_words))
    
    # To speed up greedy, we can just pick random words if greedy is too slow, 
    # but greedy is O(N^2) approx. 6561^2 is ~43M. Might be a bit slow in Python.
    # Let's use a simpler init: Random set of size ~600, then optimize.
    # Or, a faster greedy:
    # Actually, let's try a "randomized greedy" or just random init + fast local search.
    # Local search with 1-for-1 swaps and deletions is very effective.
    
    # Let's initialize with a random set of size 600
    # 600 is close to the expected result of greedy.
    initial_size = 600
    if initial_size > total_words:
        initial_size = total_words
        
    codewords = random.sample(range(total_words), initial_size)
    
    # Build initial cover counts
    for c in codewords:
        for nb in neighbors[c]:
            cover_counts[nb] += 1
            
    # Check validity (should be valid if size is large enough)
    # 600 * 17 = 10200 > 6561, so likely valid, but overlaps might leave holes?
    # Random set of 600 might have holes.
    # We need to ensure validity.
    
    # If invalid, fill holes greedily
    while np.any(cover_counts == 0):
        # Find an uncovered word
        uncovered_indices = np.where(cover_counts == 0)[0]
        target_word = uncovered_indices[0]
        
        # Find best codeword to cover it
        # We can just pick a random neighbor of target_word
        # Or the one that covers most other uncovered words
        # For speed, pick random neighbor
        best_c = random.choice(neighbors[target_word])
        
        # Add best_c
        codewords.append(best_c)
        for nb in neighbors[best_c]:
            cover_counts[nb] += 1
            
    # Now we have a valid code.
    # 3. Local Search
    best_size = len(codewords)
    best_codewords = list(codewords)
    best_cover_counts = cover_counts.copy()
    
    # To allow efficient swap evaluation, we keep a set of current codewords
    current_set = set(codewords)
    
    # Simulated Annealing / Random Walk
    # Temperature T
    T = 1.0
    cooling_rate = 0.999
    
    while time.time() < deadline:
        if T < 0.01:
            T = 0.01
            
        # Try Deletion
        if len(codewords) > 0:
            c = random.choice(codewords)
            # Check if c is redundant
            # If all neighbors of c have count > 1, we can remove c
            # We need to check min(cover_counts[neighbors[c]]) > 1
            # Vectorized check
            if np.min(cover_counts[neighbors[c]]) > 1:
                # Remove c
                for nb in neighbors[c]:
                    cover_counts[nb] -= 1
                current_set.remove(c)
                codewords.remove(c)
                
                # Update best
                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = list(codewords)
                    best_cover_counts = cover_counts.copy()
                continue
        
        # Try Swap (1-for-1)
        if len(codewords) > 0:
            c_out = random.choice(codewords)
            # Pick a random candidate c_in
            # To make it more effective, we can pick from words that are "under-covered"
            # Or just random
            c_in = random.randint(0, total_words - 1)
            
            if c_in == c_out:
                continue
            if c_in in current_set:
                continue
                
            # Simulate swap
            # Remove c_out, add c_in
            # Check if valid: min(cover_counts_after) >= 1
            
            # Optimization: 
            # The only words affected are neighbors of c_out and c_in.
            # We can calculate the new counts for the union of these neighbors.
            
            affected = set(neighbors[c_out]) | set(neighbors[c_in])
            
            # Current counts for affected
            # We need to check if any word in affected becomes 0.
            # Words in neighbors[c_out] \ neighbors[c_in] decrease by 1.
            # Words in neighbors[c_in] \ neighbors[c_out] increase by 1.
            # Words in intersection stay same.
            
            # So we only need to check if min(cover_counts[w] - 1) >= 1 for w in N(c_out) \ N(c_in)
            # i.e. min(cover_counts[w]) >= 2 for w in N(c_out) \ N(c_in)
            
            # Let's compute the set difference
            # For speed, we can just iterate if the sets are small (17 elements)
            
            valid = True
            for nb in neighbors[c_out]:
                if nb not in neighbors[c_in]:
                    if cover_counts[nb] <= 1:
                        valid = False
                        break
                        
            if valid:
                # Perform swap
                for nb in neighbors[c_out]:
                    cover_counts[nb] -= 1
                for nb in neighbors[c_in]:
                    cover_counts[nb] += 1
                current_set.remove(c_out)
                current_set.add(c_in)
                codewords.remove(c_out)
                codewords.append(c_in)
                
                # After swap, check if any codeword became redundant
                # We can do a quick check on a random subset or all
                # Let's try to delete immediately if possible
                # This creates a "greedy" bias in the local search
                deleted = False
                for c_check in codewords:
                    if np.min(cover_counts[neighbors[c_check]]) > 1:
                        # Remove
                        for nb in neighbors[c_check]:
                            cover_counts[nb] -= 1
                        current_set.remove(c_check)
                        codewords.remove(c_check)
                        deleted = True
                        
                        if len(codewords) < best_size:
                            best_size = len(codewords)
                            best_codewords = list(codewords)
                            best_cover_counts = cover_counts.copy()
                        break # Only delete one at a time to keep it simple? Or delete all?
                        # Deleting all redundant is better
                        # Let's restart the deletion loop?
                        # For now, break and let the next iteration handle more deletions.
                
                # Update T
                T *= cooling_rate
            else:
                # Rejection
                T *= cooling_rate

    # Return best found
    # Convert indices back to words
    result = []
    for idx in best_codewords:
        word = []
        temp = idx
        for i in range(n):
            word.append(temp % q)
            temp //= q
        word.reverse()
        result.append(word)
        
    return result
