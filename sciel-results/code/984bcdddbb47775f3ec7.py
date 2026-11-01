import numpy as np
import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Total number of words in the space
    total_words = q ** n
    
    # Precompute all words in the space as integers for faster distance calculation
    # We represent words as base-q integers
    all_words_int = np.arange(total_words)
    
    # Helper to convert integer representation to list of digits
    def int_to_digits(x):
        digits = []
        for _ in range(n):
            digits.append(x % q)
            x //= q
        return digits
    
    # Helper to convert list of digits to integer representation
    def digits_to_int(digits):
        x = 0
        for i, d in enumerate(reversed(digits)):
            x += d * (q ** i)
        return x
    
    # Precompute the "ball" of radius 1 around each word
    # A word covers itself and all words at Hamming distance 1
    # For q=3, n=7, radius=1, the ball size is 1 + n*(q-1) = 1 + 7*2 = 15
    
    # We will maintain a set of uncovered words
    # Using a boolean array for speed
    uncovered = np.ones(total_words, dtype=bool)
    
    selected_codewords = []
    
    # Greedy Set Cover
    while uncovered.any():
        if time.time() > deadline:
            break
            
        best_idx = -1
        max_covered = 0
        
        # To speed up, we can check candidates from the uncovered set
        # Or just iterate all words if the space is small enough (2187 is small)
        # Let's iterate all words but only consider those that cover at least one uncovered word
        
        # For efficiency, we can precompute the neighbors for each word
        # But doing it on the fly might be faster if we use vectorization
        
        # Let's try a vectorized approach for finding the best word
        # For each word w, count how many uncovered words are in its ball
        
        # Precompute neighbors for all words
        # neighbors[i] is a list of integers representing the ball around word i
        neighbors = []
        for w_int in range(total_words):
            w_digits = int_to_digits(w_int)
            ball = [w_int]
            for i in range(n):
                for d in range(q):
                    if d != w_digits[i]:
                        new_digits = w_digits.copy()
                        new_digits[i] = d
                        ball.append(digits_to_int(new_digits))
            neighbors.append(ball)
            
        # Convert neighbors to a numpy array for vectorized operations
        # neighbors_array[i] contains the indices of words covered by word i
        neighbors_array = np.array(neighbors, dtype=np.int32)
        
        # Count uncovered words covered by each candidate
        # We can do this by: for each word i, count sum(uncovered[neighbors_array[i]])
        # This is a bit slow if done naively.
        
        # Alternative: 
        # For each uncovered word u, increment the count for all words that cover u.
        # This is equivalent to: for each u in uncovered, for each v in neighbors[u], count[v] += 1
        # Since the graph is symmetric (u covers v iff v covers u), we can use the same neighbors array.
        
        counts = np.zeros(total_words, dtype=np.int32)
        
        # Get indices of uncovered words
        uncovered_indices = np.where(uncovered)[0]
        
        # For each uncovered word, add 1 to the count of its neighbors
        # This can be done with a loop over uncovered_indices, which is at most 2187
        # In each iteration, we add 1 to ~15 elements. Total ops ~ 2187 * 15 = 32805. Very fast.
        for u in uncovered_indices:
            counts[neighbors_array[u]] += 1
            
        # Find the word with the maximum count
        best_idx = np.argmax(counts)
        max_covered = counts[best_idx]
        
        if max_covered == 0:
            # This shouldn't happen if uncovered.any() is true
            break
            
        # Select the best word
        selected_codewords.append(best_idx)
        
        # Update uncovered set
        uncovered[neighbors_array[best_idx]] = False
        
    # Redundancy Removal
    # Try to remove each codeword if the remaining set still covers the space
    # We do this iteratively until no more can be removed
    
    # Convert selected_codewords to a list of lists for the final output
    # But first, let's prune
    
    # We will maintain a set of selected codewords
    # And check coverage by summing up the balls of the remaining codewords
    
    # To make pruning fast, we can use a bitset or boolean array
    # covered_by_current = sum of balls of current selected codewords
    
    # Let's represent the current selection as a list of indices
    current_selection = list(selected_codewords)
    
    # Precompute the union of balls for the current selection
    # We can do this by OR-ing the boolean masks of the balls
    # But we don't have boolean masks precomputed.
    
    # Let's create boolean masks for each word's ball
    # ball_mask[i] is a boolean array of length total_words
    # This might use too much memory: 2187 * 2187 bytes ~ 4.7 MB. Acceptable.
    
    ball_masks = np.zeros((total_words, total_words), dtype=bool)
    for i in range(total_words):
        ball_masks[i, neighbors_array[i]] = True
        
    # Compute the union of balls for the current selection
    current_cover = np.zeros(total_words, dtype=bool)
    for idx in current_selection:
        current_cover |= ball_masks[idx]
        
    # Pruning loop
    # We try to remove each word in the current selection
    # If removing it doesn't change the coverage, we remove it
    
    # To speed up, we can iterate through the list and check
    # If we remove word i, the new coverage is current_cover & ~ball_masks[i]
    # If this is still all True, we can remove i.
    
    # We do this in multiple passes because removing one word might make another removable
    
    changed = True
    while changed:
        changed = False
        new_selection = []
        for idx in current_selection:
            # Check if removing idx is safe
            # New coverage = current_cover & ~ball_masks[idx]
            # If new coverage is all True, remove idx
            # Note: current_cover is the coverage of the *current* selection including idx
            
            # Actually, a better way:
            # Let C be the coverage of the selection S.
            # If we remove i from S, the new coverage is C \ B(i) (set difference)
            # This is C & ~B(i)
            # If C & ~B(i) is all True, then i is redundant.
            
            # But we need to be careful: current_cover is the coverage of the *entire* current_selection.
            # If we are iterating and potentially removing previous elements, current_cover is stale.
            
            # So we should recompute current_cover if we remove an element, or do it in a smarter way.
            
            # Let's just recompute the coverage from scratch for each candidate removal?
            # That would be O(N^2 * ball_size). N ~ 200, ball_size ~ 15. 200^2 * 15 = 600,000 ops. Fast.
            
            pass
            
        # Let's implement the simple recompute approach for correctness and simplicity
        # We will try to remove each word in the current list
        # If it can be removed, we remove it and restart the loop (or continue with the new list)
        
        # To avoid O(N^3), we can just do one pass and if we remove one, we restart the pass.
        # Or we can just do a single pass, but then we might miss opportunities.
        # Given the small N, let's do multiple passes.
        
        for i in range(len(current_selection)):
            idx = current_selection[i]
            # Check if idx is redundant
            # Coverage of current_selection without idx
            # We can compute this by summing the balls of all other elements
            
            # Optimization: 
            # If we maintain the coverage of the *remaining* elements, we can check quickly.
            # But let's stick to the simple method first.
            
            # Compute coverage of current_selection without idx
            cover_without = np.zeros(total_words, dtype=bool)
            for j, other_idx in enumerate(current_selection):
                if j != i:
                    cover_without |= ball_masks[other_idx]
                    
            if cover_without.all():
                # idx is redundant
                current_selection.pop(i)
                changed = True
                break  # Restart the pass with the new list
                
    # Convert the final selection to the required format
    final_codewords = []
    for idx in current_selection:
        final_codewords.append(int_to_digits(idx))
        
    return final_codewords
