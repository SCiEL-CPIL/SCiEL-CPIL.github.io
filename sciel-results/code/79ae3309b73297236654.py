import itertools
import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Total number of words
    total_words = q ** n
    
    # Precompute all words as tuples
    all_words = list(itertools.product(range(q), repeat=n))
    
    # Map word to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute ball masks for each word
    # For radius 1, we can generate neighbors directly
    ball_masks = [0] * total_words
    
    # Helper to get index from word
    def get_idx(word):
        return word_to_idx[word]
        
    # Precomputation step: Generate neighbors for each word
    # Since radius is 1, neighbors are at distance 0 or 1.
    # Distance 0: the word itself.
    # Distance 1: change one coordinate to a different value.
    for i, w in enumerate(all_words):
        mask = 1 << i  # Include self
        
        # Generate distance 1 neighbors
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                # Create new word
                new_word = w[:pos] + (new_val,) + w[pos+1:]
                j = word_to_idx[new_word]
                mask |= (1 << j)
                
        ball_masks[i] = mask
        
    # Initialize uncovered set as all 1s
    # Using an integer as a bitset
    uncovered = (1 << total_words) - 1
    
    code = []
    code_set = set()
    
    # Symmetry breaking: Fix the first codeword to all zeros
    first_word = (0,) * n
    first_idx = get_idx(first_word)
    code.append(list(first_word))
    code_set.add(first_word)
    uncovered &= ~ball_masks[first_idx]
    
    # Greedy Set Cover
    while uncovered != 0:
        if time.time() > deadline:
            break
            
        best_idx = -1
        best_count = 0
        
        # Find the word that covers the most uncovered words
        for i in range(total_words):
            if all_words[i] in code_set:
                continue
            covered = uncovered & ball_masks[i]
            count = covered.bit_count()
            if count > best_count:
                best_count = count
                best_idx = i
                
        if best_idx == -1:
            break
            
        # Add the best word
        best_word = all_words[best_idx]
        code.append(list(best_word))
        code_set.add(best_word)
        uncovered &= ~ball_masks[best_idx]
        
    # Local Search: Try to remove codewords
    # Iterate until no more can be removed
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        # Try removing each codeword
        # We need to check if the remaining code still covers everything.
        # Efficient check:
        # Let U be the union of balls of all codewords except C_i.
        # If U covers all words, C_i can be removed.
        # This is expensive to compute from scratch each time.
        # Alternative:
        # Maintain a count of how many codewords cover each word.
        # coverage_count[w] = number of codewords c such that dist(c,w) <= radius.
        # If we remove c, coverage_count[w] decreases by 1 for all w in ball(c).
        # If any coverage_count[w] becomes 0, c cannot be removed.
        
        # Let's implement the coverage count approach.
        # This is O(N * |code| * ball_size) to initialize, then O(ball_size) per removal attempt.
        
        # Re-initialize coverage counts
        coverage_counts = [0] * total_words
        for c_word in code:
            c_idx = get_idx(tuple(c_word))
            mask = ball_masks[c_idx]
            # Add 1 to all bits in mask
            # This is tricky with integers. We can iterate through set bits.
            # Or use numpy?
            # Since N=729, iterating through set bits of a 729-bit int is fast.
            m = mask
            while m:
                # Get lowest set bit
                lb = m & -m
                idx = lb.bit_length() - 1
                coverage_counts[idx] += 1
                m ^= lb
                
        # Try removing each codeword
        for k in range(len(code) - 1, -1, -1): # Try removing from the end first (greedy added last)
            if time.time() > deadline:
                break
            c_word = tuple(code[k])
            c_idx = get_idx(c_word)
            mask = ball_masks[c_idx]
            
            # Check if any word in this ball has coverage_count == 1
            # If yes, removing this codeword would leave that word uncovered.
            can_remove = True
            m = mask
            while m:
                lb = m & -m
                idx = lb.bit_length() - 1
                if coverage_counts[idx] == 1:
                    can_remove = False
                    break
                m ^= lb
                
            if can_remove:
                # Remove it
                code.pop(k)
                # Update coverage counts
                m = mask
                while m:
                    lb = m & -m
                    idx = lb.bit_length() - 1
                    coverage_counts[idx] -= 1
                    m ^= lb
                improved = True
                # Restart the loop or continue?
                # Continuing is fine, but indices shift.
                # Breaking to restart the while loop is safer.
                break
                
    # Additional local search: try swapping codewords to reduce count
    # This is a more aggressive local search that tries to replace one codeword
    # with another that covers more uncovered words
    swap_improved = True
    while swap_improved:
        if time.time() > deadline:
            break
        swap_improved = False
        
        # Compute coverage counts
        coverage_counts = [0] * total_words
        for c_word in code:
            c_idx = get_idx(tuple(c_word))
            mask = ball_masks[c_idx]
            m = mask
            while m:
                lb = m & -m
                idx = lb.bit_length() - 1
                coverage_counts[idx] += 1
                m ^= lb
        
        # Try swapping each codeword with a better one
        for k in range(len(code)):
            if time.time() > deadline:
                break
            
            c_word = tuple(code[k])
            c_idx = get_idx(c_word)
            c_mask = ball_masks[c_idx]
            
            # Find words that are only covered by this codeword
            # These are the "essential" words for this codeword
            essential_words = []
            m = c_mask
            while m:
                lb = m & -m
                idx = lb.bit_length() - 1
                if coverage_counts[idx] == 1:
                    essential_words.append(idx)
                m ^= lb
            
            if not essential_words:
                continue
            
            # Try to find a replacement codeword that covers all essential words
            # and potentially more uncovered words
            best_swap_idx = -1
            best_swap_gain = 0
            
            for i in range(total_words):
                if all_words[i] in code_set:
                    continue
                
                new_mask = ball_masks[i]
                
                # Check if new codeword covers all essential words
                covers_essential = True
                for ew in essential_words:
                    if not (new_mask & (1 << ew)):
                        covers_essential = False
                        break
                
                if not covers_essential:
                    continue
                
                # Calculate gain: how many more words does the new codeword cover
                # compared to the old one
                old_only = c_mask & ~new_mask
                new_only = new_mask & ~c_mask
                
                old_count = old_only.bit_count()
                new_count = new_only.bit_count()
                
                gain = new_count - old_count
                
                if gain > best_swap_gain:
                    best_swap_gain = gain
                    best_swap_idx = i
            
            if best_swap_idx != -1 and best_swap_gain > 0:
                # Perform the swap
                old_word = code[k]
                new_word = all_words[best_swap_idx]
                
                code[k] = list(new_word)
                code_set.discard(old_word)
                code_set.add(new_word)
                
                # Update coverage counts
                old_mask = c_mask
                new_mask = ball_masks[best_swap_idx]
                
                m = old_mask
                while m:
                    lb = m & -m
                    idx = lb.bit_length() - 1
                    coverage_counts[idx] -= 1
                    m ^= lb
                
                m = new_mask
                while m:
                    lb = m & -m
                    idx = lb.bit_length() - 1
                    coverage_counts[idx] += 1
                    m ^= lb
                
                swap_improved = True
                break
                
    return code
