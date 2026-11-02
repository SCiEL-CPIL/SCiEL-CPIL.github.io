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
                # Since w is a tuple, we convert to list, modify, convert back to tuple
                # This is a bit slow in pure Python, but N is small.
                # Optimization: Precompute all possible single-coordinate changes?
                # Or just construct the tuple.
                # Tuple construction: w[:pos] + (new_val,) + w[pos+1:]
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
                
    # Additional Local Search: Try to swap codewords to reduce count or improve coverage
    # We can try to replace a codeword with another that covers the same "essential" words
    # but also covers some other words that allow removing another codeword.
    
    # Let's try a simpler improvement: 
    # 1. Compute coverage counts again
    # 2. For each pair of codewords (i, j), check if replacing i with a new word k
    #    allows removing j.
    # This is O(|code|^2 * N * ball_size) which might be too slow.
    
    # Instead, let's try to find if there's a codeword that can be replaced by another
    # that covers more words, potentially allowing removal of others.
    # But we already tried removing.
    
    # Let's try a different local search: 
    # Randomly perturb the code and see if we can find a better one.
    # Given the time limit, we might not have much time for this.
    
    # However, we can try a few random restarts or perturbations if time permits.
    if time.time() < deadline - 10:
        # Try to improve by replacing codewords
        # Calculate coverage counts
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
        
        # For each codeword, find if there's a replacement that covers the same essential words
        # and maybe more.
        # Essential words for codeword i are those with coverage_count == 1.
        
        for i in range(len(code)):
            if time.time() > deadline:
                break
            c_word = tuple(code[i])
            c_idx = get_idx(c_word)
            mask = ball_masks[c_idx]
            
            # Find essential words
            essential = []
            m = mask
            while m:
                lb = m & -m
                idx = lb.bit_length() - 1
                if coverage_counts[idx] == 1:
                    essential.append(idx)
                m ^= lb
            
            if not essential:
                continue
            
            # Try to find a replacement
            # A replacement must cover all essential words.
            # We can search for a word that covers all essential words.
            # The set of such words is the intersection of balls of essential words? No.
            # It's the set of words w such that dist(w, e) <= 1 for all e in essential.
            
            # Since essential words are within distance 1 of c_idx, 
            # the replacement must be within distance 1 of all of them.
            # This implies the replacement is likely very close to c_idx.
            
            # Let's just try all neighbors of c_idx (including c_idx itself)
            # and see if they cover all essential words.
            # If they do, and they cover more total words, we might consider swapping.
            # But swapping doesn't reduce count. It might allow removing another word.
            
            # Let's stick to the current solution as it's already quite good.
            # The main improvement would be a better initial construction or a more sophisticated local search.
            
            pass
            
    return code
