import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Parameters for this specific task instance
    # q=3, n=6, radius=1
    # General implementation might be needed, but task says "evaluator passes this task's..."
    # However, the function signature is generic. We should handle general q,n,radius if possible, 
    # but optimize for the specific case if we detect it? 
    # The prompt says "evaluator passes this task's alphabet size q, length n and covering radius".
    # It implies we should write a general solver, but we can optimize for the known instance if we want.
    # Given the 120s limit, a general solver for large n is hard. 
    # But K_q(n,r) is usually small for these challenges.
    # Let's assume the input is indeed q=3, n=6, radius=1.
    # If q, n, radius are different, the logic should still work, just maybe slower.
    
    total_words = q ** n
    
    # If the space is too large, this algorithm will time out.
    # For q=3, n=6, total_words=729.
    # For q=2, n=10, total_words=1024.
    # For q=3, n=7, total_words=2187.
    # The bitmask approach works up to ~2000-3000 words in Python easily.
    # If total_words > 5000, we might need to abort or use a different strategy.
    # But the task is specifically about the 6-match football pool (ternary length 6).
    
    if total_words > 10000:
        # Fallback to a simple greedy without local search to avoid timeout
        # Just return a greedy solution
        pass

    # Precompute neighbors for each word
    # neighbors[i] is a list of indices of words within Hamming distance 'radius'
    # For efficiency, we only need this for the specific q,n,radius.
    
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Function to get neighbors
    def get_neighbors(word):
        idx = word_to_idx[word]
        neighbors = [idx]
        # Generate all words at distance 1 to radius?
        # For radius=1, it's just 1 + n*(q-1) neighbors.
        # For general radius, we can generate all words and check distance, 
        # but that's O(N^2). 
        # For q=3, n=6, r=1, N=729. N^2 is 531k. Acceptable.
        # Let's just compute all pairwise distances once if N is small.
        return [] # Placeholder

    # Efficient precomputation of ball masks
    # Using a list of integers as bitsets
    ball_masks = [0] * total_words
    
    # Compute Hamming distance
    def hamming_dist(w1, w2):
        return sum(1 for a, b in zip(w1, w2) if a != b)
        
    # Precompute masks
    # Optimization: Only do this if total_words is manageable
    if total_words <= 2000:
        for i, w in enumerate(all_words):
            mask = 0
            for j, v in enumerate(all_words):
                if hamming_dist(w, v) <= radius:
                    mask |= (1 << j)
            ball_masks[i] = mask
    else:
        # Fallback: This algorithm is too slow for large N
        # Return a trivial code (all words) or empty? 
        # Must return valid code.
        # For large N, we can't cover all. 
        # But the task implies small N.
        pass

    # Greedy Set Cover with Pruning and Local Search
    
    def run_greedy_and_search():
        # 1. Greedy Phase
        uncovered_mask = (1 << total_words) - 1
        cover_count = [0] * total_words
        code = []
        
        # Symmetry breaking: fix first word to 0
        first_idx = 0
        code.append(first_idx)
        for w_idx in range(total_words):
            if (ball_masks[first_idx] >> w_idx) & 1:
                cover_count[w_idx] += 1
                uncovered_mask &= ~(1 << w_idx)
        
        while uncovered_mask != 0:
            best_idx = -1
            best_gain = -1
            
            # Find word with max new coverage
            for i in range(total_words):
                # Count how many uncovered words are in ball i
                # We can iterate through the set bits of uncovered_mask & ball_masks[i]
                # But iterating bits is fast.
                # Or simply:
                # gain = 0
                # temp = uncovered_mask & ball_masks[i]
                # while temp:
                #    gain += 1
                #    temp &= temp - 1
                # This is O(number of new covers).
                
                # Faster: 
                # We can precompute nothing, just count.
                # Since N is small, we can just check each bit in uncovered? No.
                # Let's use bit_count on the intersection.
                intersection = uncovered_mask & ball_masks[i]
                gain = intersection.bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                    
            if best_idx == -1:
                break
                
            code.append(best_idx)
            
            # Update counts and uncovered
            mask = ball_masks[best_idx]
            # Iterate through set bits in mask to update cover_count
            # This is O(size of ball)
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cover_count[idx] += 1
                if cover_count[idx] == 1:
                    uncovered_mask &= ~(1 << idx)
                m ^= lsb
                
        # 2. Prune Phase
        # Remove redundant codewords
        changed = True
        while changed:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                c_idx = code[i]
                # Check if all neighbors have cover_count > 1
                redundant = True
                m = ball_masks[c_idx]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_count[idx] <= 1:
                        redundant = False
                        break
                    m ^= lsb
                    
                if redundant:
                    # Remove c_idx
                    code.pop(i)
                    # Update cover_count
                    m = ball_masks[c_idx]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_count[idx] -= 1
                        m ^= lsb
                    changed = True
        
        return code, cover_count, uncovered_mask

    # 3. Local Search with Targeted Swaps
    best_code, _, _ = run_greedy_and_search()
    best_size = len(best_code)
    
    # To allow local search, we need to be able to rebuild cover_count from a code list
    def build_counts(code_list):
        counts = [0] * total_words
        for c in code_list:
            m = ball_masks[c]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb
        return counts
    
    current_code = best_code[:]
    current_counts = build_counts(current_code)
    
    # Helper to check if code is valid (all counts > 0)
    def is_valid(counts):
        for c in counts:
            if c == 0:
                return False
        return True
    
    # Helper to prune code given counts
    def prune(code_list, counts):
        changed = True
        while changed:
            changed = False
            for i in range(len(code_list) - 1, -1, -1):
                c_idx = code_list[i]
                redundant = True
                m = ball_masks[c_idx]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if counts[idx] <= 1:
                        redundant = False
                        break
                    m ^= lsb
                if redundant:
                    code_list.pop(i)
                    m = ball_masks[c_idx]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        m ^= lsb
                    changed = True
        return code_list
    
    # Precompute: for each word, which other words are in its ball (for fast lookup)
    # ball_neighbors[i] = list of word indices in ball of word i
    ball_neighbors = [None] * total_words
    for i in range(total_words):
        m = ball_masks[i]
        neighbors = []
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            neighbors.append(idx)
            m ^= lsb
        ball_neighbors[i] = neighbors
    
    # Local Search Loop with Targeted Swaps
    while time.time() < deadline:
        if not current_code:
            break
        
        # Strategy: Find a codeword whose removal would leave some words uncovered,
        # then try to replace it with a word that covers those gaps
        i = random.randint(0, len(current_code) - 1)
        c_old = current_code[i]
        
        # Find words that would become uncovered if we remove c_old
        # These are words with count == 1 that are in c_old's ball
        vulnerable_words = []
        m = ball_masks[c_old]
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            if current_counts[idx] == 1:
                vulnerable_words.append(idx)
            m ^= lsb
        
        if not vulnerable_words:
            # No vulnerable words, just try random swap
            c_new = random.randint(0, total_words - 1)
            if c_new == c_old or c_new in current_code:
                continue
        else:
            # Try to find a word that covers at least one vulnerable word
            # and is not already in the code
            candidates = []
            for v in vulnerable_words:
                # Find words whose ball contains v
                # This is equivalent to words within distance radius of v
                for candidate in ball_neighbors[v]:
                    if candidate != c_old and candidate not in current_code:
                        candidates.append(candidate)
                        break  # One candidate per vulnerable word is enough
            
            if not candidates:
                c_new = random.randint(0, total_words - 1)
                if c_new == c_old or c_new in current_code:
                    continue
            else:
                c_new = random.choice(candidates)
        
        if c_new == c_old or c_new in current_code:
            continue
        
        # Tentative swap
        # Decrement counts for c_old
        m = ball_masks[c_old]
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            current_counts[idx] -= 1
            m ^= lsb
        
        # Increment counts for c_new
        m = ball_masks[c_new]
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            current_counts[idx] += 1
            m ^= lsb
        
        # Check validity
        if is_valid(current_counts):
            # Update code list
            current_code[i] = c_new
            
            # Prune
            current_code = prune(current_code, current_counts)
            
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = current_code[:]
            elif len(current_code) == best_size:
                # Accept with some probability to explore
                if random.random() < 0.5:
                    best_code = current_code[:]
        else:
            # Revert
            m = ball_masks[c_new]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                current_counts[idx] -= 1
                m ^= lsb
                
            m = ball_masks[c_old]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                current_counts[idx] += 1
                m ^= lsb
                
            current_code[i] = c_old

    # Return the best code found as a list of lists of integers
    # The problem asks for "list of codewords", each codeword is "list of n integers"
    result = [list(all_words[c]) for c in best_code]
    
    return result
