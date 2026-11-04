import numpy as np
import itertools
import random

def construct(q, n, radius):
    if radius != 1 or q != 3 or n != 8:
        # Fallback to direct sum for other parameters
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    total_words = q ** n  # 6561
    all_words = list(itertools.product(range(q), repeat=n))
    
    # Precompute spheres as bitmasks (integers)
    # Bit i is 1 if word i is in the sphere
    sphere_masks = [0] * total_words
    
    for i, word in enumerate(all_words):
        mask = 1 << i
        for pos in range(n):
            for val in range(q):
                if val != word[pos]:
                    new_word = list(word)
                    new_word[pos] = val
                    # Find index of new_word
                    # This is slow. We need a mapping.
                    # But we can't easily hash lists to indices without the dict.
                    # Let's use the dict from the original code.
        # We need the dict to map word -> index
        # Let's recreate it or pass it.
        
    # The original code had:
    # word_to_idx = {w: i for i, w in enumerate(all_words)}
    # We need this.
    
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    for i, word in enumerate(all_words):
        mask = 1 << i
        for pos in range(n):
            for val in range(q):
                if val != word[pos]:
                    new_word = list(word)
                    new_word[pos] = val
                    j = word_to_idx[tuple(new_word)]
                    mask |= (1 << j)
        sphere_masks[i] = mask
        
    best_code = None
    best_size = total_words
    
    # Run multiple random greedy attempts
    # Estimate time: One greedy run is ~400 steps.
    # Each step: 6561 popcounts.
    # 400 * 6561 * (cost of popcount on 800-byte int).
    # Popcount is very fast.
    # Let's try 20 restarts.
    
    for _ in range(20):
        uncovered = (1 << total_words) - 1
        code = []
        
        while uncovered:
            # Find best candidates
            max_count = 0
            candidates = []
            
            # Optimization: If we have many candidates, we can sample?
            # No, let's just find the max.
            # We can break early if we find a count == len(uncovered bits)?
            # But we need the list of candidates for randomization.
            
            # To speed up, we can pre-compute the list of indices?
            # Just loop.
            for i in range(total_words):
                # Count bits in sphere_masks[i] & uncovered
                # Python int bit_count() is available in 3.10+?
                # If not, use bin(x).count('1') which is slower.
                # Assuming Python 3.10+ for int.bit_count()
                count = (sphere_masks[i] & uncovered).bit_count()
                if count > max_count:
                    max_count = count
                    candidates = [i]
                elif count == max_count and count > 0:
                    candidates.append(i)
            
            if not candidates:
                break
                
            # Randomly pick from top candidates
            best_idx = random.choice(candidates)
            code.append(list(all_words[best_idx]))
            uncovered &= ~sphere_masks[best_idx]
            
        # Refinement: Remove redundant codewords
        # Calculate coverage counts
        coverage_count = [0] * total_words
        for c in code:
            idx = word_to_idx[tuple(c)]
            mask = sphere_masks[idx]
            # Iterate bits in mask?
            # Or just check if any word becomes uncovered.
            # Easier: Re-calculate uncovered set for the final code?
            # No, we can just try to remove each codeword and check validity.
            
        # Efficient removal:
        # Maintain a list of active codeword indices.
        active_indices = [word_to_idx[tuple(c)] for c in code]
        
        # We can try to remove each one.
        # To check if removal is valid, we need to know if any word is covered ONLY by this codeword.
        # Let's compute "essential" status.
        # For each word w, count how many active codewords cover it.
        
        # This O(N * |Code|) might be slow if done naively.
        # |Code| ~ 400. N = 6561.
        # 400 * 6561 = 2.6M ops. Fast.
        
        # Let's do a simple pass:
        # 1. Compute coverage count for all words given current active_indices.
        # 2. Identify words with count 1.
        # 3. If a codeword covers any word with count 1, it is essential.
        # 4. Remove non-essential codewords.
        # 5. Repeat until no more removals possible.
        
        while True:
            coverage_count = [0] * total_words
            # Build coverage counts
            for idx in active_indices:
                mask = sphere_masks[idx]
                # Add 1 to all bits set in mask
                # This is tricky with integers.
                # We can iterate through the 17 neighbors.
                word = all_words[idx]
                coverage_count[word_to_idx[tuple(word)]] += 1
                for pos in range(n):
                    for val in range(q):
                        if val != word[pos]:
                            new_word = list(word)
                            new_word[pos] = val
                            j = word_to_idx[tuple(new_word)]
                            coverage_count[j] += 1
            
            # Find essential codewords
            essential = [False] * len(active_indices)
            for i, idx in enumerate(active_indices):
                word = all_words[idx]
                # Check if any neighbor has count 1
                # Center
                if coverage_count[word_to_idx[tuple(word)]] == 1:
                    essential[i] = True
                    continue
                for pos in range(n):
                    for val in range(q):
                        if val != word[pos]:
                            new_word = list(word)
                            new_word[pos] = val
                            j = word_to_idx[tuple(new_word)]
                            if coverage_count[j] == 1:
                                essential[i] = True
                                break
                    if essential[i]:
                        break
            
            # Remove non-essential
            new_active = [idx for i, idx in enumerate(active_indices) if essential[i]]
            if len(new_active) == len(active_indices):
                break
            active_indices = new_active
        
        current_code = [list(all_words[idx]) for idx in active_indices]
        
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = current_code
            
    return best_code
