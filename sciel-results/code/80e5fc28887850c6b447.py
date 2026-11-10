"""Improved program for the suite task; uses greedy set cover with bitsets."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct(q, n, radius):
    # The task specifies q=3, n=6, radius=1.
    # We implement a general greedy set cover for small q, n, radius.
    # However, given the specific constraints and time limit, we optimize for the known case.
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback to the original direct product construction if parameters differ
        # This ensures the function works for other potential test cases, though the task is specific.
        if n < 4:
            return [list(t) for t in itertools.product(range(q), repeat=n)]
        
        # Heuristic fallback: use tetracode if possible, else greedy
        # For now, we'll just run the greedy algorithm generally, but it might be slow for large n.
        # Since the prompt implies this specific task, we focus on the q=3, n=6 case.
        # If other cases appear, we might need a more robust fallback.
        # Let's implement a general greedy but with a time guard.
        
        start_time = time.time()
        deadline = 110.0  # 110s limit to be safe
        
        total_words = q ** n
        # Map word to index
        def word_to_idx(w):
            idx = 0
            for d in w:
                idx = idx * q + d
            return idx
        
        def idx_to_word(idx):
            w = [0] * n
            for i in range(n-1, -1, -1):
                w[i] = idx % q
                idx //= q
            return w
        
        # Precompute coverage for each candidate
        # This is O(q^n * (1 + n*(q-1)))
        # For general n, this might be too slow if n is large.
        # But for n=6, q=3, it's 729 * 13 ~ 9k ops. Very fast.
        
        candidates = []
        coverage_masks = []
        
        for c_idx in range(total_words):
            c_word = idx_to_word(c_idx)
            covered = set()
            covered.add(c_idx)
            for pos in range(n):
                orig_val = c_word[pos]
                for new_val in range(q):
                    if new_val != orig_val:
                        new_word = c_word[:]
                        new_word[pos] = new_val
                        covered.add(word_to_idx(new_word))
            
            # Create bitmask
            mask = 0
            for idx in covered:
                mask |= (1 << idx)
            candidates.append(c_word)
            coverage_masks.append(mask)
            
            if time.time() > deadline:
                break
        
        if not candidates:
             # Emergency fallback
             return [list(t) for t in itertools.product(range(q), repeat=n)]

        # Greedy Set Cover
        all_words_mask = (1 << total_words) - 1
        uncovered = all_words_mask
        selected_indices = []
        
        while uncovered:
            best_mask = 0
            best_idx = -1
            # Find candidate covering most uncovered words
            for i, mask in enumerate(coverage_masks):
                # Count bits in (mask & uncovered)
                # Python 3.10+ has int.bit_count()
                count = (mask & uncovered).bit_count()
                if count > (best_mask & uncovered).bit_count():
                    best_mask = mask
                    best_idx = i
                    if count == 0:
                        break # No progress possible? Should not happen if valid input
            
            if best_idx == -1 or (best_mask & uncovered) == 0:
                break
            
            selected_indices.append(best_idx)
            uncovered &= ~best_mask
            
            if time.time() > deadline:
                break
        
        if not selected_indices:
            # Fallback to direct product if greedy failed or didn't run
            if n >= 4 and q == 3:
                 tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
                 return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
            else:
                 return [list(t) for t in itertools.product(range(q), repeat=n)]
        
        return [candidates[i] for i in selected_indices]

    # Specific optimization for q=3, n=6, radius=1
    # 3^6 = 729 words
    total_words = 729
    
    # Precompute index mapping
    # We can generate candidates and their coverage masks
    
    candidates = []
    coverage_masks = []
    
    for c_idx in range(total_words):
        # Convert index to word
        w = [0] * 6
        temp = c_idx
        for i in range(5, -1, -1):
            w[i] = temp % 3
            temp //= 3
        candidates.append(w)
        
        # Calculate coverage mask
        mask = 0
        # Self
        mask |= (1 << c_idx)
        
        # Neighbors
        for pos in range(6):
            orig = w[pos]
            for new_val in range(3):
                if new_val != orig:
                    # Calculate new index efficiently
                    # delta = (new_val - orig) * 3^pos
                    # Since we are in base 3, we can just adjust the digit
                    # But integer arithmetic is safer.
                    diff = new_val - orig
                    new_idx = c_idx + diff * (3 ** pos)
                    mask |= (1 << new_idx)
        
        coverage_masks.append(mask)
    
    # Greedy Set Cover
    all_words_mask = (1 << total_words) - 1
    uncovered = all_words_mask
    selected_indices = []
    
    # Optimization: Instead of iterating all 729 every time, we can just do it.
    # 729 is small.
    while uncovered:
        best_mask = 0
        best_idx = -1
        max_count = 0
        
        for i, mask in enumerate(coverage_masks):
            # Check if this candidate adds anything
            # We want to maximize (mask & uncovered)
            # bit_count is fast
            count = (mask & uncovered).bit_count()
            if count > max_count:
                max_count = count
                best_mask = mask
                best_idx = i
        
        if best_idx == -1 or max_count == 0:
            break
        
        selected_indices.append(best_idx)
        uncovered &= ~best_mask
    
    return [candidates[i] for i in selected_indices]

# EVOLVE-BLOCK-END
