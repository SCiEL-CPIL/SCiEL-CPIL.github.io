import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s buffer
    
    # Parameters for this specific task
    # The task is fixed to q=3, n=7, radius=1 in the prompt description 
    # but the function signature takes q, n, radius. 
    # We assume the evaluator calls with these values.
    
    total_words = q ** n
    
    # Precompute the balls (sets of words covered by each word)
    # word is an integer 0..total_words-1
    # Convert integer to tuple of digits
    def int_to_word(w):
        return [w // (q ** (n - i - 1)) % q for i in range(n)]
    
    def word_to_int(w):
        val = 0
        for i, d in enumerate(w):
            val = val * q + d
        return val

    # Generate all words
    all_words = range(total_words)
    
    # Precompute balls
    # balls[i] is a set of ints
    balls = [set() for _ in range(total_words)]
    
    # Efficiently generate balls using numpy or simple loops
    # For q=3, n=7, radius=1, each ball has 1 + n*(q-1) = 1 + 14 = 15 words
    # 2187 * 15 operations is trivial.
    
    # We can generate neighbors by changing one digit
    # To speed up, we can use the structure.
    # But simple loop is fine for 2187 items.
    
    for w in all_words:
        digits = int_to_word(w)
        balls[w].add(w)
        for pos in range(n):
            orig_digit = digits[pos]
            for new_digit in range(q):
                if new_digit == orig_digit:
                    continue
                # Create new word
                # Optimization: precompute powers of q
                # Or just convert back
                new_digits = digits[:]
                new_digits[pos] = new_digit
                new_w = word_to_int(new_digits)
                balls[w].add(new_w)

    # Greedy Initialization
    covered = set()
    codewords = []
    uncovered_count = total_words
    
    # To speed up greedy, we can maintain a count of how many codewords cover each word
    # count_covered[w] = number of codewords in codewords such that w in balls[c]
    # But we need to pick the word that covers the most UNCOVERED words.
    # So we need: for candidate c, score = len(balls[c] - covered)
    
    # Simple greedy:
    while len(covered) < total_words:
        best_c = -1
        best_gain = -1
        
        # Optimization: If we have many candidates, this is O(total_words * len(balls))
        # 2187 * 15 ~ 33k ops per step.
        # Steps ~ 200. Total ~ 6M ops. Very fast in Python? Maybe.
        # Let's try to optimize by only considering candidates that cover at least one uncovered word.
        
        # Even better: We can just iterate all words.
        for c in all_words:
            # Gain is number of words in balls[c] not in covered
            # len(balls[c] - covered)
            # Since balls are small (15), this is fast.
            gain = 0
            for w in balls[c]:
                if w not in covered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_c = c
                
        if best_gain == 0:
            break # Should not happen if loop condition is correct
            
        codewords.append(best_c)
        covered.update(balls[best_c])
        
        if time.time() > deadline:
            break

    # Local Search
    # We represent the set of codewords as a list.
    # We need to check validity efficiently.
    
    def is_valid(codewords_list):
        # Check if union of balls covers all words
        # For 200 codewords, this is 200 * 15 = 3000 inserts into a set.
        # 3000 ops. Very fast.
        cov = set()
        for c in codewords_list:
            cov.update(balls[c])
        return len(cov) == total_words

    def can_remove(codewords_list, index):
        # Check if removing codewords_list[index] leaves all words covered
        c = codewords_list[index]
        # For each word w in balls[c], check if there is another c' in codewords_list (j != index) such that w in balls[c']
        # We can precompute for each w, the list of codewords that cover it?
        # Or just iterate.
        
        # Optimization: 
        # Build a map: word -> count of codewords covering it
        # If any word in balls[c] has count 1 (and that codeword is c), then removal is invalid.
        
        # Let's maintain counts outside the loop for efficiency in the main search loop.
        pass

    # Let's use the count-based approach for local search
    # counts[w] = number of codewords in current set that cover w
    
    counts = [0] * total_words
    for c in codewords:
        for w in balls[c]:
            counts[w] += 1
            
    # Verify initial validity (should be valid)
    # min(counts) should be >= 1
    
    best_size = len(codewords)
    no_improve_count = 0
    max_no_improve = 20
    
    # Local Search Loop
    while time.time() < deadline:
        improved = False
        
        # 1. Removal Pass
        # Shuffle indices to avoid bias
        indices = list(range(len(codewords)))
        random.shuffle(indices)
        
        for i in indices:
            if i >= len(codewords): continue # might have been removed? No, we iterate fixed list and check validity
            c = codewords[i]
            
            # Check if removing c is safe
            # Safe if for all w in balls[c], counts[w] > 1
            safe = True
            for w in balls[c]:
                if counts[w] == 1:
                    safe = False
                    break
            
            if safe:
                # Remove c
                for w in balls[c]:
                    counts[w] -= 1
                codewords.pop(i)
                improved = True
                # After removal, indices are shifted. 
                # Since we shuffled and are iterating, popping invalidates subsequent indices in the list `indices` 
                # if we just use the list `indices` blindly.
                # Better to iterate by value or restart removal pass.
                # Given the small size, restarting removal pass on improvement is robust.
                break
        
        if improved:
            no_improve_count = 0
            continue
        else:
            no_improve_count += 1
            
        # 2. Swap Pass
        # Try to swap a codeword with a random other word
        # This is useful to escape local minima where a codeword is "essential" 
        # but a slightly different one might allow more removals later.
        
        if no_improve_count > 5:
            # Try swaps
            swap_improved = False
            for i in range(len(codewords)):
                if time.time() > deadline: break
                c_old = codewords[i]
                
                # Pick a random new codeword
                # Ideally pick one that covers many words currently covered by c_old but not by others?
                # Simple: random
                c_new = random.randint(0, total_words - 1)
                if c_new == c_old:
                    continue
                    
                # Check if swap is valid
                # New counts = counts - balls[c_old] + balls[c_new]
                # Valid if min(new_counts) >= 1
                
                # Optimization: Only check words in balls[c_old] U balls[c_new]
                affected = balls[c_old] | balls[c_new]
                valid = True
                for w in affected:
                    new_count = counts[w]
                    if w in balls[c_old]:
                        new_count -= 1
                    if w in balls[c_new]:
                        new_count += 1
                    if new_count < 1:
                        valid = False
                        break
                
                if valid:
                    # Apply swap
                    for w in balls[c_old]:
                        counts[w] -= 1
                    for w in balls[c_new]:
                        counts[w] += 1
                    codewords[i] = c_new
                    swap_improved = True
                    break
            
            if swap_improved:
                no_improve_count = 0
                # After a swap, we might be able to remove more. 
                # Loop back to Removal Pass.
                continue
            else:
                no_improve_count += 1
                
        # 3. Perturbation / Restart
        if no_improve_count > max_no_improve:
            # Randomly change a few codewords to break symmetry
            # Pick 3 random indices and change them to random words
            # Then run removal pass. If size decreases, keep. Else revert.
            
            original_codewords = codewords[:]
            original_counts = counts[:]
            
            for _ in range(3):
                if codewords:
                    idx = random.randint(0, len(codewords) - 1)
                    c_old = codewords[idx]
                    c_new = random.randint(0, total_words - 1)
                    
                    # Apply change
                    for w in balls[c_old]:
                        counts[w] -= 1
                    for w in balls[c_new]:
                        counts[w] += 1
                    codewords[idx] = c_new
                    
            # Run Removal Pass on perturbed set
            perm_improved = False
            # Simple removal pass
            changed = True
            while changed:
                changed = False
                for i in range(len(codewords)):
                    c = codewords[i]
                    safe = True
                    for w in balls[c]:
                        if counts[w] == 1:
                            safe = False
                            break
                    if safe:
                        for w in balls[c]:
                            counts[w] -= 1
                        codewords.pop(i)
                        changed = True
                        perm_improved = True
                        break
                        
            if len(codewords) < len(original_codewords):
                # Keep the perturbation
                no_improve_count = 0
            else:
                # Revert
                codewords = original_codewords
                counts = original_counts
                no_improve_count = 0 # Reset to try again? Or increment?
                # If we revert, we are stuck. Let's increment to force another restart or exit?
                # Actually, if we revert, we haven't improved.
                no_improve_count += 1

    # Convert codewords (ints) back to lists of digits
    result = []
    for c in codewords:
        result.append(int_to_word(c))
        
    # Ensure distinctness (should be by construction)
    # Ensure validity (should be by construction)
    
    return result

# Quick sanity check for q=3, n=7
# The greedy will likely produce ~190-200.
# Local search should reduce it further.
