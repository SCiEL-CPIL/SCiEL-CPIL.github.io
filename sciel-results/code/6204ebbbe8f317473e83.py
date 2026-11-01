import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # For this specific task, q=3, n=6, radius=1 is expected.
    # We handle the general case but optimize for the known parameters.
    if q != 3 or n != 6 or radius != 1:
        # Fallback to the seed logic or a simple greedy for other params
        # But the task specifies this exact problem.
        pass
        
    space_size = q ** n
    # Map words to indices 0..728
    # Word is a tuple of n ints
    
    # Precompute neighbors for each word
    # neighbors[i] is a list of indices of words within Hamming distance <= radius
    # For q=3, n=6, radius=1:
    # Center + 6 positions * 2 alternatives = 13 neighbors
    
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    neighbors = []
    for i, w in enumerate(all_words):
        ball = [i]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_w = w.copy()
                new_w[pos] = new_val
                ball.append(word_to_idx[tuple(new_w)])
        neighbors.append(ball)
        
    # Greedy Construction
    # uncovered: boolean array
    uncovered = [True] * space_size
    code_indices = []
    
    # To speed up greedy, we can just iterate. 
    # For each candidate c, count how many uncovered words it covers.
    # Pick the max.
    
    while any(uncovered):
        best_c = -1
        max_covered = -1
        # Optimization: if time is tight, we can sample, but 729 is small.
        for c in range(space_size):
            # If c is already in code, skip? No, greedy usually adds unique.
            # Actually, adding a duplicate is useless.
            if c in code_indices:
                continue
            cnt = 0
            for nb in neighbors[c]:
                if uncovered[nb]:
                    cnt += 1
            if cnt > max_covered:
                max_covered = cnt
                best_c = c
            if cnt == max_covered and cnt > 0:
                # Tie breaking: just keep first or random? 
                # Deterministic is fine.
                pass
                
        if best_c == -1:
            break # Should not happen
            
        code_indices.append(best_c)
        for nb in neighbors[best_c]:
            uncovered[nb] = False
            
        if time.time() > deadline:
            break
            
    # Refinement: Removal
    # coverage_count[i] = number of codewords in code that cover word i
    coverage_count = [0] * space_size
    for c in code_indices:
        for nb in neighbors[c]:
            coverage_count[nb] += 1
            
    # Helper to check if a codeword is redundant
    def is_redundant(c, coverage_count, neighbors):
        for nb in neighbors[c]:
            if coverage_count[nb] <= 1:
                return False
        return True

    # Helper to remove codeword
    def remove_codeword(c, code_indices, coverage_count, neighbors):
        for nb in neighbors[c]:
            coverage_count[nb] -= 1
        code_indices.remove(c)
        
    # Helper to add codeword
    def add_codeword(c, code_indices, coverage_count, neighbors):
        code_indices.append(c)
        for nb in neighbors[c]:
            coverage_count[nb] += 1

    # Iterative Removal
    changed = True
    while changed:
        changed = False
        for c in list(code_indices):
            if is_redundant(c, coverage_count, neighbors):
                remove_codeword(c, code_indices, coverage_count, neighbors)
                changed = True
        if time.time() > deadline:
            break

    # Swapping Phase
    # Try to replace c with c' if valid.
    # Valid if: for all nb in neighbors[c], if coverage_count[nb] == 1, then nb must be in neighbors[c'].
    # We can precompute the "critical" set for each c: criticals[c] = [nb for nb in neighbors[c] if coverage_count[nb]==1]
    # If criticals[c] is empty, c is redundant (handled above).
    # So we only need to check if neighbors[c'] covers criticals[c].
    
    # To make this efficient, we can iterate.
    # Since space is small, we can try all c' not in code.
    
    improved = True
    while improved:
        improved = False
        for c in list(code_indices):
            # Calculate critical words for c
            criticals = []
            for nb in neighbors[c]:
                if coverage_count[nb] == 1:
                    criticals.append(nb)
            
            if not criticals:
                # Should have been removed
                remove_codeword(c, code_indices, coverage_count, neighbors)
                improved = True
                break # Restart loop as indices changed
            
            # Try to find a replacement c'
            # c' must not be in code_indices
            # c' must cover all criticals
            
            # Optimization: Precompute a set of criticals for fast lookup?
            # Or just check.
            
            found_swap = False
            # Iterate over all possible c'
            # We can restrict c' to those that cover at least one critical?
            # Actually, c' MUST cover ALL criticals.
            
            # Heuristic: Try c' that are close to c? Or random?
            # Let's try all c' not in code. 729 * 729 is ~500k checks per pass.
            # Each check is len(criticals) lookups.
            # This might be slow if many passes.
            
            # Better: For each c', check if it covers criticals.
            # We can precompute for each word w, the list of codewords that cover it?
            # That's what coverage_count is, but we need specific c'.
            
            # Let's just iterate c'.
            # To speed up, we can check if c' is in code first.
            code_set = set(code_indices)
            
            for c_prime in range(space_size):
                if c_prime in code_set:
                    continue
                # Check if c_prime covers all criticals
                valid = True
                for nb in criticals:
                    # Is nb in neighbors[c_prime]?
                    # We can use a set for neighbors[c_prime] for O(1) lookup
                    # But neighbors are lists. Converting to set is O(13).
                    # Doing this for 729 c' and ~6 criticals is fine.
                    if nb not in neighbors[c_prime]:
                        valid = False
                        break
                if valid:
                    # Perform swap
                    # Remove c
                    remove_codeword(c, code_indices, coverage_count, neighbors)
                    code_set.remove(c)
                    # Add c_prime
                    add_codeword(c_prime, code_indices, coverage_count, neighbors)
                    code_set.add(c_prime)
                    improved = True
                    found_swap = True
                    break # Restart outer loop as state changed
                    
            if found_swap:
                break
                
        if time.time() > deadline:
            break

    # Final removal pass
    changed = True
    while changed:
        changed = False
        for c in list(code_indices):
            if is_redundant(c, coverage_count, neighbors):
                remove_codeword(c, code_indices, coverage_count, neighbors)
                changed = True
        if time.time() > deadline:
            break

    # Convert indices back to words
    result = [all_words[i] for i in code_indices]
    
    # Ensure distinctness (should be guaranteed by set logic)
    # And validity check (optional but good for debugging)
    # The evaluator checks validity.
    
    return result
