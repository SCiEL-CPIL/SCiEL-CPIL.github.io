# EVOLVE-BLOCK-START
import time
import itertools
import random


def construct(q, n, radius):
    # Task: q=3, n=7, radius=1
    # Total words: 3^7 = 2187
    # Ball size: 1 + 7*2 = 15
    
    # Fallback for other parameters (not expected in this specific evaluation)
    if q != 3 or n != 7 or radius != 1:
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return [list(word) for word in itertools.product(range(q), repeat=n)]

    total_words = q ** n
    deadline = time.time() + 110.0  # Safety margin before 120s limit

    # Precompute powers of q for index conversion
    q_pows = [1]
    for i in range(n):
        q_pows.append(q_pows[-1] * q)
    # q_pows[i] = q^i

    # Precompute balls as list of indices and bitmask
    ball_indices = [[] for _ in range(total_words)]
    ball_masks = [0] * total_words
    
    for i in range(total_words):
        neighbors = [i]
        mask = 1 << i
        # For each position
        for pos in range(n):
            current_digit = (i // q_pows[pos]) % q
            for new_digit in range(q):
                if new_digit == current_digit:
                    continue
                diff = new_digit - current_digit
                new_idx = i + diff * q_pows[pos]
                neighbors.append(new_idx)
                mask |= (1 << new_idx)
        ball_indices[i] = neighbors
        ball_masks[i] = mask

    full_mask = (1 << total_words) - 1

    # Convert an index to a list of digits
    def idx_to_word(idx):
        word = []
        temp = idx
        for _ in range(n):
            word.append(temp % q)
            temp //= q
        return word  # Note: word[0] is LSB, word[n-1] is MSB

    best_code = None
    best_size = float('inf')

    # Use the seed solution as a starting point
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_words = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    # Convert seed words to indices
    seed_indices = []
    for w in seed_words:
        # w is in MSB-first order (word[0] is most significant)
        # But idx_to_word gives LSB-first
        # So we need to reverse
        idx = 0
        for k in range(n):
            idx += w[k] * (q_pows[n - 1 - k])
        seed_indices.append(idx)
    
    # Helper to minimize a code using redundancy removal
    def minimize_code(code):
        # code is a list of indices
        # Initialize coverage counts
        counts = [0] * total_words
        for c in code:
            for w in ball_indices[c]:
                counts[w] += 1
        
        # Iterative removal
        changed = True
        while changed:
            changed = False
            # Iterate in reverse to safely remove
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                # Check if c is essential
                essential = False
                for w in ball_indices[c]:
                    if counts[w] == 1:
                        essential = True
                        break
                
                if not essential:
                    # Remove c
                    code.pop(i)
                    changed = True
                    # Update counts
                    for w in ball_indices[c]:
                        counts[w] -= 1
                    break # Restart the loop from the end? Or just continue? 
                    # Breaking and restarting ensures we don't miss opportunities 
                    # caused by the shift in indices, but it's slower.
                    # However, since we update counts immediately, continuing 
                    # to the next i (which is now a different element) is safe 
                    # because we are moving backwards.
                    # Actually, if we pop i, the element at i is now the one 
                    # that was at i+1. If we don't break, the loop moves to i-1.
                    # This is fine. We don't need to break.
        
        return code

    # Run greedy with minimization
    candidates = list(range(total_words))
    
    def greedy_cover(uncovered_mask, code_list):
        """Add codewords to code_list to cover all points in uncovered_mask."""
        unc = uncovered_mask
        while unc:
            max_count = -1
            best_cands = []
            for c in candidates:
                count = (ball_masks[c] & unc).bit_count()
                if count > max_count:
                    max_count = count
                    best_cands = [c]
                elif count == max_count:
                    best_cands.append(c)
            if best_cands:
                chosen = random.choice(best_cands)
            else:
                break
            code_list.append(chosen)
            unc &= ~ball_masks[chosen]
        return unc
    
    # Phase 1: Multiple random greedy runs
    while time.time() < deadline - 20:
        uncovered = full_mask
        code = []
        greedy_cover(uncovered, code)
        min_code = minimize_code(code)
        
        if len(min_code) < best_size:
            best_size = len(min_code)
            best_code = min_code[:]
        
        if best_size <= 186:
            break
    
    # Phase 2: Perturbation-based local search
    # Remove random codewords, re-greedy the holes, minimize
    if best_code is not None and time.time() < deadline - 10:
        current_code = best_code[:]
        current_size = best_size
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() < deadline - 5 and no_improve_count < max_no_improve:
            # Perturbation: remove k random codewords
            k = random.randint(1, min(5, len(current_code) - 1))
            removed = random.sample(current_code, k)
            removed_set = set(removed)
            
            # Compute uncovered after removal
            counts = [0] * total_words
            for c in current_code:
                if c in removed_set:
                    continue
                for w in ball_indices[c]:
                    counts[w] += 1
            
            # Find uncovered points
            uncovered = 0
            for w in range(total_words):
                if counts[w] == 0:
                    uncovered |= (1 << w)
            
            if not uncovered:
                no_improve_count += 1
                continue
            
            # Re-greedy cover the uncovered points
            perturbed_code = [c for c in current_code if c not in removed_set]
            greedy_cover(uncovered, perturbed_code)
            
            # Minimize
            new_code = minimize_code(perturbed_code)
            
            if len(new_code) < current_size:
                current_size = len(new_code)
                current_code = new_code[:]
                best_size = current_size
                best_code = current_code[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
    
    # Phase 3: Final polish - try removing one codeword at a time more aggressively
    if best_code is not None and time.time() < deadline - 3:
        code = best_code[:]
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for i in range(len(code)):
                c = code[i]
                # Check if removing c leaves everything covered
                counts = [0] * total_words
                for j, cj in enumerate(code):
                    if j == i:
                        continue
                    for w in ball_indices[cj]:
                        counts[w] += 1
                # Check if c's ball is fully covered by others
                can_remove = True
                for w in ball_indices[c]:
                    if counts[w] == 0:
                        can_remove = False
                        break
                if can_remove:
                    code.pop(i)
                    improved = True
                    best_size = len(code)
                    best_code = code[:]
                    break

    # Fallback: if we didn't find any code, use the seed
    if best_code is None:
        best_code = seed_indices[:]

    # Convert best_code indices to words
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
    
    return result
# EVOLVE-BLOCK-END
