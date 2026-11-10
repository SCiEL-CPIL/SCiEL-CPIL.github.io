import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    # Fallback for non-standard parameters
    if q != 3 or n != 7 or radius != 1:
        if q == 3 and n >= 4 and radius == 1:
            # Tetracode for n=4
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            # Product code for n > 4
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        # Generic fallback: all words (valid but large)
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    # Specific case: q=3, n=7, radius=1
    total_words = 3 ** 7  # 2187
    all_words = [tuple(w) for w in itertools.product(range(3), repeat=7)]
    word_index = {w: i for i, w in enumerate(all_words)}
    
    # Precompute balls (words within distance 1)
    balls = []
    for w in all_words:
        ball = set()
        ball.add(w)
        for pos in range(7):
            for val in range(3):
                if val != w[pos]:
                    new_w = list(w)
                    new_w[pos] = val
                    ball.add(tuple(new_w))
        balls.append(ball)
        
    # Convert to indices for faster operations
    ball_indices = []
    for b in balls:
        ball_indices.append([word_index[v] for v in b])
    ball_indices_np = np.array(ball_indices, dtype=np.int32)
    
    # --- Initial Construction: Decomposition ---
    # Partition by first 4 coordinates (prefix). There are 3^4 = 81 prefixes.
    # For each prefix p, the sub-space is {p} x {0,1,2}^3.
    # We need to cover {0,1,2}^3 with radius 1.
    # The optimal covering code for q=3, n=3, r=1 is the Tetracode (9 words).
    # Tetracode words: (a, b, a+b mod 3) for a,b in {0,1,2}
    tetracode_3 = [(a, b, (a + b) % 3) for a in range(3) for b in range(3)]
    
    initial_code = []
    for prefix in itertools.product(range(3), repeat=4):
        p_tuple = prefix
        for t in tetracode_3:
            full_word = p_tuple + t
            initial_code.append(full_word)
            
    # Convert to list of indices
    code_indices = [word_index[w] for w in initial_code]
    covered = np.zeros(total_words, dtype=np.int32)
    for idx in code_indices:
        covered[ball_indices_np[idx]] += 1
    
    # Check validity (should be valid by construction)
    # assert np.all(covered > 0)
    
    # --- Local Search: Remove and Repair ---
    # Goal: Minimize len(code_indices)
    # Strategy: 
    # 1. Try to remove a codeword if it is redundant (all words in its ball are covered by others).
    # 2. If not redundant, try to swap it with a different codeword that covers the "lost" words 
    #    and potentially gains new coverage (though in this tight space, gains are rare).
    #    Actually, since we start with a valid cover, removing a non-redundant word creates holes.
    #    We can try to fill holes with one new word. If a single word fills all holes created by removal,
    #    it's a valid swap (size remains same). If we can remove without replacement, size decreases.
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Shuffle order to avoid bias
        random.shuffle(code_indices)
        
        for i in range(len(code_indices)):
            if time.time() > deadline - 5:
                break
                
            idx_to_remove = code_indices[i]
            ball_to_remove = ball_indices_np[idx_to_remove]
            
            # Check if redundant
            # A word is redundant if for every word v in its ball, covered[v] > 1
            # (meaning some other codeword covers v)
            
            # Efficient check:
            # min_cover_in_ball = np.min(covered[ball_to_remove])
            # if min_cover_in_ball > 1:
            #     # Redundant
            #     covered[ball_to_remove] -= 1
            #     code_indices.pop(i)
            #     improved = True
            #     break
            # else:
            #     # Not redundant. Find the words that become uncovered (covered == 1)
            #     lost_mask = (covered[ball_to_remove] == 1)
            #     lost_words = ball_to_remove[lost_mask]
            #     
            #     if len(lost_words) == 0:
            #         # Should not happen if min_cover <= 1, but safe
            #         pass
            #     else:
            #         # Try to find a replacement word w such that ball(w) covers all lost_words
            #         # AND w is not in the current code (or we are swapping)
            #         # Note: w can be any word in the space.
            #         # Optimization: The replacement word must be within distance 1 of all lost_words.
            #         # This is a strong constraint.
            #         
            #         # Iterate through potential replacements? 
            #         # To speed up, we can check candidates that are "close" to the removed word 
            #         # or close to the lost words.
            #         # However, a global search over 2187 words for each removal (2187 * 2187 * 15) is too slow 
            #         # if done naively in Python.
            #         
            #         # Let's optimize: 
            #         # 1. Identify the set of lost words.
            #         # 2. Any valid replacement MUST be in the intersection of balls of all lost words.
            #         # 3. Compute this intersection.
            
            min_cov = np.min(covered[ball_to_remove])
            if min_cov > 1:
                covered[ball_to_remove] -= 1
                code_indices.pop(i)
                improved = True
                break
            else:
                lost_mask = (covered[ball_to_remove] == 1)
                lost_indices = ball_to_remove[lost_mask]
                
                if len(lost_indices) == 0:
                    continue
                    
                # Find candidates: words whose ball contains ALL lost_indices
                # We can iterate through all words and check, but that's slow.
                # Instead, we can use the fact that the replacement word must be close to the lost words.
                # Let's pick one lost word L. The replacement R must be in ball(L).
                # So R is one of the 15 words in ball(lost_indices[0]).
                # We check these 15 candidates.
                
                candidate_indices = ball_indices_np[lost_indices[0]]
                found_swap = False
                for cand_idx in candidate_indices:
                    # Check if cand_idx covers all lost_indices
                    cand_ball = ball_indices_np[cand_idx]
                    # Check if all lost_indices are in cand_ball
                    # Since cand_ball is an array, we can use set intersection or binary search
                    # But cand_ball is small (15). lost_indices is small.
                    # Convert cand_ball to set for O(1) lookup? Or just check.
                    # For speed, let's use a boolean array or set.
                    # Since we do this many times, creating a set each time is expensive.
                    # Alternative: 
                    # All lost indices must be in cand_ball.
                    # We can check: np.isin(lost_indices, cand_ball).all()
                    if np.isin(lost_indices, cand_ball).all():
                        # Valid candidate
                        # Check if it improves the code size or just maintains?
                        # If we swap, size remains same. We only care about size reduction.
                        # Wait, the goal is to REDUCE size.
                        # A swap keeps size same. It might enable future removals.
                        # However, pure removal is the primary driver for score improvement.
                        # Swaps can help escape local minima where no single word is redundant,
                        # but swapping allows a different word to become redundant.
                        
                        # Let's try the swap.
                        # Remove idx_to_remove
                        covered[ball_to_remove] -= 1
                        # Add cand_idx
                        # But wait, if cand_idx is already in the code, we just moved it?
                        # No, we are replacing idx_to_remove with cand_idx.
                        # If cand_idx is already in the code, we have a duplicate? No, we remove one, add another.
                        # If cand_idx == idx_to_remove, no change.
                        if cand_idx == idx_to_remove:
                            # Revert
                            covered[ball_to_remove] += 1
                            continue
                        
                        # Check if cand_idx is already in code_indices
                        # If it is, we are effectively just moving the coverage?
                        # No, we are removing one instance and adding another.
                        # If cand_idx is in code_indices, we are just changing which word is at position i?
                        # Actually, if cand_idx is already in the code, adding it increases its count to 2?
                        # No, the code is a set of distinct words.
                        # If cand_idx is already in the code, we cannot add it again.
                        # We must ensure distinctness.
                        # So, only swap if cand_idx is NOT in code_indices.
                        
                        # Efficient check for membership:
                        # code_indices is a list.
                        # If cand_idx in code_indices:
                        #     continue
                        
                        # To make this faster, maintain a set of current code indices.
                        # Let's refactor to use a set for membership.
                        
                        # For now, let's just do the swap if it's not present.
                        # We'll maintain a set `code_set` outside the loop.
                        
                        # Since I didn't define code_set yet, let's add it.
                        # But I'm inside the loop logic. Let's restructure slightly in the final code.
                        
                        # For this draft, let's assume we check membership.
                        # If cand_idx is not in code:
                        #    covered[ball_to_remove] -= 1
                        #    covered[ball_indices_np[cand_idx]] += 1
                        #    code_indices[i] = cand_idx
                        #    improved = True
                        #    break
                        # else:
                        #    # Revert
                        #    covered[ball_to_remove] += 1
                        #    pass
                        
                        # Let's implement the set logic properly in the final code.
                        # For now, let's just attempt the swap if it looks promising.
                        # We'll use a helper to check membership.
                        
                        # Optimization: Only accept swaps if they don't increase the "minimum coverage" elsewhere?
                        # No, just keep size same.
                        
                        # Let's implement the swap.
                        # We need to know if cand_idx is in code_indices.
                        # Let's use a set `current_code_set` that we update.
                        
                        # I will add `current_code_set` to the scope.
                        
                        # (Self-correction: I will write the final code with the set)
                        
                        # Let's just break out and handle it in the final code structure.
                        pass

    # The logic above was getting complex with state management. 
    # Let's rewrite the local search part cleanly with a set for membership.
    
    # --- Reset and Clean Local Search ---
    # Re-initialize state from initial_code to be safe
    code_indices = [word_index[w] for w in initial_code]
    current_code_set = set(code_indices)
    covered = np.zeros(total_words, dtype=np.int32)
    for idx in code_indices:
        covered[ball_indices_np[idx]] += 1
        
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Iterate in random order
        indices_to_check = code_indices.copy()
        random.shuffle(indices_to_check)
        
        for idx_to_remove in indices_to_check:
            if time.time() > deadline - 5:
                break
            if idx_to_remove not in current_code_set:
                continue # Might have been removed in a previous iteration of this pass? 
                         # Actually, if we remove it, we break. If we swap, it's still in set.
                         # If we just check all, and remove one, the list `code_indices` changes.
                         # So iterating over a copy is safer, but we must check if it's still there.
            
            ball_to_remove = ball_indices_np[idx_to_remove]
            min_cov = np.min(covered[ball_to_remove])
            
            if min_cov > 1:
                # Redundant
                covered[ball_to_remove] -= 1
                # Remove from list and set
                code_indices.remove(idx_to_remove)
                current_code_set.remove(idx_to_remove)
                improved = True
                break # Restart loop with new state
                
            else:
                # Not redundant. Try to swap.
                lost_mask = (covered[ball_to_remove] == 1)
                lost_indices = ball_to_remove[lost_mask]
                
                if len(lost_indices) == 0:
                    continue
                    
                # Candidates must be in ball(lost_indices[0])
                candidate_indices = ball_indices_np[lost_indices[0]]
                
                for cand_idx in candidate_indices:
                    if cand_idx == idx_to_remove:
                        continue
                    if cand_idx in current_code_set:
                        continue # Cannot add duplicate
                    
                    # Check if cand_idx covers all lost_indices
                    cand_ball = ball_indices_np[cand_idx]
                    if np.isin(lost_indices, cand_ball).all():
                        # Valid swap
                        # Remove idx_to_remove
                        covered[ball_to_remove] -= 1
                        current_code_set.remove(idx_to_remove)
                        
                        # Add cand_idx
                        covered[ball_indices_np[cand_idx]] += 1
                        current_code_set.add(cand_idx)
                        
                        # Update code_indices list
                        # Find position of idx_to_remove and replace
                        pos = code_indices.index(idx_to_remove)
                        code_indices[pos] = cand_idx
                        
                        improved = True
                        break # Restart loop with new state
                
                if improved:
                    break

    return [list(all_words[idx]) for idx in code_indices]
