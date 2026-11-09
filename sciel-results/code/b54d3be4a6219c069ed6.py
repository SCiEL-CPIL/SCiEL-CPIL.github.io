"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
from collections import defaultdict
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code for {0,1,2}^n with covering radius 1.
    Uses a local search approach starting from the 486-word direct sum,
    focusing on swapping codewords to increase redundancy and enabling removals.
    """
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters
        if n <= 4:
            if n == 1:
                return [[i] for i in range(q)]
            elif n == 2:
                return [list(w) for w in itertools.product(range(q), repeat=2)]
            elif n == 4:
                tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
                return tetracode
            else:
                tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    # Initialize with the 486-word direct sum construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = [tuple(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=4)]
    
    # Precompute balls for all possible words in the space
    # Space size is 3^8 = 6561
    # Ball size is 1 + 8*2 = 17
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    word_index = {w: i for i, w in enumerate(all_words)}
    
    # balls[i] is a list of indices of words covered by word i
    balls = []
    for w in all_words:
        covered_indices = [word_index[w]]
        for i in range(n):
            for val in range(q):
                if val != w[i]:
                    new_word = list(w)
                    new_word[i] = val
                    covered_indices.append(word_index[tuple(new_word)])
        balls.append(covered_indices)
        
    # State variables
    codewords = set(initial_codewords)
    coverage_count = np.zeros(len(all_words), dtype=np.int32)
    
    # Initialize coverage
    for c in codewords:
        idx = word_index[c]
        for w_idx in balls[idx]:
            coverage_count[w_idx] += 1
            
    def try_remove_redundant():
        """Attempt to remove codewords that are not uniquely covering any word."""
        removed = 0
        # Iterate until no more removals
        while True:
            found = False
            # Shuffle to randomize order
            cw_list = list(codewords)
            random.shuffle(cw_list)
            for c in cw_list:
                if c not in codewords:
                    continue
                idx = word_index[c]
                can_remove = True
                for w_idx in balls[idx]:
                    if coverage_count[w_idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    # Remove c
                    for w_idx in balls[idx]:
                        coverage_count[w_idx] -= 1
                    codewords.remove(c)
                    removed += 1
                    found = True
            if not found:
                break
        return removed

    def find_critical_words():
        """Return list of word indices with coverage count 1."""
        return np.where(coverage_count == 1)[0].tolist()

    # Main loop
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    # Initial removals
    try_remove_redundant()
    
    while time.time() - start_time < time_limit:
        criticals = find_critical_words()
        
        if not criticals:
            # No critical words, try to remove more
            try_remove_redundant()
            # If we can't remove, we are stuck in a local minimum with respect to removals.
            # We need to perturb by swapping to create redundancy.
            # But if there are no critical words, any swap is "safe" in terms of not breaking coverage?
            # No, a swap might create critical words.
            # If we have no critical words, we can try to remove codewords.
            # If we can't remove, we are in a stable state.
            # We can try to swap to find a state where removal is possible.
            # However, if coverage_count >= 2 everywhere, we are in a "2-covering".
            # We can try to swap to increase average coverage or just shuffle.
            pass
            
        if time.time() - start_time > time_limit:
            break
            
        # If we have critical words, try to swap
        if criticals:
            # Pick a random critical word
            target_w_idx = random.choice(criticals)
            target_word = all_words[target_w_idx]
            
            # Find the unique codeword covering this word
            covering_cw = None
            for c in codewords:
                if target_w_idx in balls[word_index[c]]:
                    covering_cw = c
                    break
                    
            if covering_cw is None:
                # Should not happen if coverage is valid
                continue
                
            old_idx = word_index[covering_cw]
            
            # Generate candidates for new codeword
            # Candidates must be within distance 1 of target_word
            # So they are in balls[target_w_idx]
            candidates = balls[target_w_idx]
            
            best_swap = None
            best_score = -1
            
            for new_idx in candidates:
                new_word = all_words[new_idx]
                
                # Candidate must not be the old codeword
                if new_word == covering_cw:
                    continue
                    
                # Candidate must not already be in codewords
                if new_word in codewords:
                    continue
                    
                # Simulate swap: Remove old, Add new
                # Check validity:
                # For all w in Ball(old) \ Ball(new), coverage_count[w] must be >= 2
                # (so that after removing old, count >= 1)
                
                old_ball_set = set(balls[old_idx])
                new_ball_set = set(balls[new_idx])
                
                # Words that lose coverage
                lost_words = old_ball_set - new_ball_set
                # Words that gain coverage
                gained_words = new_ball_set - old_ball_set
                
                valid = True
                new_critical_count = 0
                
                # Check lost words
                for w_idx in lost_words:
                    if coverage_count[w_idx] < 2:
                        valid = False
                        break
                        
                if not valid:
                    continue
                
                # Calculate impact on critical words
                # Existing critical words in lost_words: they become non-critical (count drops from 1 to 0? No, valid implies count>=2, so 2->1).
                # Wait, if count was 1, valid would be False. So lost_words have count >= 2.
                # After removal, count becomes >= 1.
                # If count was 2, it becomes 1 -> New Critical Word.
                # If count was > 2, it becomes > 1 -> Not Critical.
                
                # Existing critical words in gained_words:
                # They are covered by some other codeword (count=1).
                # After adding new, count becomes 2 -> Not Critical.
                # So we "fix" critical words in gained_words.
                
                # New critical words created:
                # 1. From lost_words: words that had count 2.
                # 2. From gained_words: words that had count 0? No, valid implies covered.
                #    Wait, gained_words are in Ball(new). They might have count 0 in current state?
                #    No, the code covers everything. So count >= 1.
                #    If count was 1, it becomes 2 (Fixed).
                #    If count was > 1, it becomes > 2 (No change in critical status).
                
                # So:
                # Fixed criticals: |{w in gained_words : coverage_count[w] == 1}|
                # Created criticals: |{w in lost_words : coverage_count[w] == 2}|
                
                fixed = 0
                for w_idx in gained_words:
                    if coverage_count[w_idx] == 1:
                        fixed += 1
                        
                created = 0
                for w_idx in lost_words:
                    if coverage_count[w_idx] == 2:
                        created += 1
                        
                # We want to maximize fixed - created
                # If fixed > created, we reduce total critical words.
                # If fixed == created and created == 0, we maintain status but might help future removals?
                # Let's score: fixed * 10 - created
                score = fixed * 10 - created
                
                if score > best_score:
                    best_score = score
                    best_swap = (old_idx, new_idx, lost_words, gained_words, fixed, created)
                    
            if best_swap is not None and best_score > 0:
                old_idx, new_idx, lost_words, gained_words, fixed, created = best_swap
                new_word = all_words[new_idx]
                
                # Apply swap
                for w_idx in lost_words:
                    coverage_count[w_idx] -= 1
                for w_idx in gained_words:
                    coverage_count[w_idx] += 1
                    
                codewords.remove(all_words[old_idx])
                codewords.add(new_word)
                
                # After a few swaps, try to remove
                # Heuristic: try removal every 10 swaps or if we fixed many
                if random.random() < 0.2:
                    try_remove_redundant()
            else:
                # No good swap found for this critical word
                # Try a different critical word or break
                pass
        else:
            # No critical words
            # Try to remove
            try_remove_redundant()
            # If we can't remove, we are stuck.
            # We can try to swap to create a "loose" spot.
            # Pick a random codeword, try to swap it with a neighbor to increase redundancy.
            # This is essentially the same as the critical word logic but without the constraint.
            # If we have no critical words, any valid swap is good if it increases min coverage.
            pass

    # Final verification
    final_codewords = list(codewords)
    
    # Verify coverage
    covered_all = set()
    for c in final_codewords:
        idx = word_index[c]
        for w_idx in balls[idx]:
            covered_all.add(w_idx)
            
    if len(covered_all) < len(all_words):
        # Fallback
        return [list(c) for c in initial_codewords]
        
    return [list(c) for c in final_codewords]

# EVOLVE-BLOCK-END
