"""Greedy Set Cover with Kicks for the ternary covering code K_3(7,1)."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

_DEADLINE = 145.0  # safety margin before 150s wall-clock


def _precompute_balls(q, n):
    """Precompute the indices of words covered by each word in the space."""
    total = q ** n
    balls = []
    for idx in range(total):
        # Convert index to base-q digits
        word = []
        t = idx
        for _ in range(n):
            word.append(t % q)
            t //= q
        word.reverse()
        
        # Generate all words within Hamming distance 1
        covered = set()
        covered.add(idx)
        for i in range(n):
            original = word[i]
            for new_val in range(q):
                if new_val == original:
                    continue
                new_word = word.copy()
                new_word[i] = new_val
                # Convert back to index
                new_idx = 0
                for d in new_word:
                    new_idx = new_idx * q + d
                covered.add(new_idx)
        balls.append(np.array(list(covered), dtype=np.int32))
    return balls


def construct(q, n, radius):
    global _DEADLINE
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)
    
    total = q ** n
    balls = _precompute_balls(q, n)
    
    # State variables
    # cover_count[i] is the number of codewords covering word i
    # uncovered[i] is True if cover_count[i] == 0
    cover_count = np.zeros(total, dtype=np.int32)
    uncovered = np.ones(total, dtype=bool)
    
    codewords = []
    
    def add_word(idx):
        nonlocal cover_count, uncovered
        idxs = balls[idx]
        cover_count[idxs] += 1
        # If count was 0, it is now 1, so it is covered
        # If count was >0, it is still covered
        # So uncovered becomes False for all idxs
        uncovered[idxs] = False

    def remove_word(idx):
        nonlocal cover_count, uncovered
        idxs = balls[idx]
        cover_count[idxs] -= 1
        # If count becomes 0, it is now uncovered
        newly_uncovered = (cover_count[idxs] == 0)
        uncovered[idxs] = uncovered[idxs] | newly_uncovered

    def greedy_construct():
        nonlocal codewords, cover_count, uncovered
        while np.any(uncovered):
            # Find the word that covers the most uncovered words
            # scores[w] = sum(uncovered[balls[w]])
            best_score = -1
            best_idx = -1
            # Optimization: only check words that might be good?
            # For n=7, 2187 is small.
            for w in range(total):
                # If w is already in codewords, we might skip? 
                # No, we can add duplicates? No, distinct codewords required.
                # But in greedy, we usually add distinct ones.
                # If w is already added, its ball is already covered, so score is 0?
                # Not necessarily, if we removed some.
                # But in pure greedy construction, we start empty.
                
                # Fast check: if all uncovered are covered by current set, we stop.
                # Here we pick the best next one.
                
                # Using numpy sum is fast enough
                score = np.sum(uncovered[balls[w]])
                if score > best_score:
                    best_score = score
                    best_idx = w
            
            if best_idx == -1:
                break # Should not happen if uncovered is True
                
            add_word(best_idx)
            codewords.append(best_idx)
            
            # If we just covered everything, stop
            if not np.any(uncovered):
                break

    def try_remove_redundant():
        nonlocal codewords, cover_count, uncovered
        removed_any = True
        while removed_any:
            removed_any = False
            # Iterate in reverse to allow popping
            for i in range(len(codewords) - 1, -1, -1):
                if time.time() > _DEADLINE:
                    return
                idx = codewords[i]
                # Check if removing idx leaves any word uncovered
                # We need to see if any word in balls[idx] has cover_count == 1
                # If yes, removing it makes count 0 -> uncovered.
                # If no (all counts >= 2), we can remove.
                
                counts = cover_count[balls[idx]]
                if np.all(counts >= 2):
                    remove_word(idx)
                    codewords.pop(i)
                    removed_any = True
                    # Restart loop or continue? Continue is fine, but indices shift.
                    # Since we are iterating backwards, popping i is safe for j > i.
                    # But we might have missed some at j < i.
                    # Restarting is safer and cheap.
                    break

    # 1. Initial Greedy Construction
    greedy_construct()
    try_remove_redundant()
    
    best_codewords = list(codewords)
    best_len = len(codewords)
    
    # 2. Kick Local Search
    # Strategy: Randomly remove K codewords, then re-greedy.
    # Keep the best solution found.
    
    rng = np.random.default_rng(42)
    
    # Save current state to allow rollback if kick fails?
    # Or just keep the best and restart from best?
    # Restarting from best is better for exploration.
    
    current_codewords = list(codewords)
    current_cover_count = cover_count.copy()
    current_uncovered = uncovered.copy()
    
    while time.time() < _DEADLINE - 5.0:
        # Save current state
        orig_codewords = list(current_codewords)
        orig_cover_count = current_cover_count.copy()
        orig_uncovered = current_uncovered.copy()
        
        # Kick: Remove K random codewords
        K = 3 + int(rng.integers(0, 5)) # 3 to 7
        if len(current_codewords) > K:
            # Pick K indices to remove
            remove_indices = rng.choice(len(current_codewords), size=K, replace=False)
            # Remove them
            # We need to update state
            # We must be careful with indices if we pop.
            # Better: mark them for removal, then rebuild?
            # Or just pop in reverse order.
            
            # Since we have state arrays, we can just call remove_word
            # But we need to remove them from the list too.
            
            # Sort indices in descending order to pop safely
            remove_indices_sorted = sorted(remove_indices, reverse=True)
            
            for idx_pos in remove_indices_sorted:
                w = current_codewords[idx_pos]
                # Update state
                # We are modifying current_cover_count and current_uncovered
                idxs = balls[w]
                current_cover_count[idxs] -= 1
                newly_uncovered = (current_cover_count[idxs] == 0)
                current_uncovered[idxs] = current_uncovered[idxs] | newly_uncovered
                
                current_codewords.pop(idx_pos)
                
        # Re-greedy
        # We need a version of greedy that uses current state
        while np.any(current_uncovered):
            best_score = -1
            best_idx = -1
            for w in range(total):
                # Skip words already in current_codewords?
                # If a word is already in the set, adding it again is not allowed (distinct).
                # But its ball is already covered (counts >= 1).
                # However, if we removed some, some words in its ball might be uncovered?
                # No, if w is in current_codewords, its ball has count >= 1 for all words in ball?
                # Wait. If w is in the set, it contributes 1 to all words in ball(w).
                # So for any x in ball(w), cover_count[x] >= 1.
                # So uncovered[x] is False for all x in ball(w).
                # So score for w is 0.
                # So we don't need to explicitly skip, they will have score 0.
                
                score = np.sum(current_uncovered[balls[w]])
                if score > best_score:
                    best_score = score
                    best_idx = w
            
            if best_idx == -1:
                break # No word covers any uncovered? Impossible if uncovered exists.
                
            # Add best_idx
            idxs = balls[best_idx]
            current_cover_count[idxs] += 1
            current_uncovered[idxs] = False
            current_codewords.append(best_idx)
            
            if not np.any(current_uncovered):
                break
        
        # Try to remove redundant
        # We need a local version of try_remove_redundant
        removed_any = True
        while removed_any and time.time() < _DEADLINE - 5.0:
            removed_any = False
            for i in range(len(current_codewords) - 1, -1, -1):
                idx = current_codewords[i]
                counts = current_cover_count[balls[idx]]
                if np.all(counts >= 2):
                    # Remove
                    idxs = balls[idx]
                    current_cover_count[idxs] -= 1
                    newly_uncovered = (current_cover_count[idxs] == 0)
                    current_uncovered[idxs] = current_uncovered[idxs] | newly_uncovered
                    current_codewords.pop(i)
                    removed_any = True
                    break
        
        # Check if better
        if len(current_codewords) < best_len:
            best_len = len(current_codewords)
            best_codewords = list(current_codewords)
            # Update global state to best
            cover_count = current_cover_count.copy()
            uncovered = current_uncovered.copy()
            codewords = list(current_codewords)
            
            # If we found a very good one, maybe restart from here?
            # The loop already continues.
            
        # If we are stuck at best_len for a while, we might want to increase K or change strategy.
        # For now, just continue kicking.
        
        # Optional: If no improvement in last 10 kicks, increase K?
        # Keep it simple.

    # Final check and return
    # Ensure distinctness (greedy adds distinct? Yes, because score 0 for existing).
    # But what if two different words have the same ball? No, balls are unique to the center in this geometry?
    # Actually, different centers can have overlapping balls.
    # But we add `best_idx`. If `best_idx` is already in `codewords`, its score is 0.
    # So we never add duplicates.
    
    # Verify validity of best_codewords
    # Re-compute coverage from scratch to be safe
    final_cov = np.zeros(total, dtype=np.int32)
    for w in best_codewords:
        final_cov[balls[w]] += 1
        
    if np.all(final_cov >= 1):
        return [list((w // (q ** (n - 1 - i))) % q for i in range(n)) for w in best_codewords]
    else:
        # Fallback: Return the initial greedy result or tetracode
        # We should return a valid code.
        # Let's re-run greedy from scratch if this happens (shouldn't).
        pass

    # Fallback to tetracode if something went wrong
    def _tetracode_words():
        return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]

    tetracode = _tetracode_words()
    rest = n - 4
    words = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=rest):
            words.append(h + list(tail))
    return words
# EVOLVE-BLOCK-END
