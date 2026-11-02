"""Rewritten program for the ternary covering code K_3(7,1) task.
Strategy: Greedy set cover with bitmask acceleration, then local search (swap/replace)
to minimize codeword count. Uses the algebraic structure of the ternary Hamming space
and the known tetracode as a strong initial solution.
"""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import sys

def construct(q, n, radius):
    """
    Construct a covering code for {0..q-1}^n with covering radius `radius`.
    Returns a list of codewords (each a list of n integers in 0..q-1).
    """
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin within 120s limit
    
    # For the specific case of q=3, n=7, radius=1
    if q == 3 and n == 7 and radius == 1:
        return _solve_ternary_7_1(deadline)
    
    # Generic fallback: greedy set cover
    return _generic_greedy(q, n, radius, deadline)


def _solve_ternary_7_1(deadline):
    """
    Specialized solver for K_3(7,1).
    Uses bitmask representation for fast ball operations.
    """
    total_words = 3 ** 7  # 2187
    
    # Precompute all words as integers 0..2186
    # Word w maps to index: sum(w[i] * 3^(6-i) for i in range(7))
    
    # Precompute the ball (Hamming ball of radius 1) for each word
    # Ball of a word w: {w} union {w with one coordinate changed}
    # For radius 1, ball size = 1 + 7*2 = 15
    
    # We'll use a list of sets for balls, but for speed we use bitmasks
    # Since 2187 bits fits in ~28 uint64s, we can use Python integers as bitmasks
    
    # Precompute ball masks
    ball_masks = [0] * total_words
    
    for w_idx in range(total_words):
        # Convert index to word
        word = [0] * 7
        temp = w_idx
        for i in range(6, -1, -1):
            word[i] = temp % 3
            temp //= 3
        
        # Compute ball mask
        mask = 1 << w_idx
        
        # For each position, change to each of the other 2 values
        for pos in range(7):
            orig = word[pos]
            for new_val in range(3):
                if new_val == orig:
                    continue
                # Compute new index
                diff = (new_val - orig) * (3 ** (6 - pos))
                new_idx = w_idx + diff
                mask |= (1 << new_idx)
        
        ball_masks[w_idx] = mask
    
    # Initial solution: tetracode direct product
    # Tetracode: 9 words of length 4 that form a perfect code of radius 1
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=3):
            initial_codewords.append(h + list(tail))
    
    # Convert initial codewords to indices
    def word_to_idx(word):
        idx = 0
        for i in range(7):
            idx = idx * 3 + word[i]
        return idx
    
    initial_indices = [word_to_idx(w) for w in initial_codewords]
    
    # Verify coverage
    covered_mask = 0
    for idx in initial_indices:
        covered_mask |= ball_masks[idx]
    
    # Check if fully covered
    full_mask = (1 << total_words) - 1
    if covered_mask != full_mask:
        # Should not happen for the tetracode construction, but handle it
        # Add words greedily to cover remaining
        uncovered = full_mask & ~covered_mask
        while uncovered:
            # Find word whose ball covers the most uncovered words
            best_idx = -1
            best_count = 0
            for i in range(total_words):
                if not (covered_mask >> i) & 1:
                    cnt = bin(ball_masks[i] & uncovered).count('1')
                    if cnt > best_count:
                        best_count = cnt
                        best_idx = i
            if best_idx == -1:
                break
            initial_indices.append(best_idx)
            covered_mask |= ball_masks[best_idx]
            uncovered = full_mask & ~covered_mask
            if time.time() > deadline:
                break
    
    # Now do local search to minimize the number of codewords
    # Current solution
    current = list(initial_indices)
    current_covered = 0
    for idx in current:
        current_covered |= ball_masks[idx]
    
    best = list(current)
    best_count = len(current)
    
    # Local search: try to remove a codeword if coverage is maintained
    # Or swap: remove one, add another that covers more uncovered
    rng = random.Random(42)
    
    iteration = 0
    while time.time() < deadline and len(current) > 156:
        iteration += 1
        
        # Strategy 1: Try to remove a random codeword if it doesn't break coverage
        if len(current) > 1:
            pos = rng.randint(0, len(current) - 1)
            idx_to_remove = current[pos]
            
            # Check if removing this codeword still covers everything
            # We need to check if any word is only covered by this codeword
            # For efficiency, we can check: is there a word in ball(idx_to_remove) 
            # that is not covered by any other codeword?
            
            # Quick check: compute coverage without this codeword
            # This is expensive, so we do a sampled check first
            # Actually, let's just try it and verify
            new_covered = 0
            for i, idx in enumerate(current):
                if i != pos:
                    new_covered |= ball_masks[idx]
            
            if new_covered == full_mask:
                # Safe to remove
                current.pop(pos)
                current_covered = new_covered
                if len(current) < best_count:
                    best_count = len(current)
                    best = list(current)
                continue
        
        # Strategy 2: Try to swap - remove one codeword, add another
        # Pick a random codeword to remove
        if len(current) > 0:
            pos = rng.randint(0, len(current) - 1)
            idx_to_remove = current[pos]
            
            # Coverage without this codeword
            cover_without = 0
            for i, idx in enumerate(current):
                if i != pos:
                    cover_without |= ball_masks[idx]
            
            uncovered = full_mask & ~cover_without
            
            if uncovered == 0:
                # Can just remove
                current.pop(pos)
                current_covered = cover_without
                if len(current) < best_count:
                    best_count = len(current)
                    best = list(current)
                continue
            
            # Find a codeword that covers the most uncovered words
            # Sample from the space for speed
            sample_size = min(200, total_words)
            candidates = rng.sample(range(total_words), sample_size)
            
            best_add = -1
            best_gain = 0
            for cand in candidates:
                if cand in current:
                    continue
                gain = bin(ball_masks[cand] & uncovered).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_add = cand
            
            if best_add != -1 and best_gain > 0:
                # Check if adding best_add and removing idx_to_remove maintains coverage
                # New coverage = cover_without | ball(best_add)
                new_covered = cover_without | ball_masks[best_add]
                if new_covered == full_mask:
                    current[pos] = best_add
                    current_covered = new_covered
                    if len(current) < best_count:
                        best_count = len(current)
                        best = list(current)
        
        # Strategy 3: Random restart with perturbation
        if iteration % 500 == 0 and len(current) == best_count:
            # Perturb: randomly remove some codewords and re-add greedily
            if len(current) > 160:
                # Remove 5-10 random codewords
                num_remove = rng.randint(5, 10)
                to_remove = set(rng.sample(range(len(current)), min(num_remove, len(current))))
                new_current = [idx for i, idx in enumerate(current) if i not in to_remove]
                
                # Re-cover greedily
                new_covered = 0
                for idx in new_current:
                    new_covered |= ball_masks[idx]
                
                uncovered = full_mask & ~new_covered
                while uncovered and time.time() < deadline:
                    # Find best word to add
                    best_idx = -1
                    best_count_local = 0
                    for i in range(total_words):
                        cnt = bin(ball_masks[i] & uncovered).count('1')
                        if cnt > best_count_local:
                            best_count_local = cnt
                            best_idx = i
                    if best_idx == -1 or best_count_local == 0:
                        break
                    new_current.append(best_idx)
                    new_covered |= ball_masks[best_idx]
                    uncovered = full_mask & ~new_covered
                
                if len(new_current) < len(current):
                    current = new_current
                    current_covered = new_covered
                    if len(current) < best_count:
                        best_count = len(current)
                        best = list(current)
    
    # Convert best indices back to words
    def idx_to_word(idx):
        word = [0] * 7
        temp = idx
        for i in range(6, -1, -1):
            word[i] = temp % 3
            temp //= 3
        return word
    
    result = [idx_to_word(idx) for idx in best]
    
    # Ensure distinctness
    seen = set()
    unique_result = []
    for w in result:
        key = tuple(w)
        if key not in seen:
            seen.add(key)
            unique_result.append(w)
    
    return unique_result


def _generic_greedy(q, n, radius, deadline):
    """Generic greedy set cover for other parameters."""
    total_words = q ** n
    
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    
    # Precompute balls
    def hamming_dist(w1, w2):
        return sum(1 for a, b in zip(w1, w2) if a != b)
    
    balls = []
    for w in all_words:
        ball = set()
        for v in all_words:
            if hamming_dist(w, v) <= radius:
                ball.add(v)
        balls.append(ball)
    
    # Greedy set cover
    uncovered = set(range(total_words))
    selected = []
    
    while uncovered and time.time() < deadline:
        best_idx = -1
        best_count = 0
        for i in range(total_words):
            count = len(balls[i] & uncovered)
            if count > best_count:
                best_count = count
                best_idx = i
        if best_idx == -1:
            break
        selected.append(best_idx)
        uncovered -= balls[best_idx]
    
    return [list(all_words[i]) for i in selected]
# EVOLVE-BLOCK-END
