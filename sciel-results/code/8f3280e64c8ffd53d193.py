import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # 110s to be safe
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters, though task specifies this one
        # Just return a trivial cover (all words) if parameters differ significantly
        # But task says "evaluator passes this task's alphabet size..."
        # So we assume q=3, n=7, radius=1.
        # If not, we might fail. Let's stick to the specific case.
        pass

    total_words = q ** n
    # Map index to word
    # We need to generate ball masks.
    
    # Precompute neighbors for each index
    # Index is base-q representation.
    
    # Helper to get digit at pos
    def get_digit(idx, pos):
        return (idx // (q ** pos)) % q

    # Precompute ball masks
    # ball_mask[i] is a bitmask of size total_words
    # total_words = 2187, so we need integers with 2187 bits.
    # Python integers are arbitrary precision, so this is fine.
    
    ball_masks = [0] * total_words
    for i in range(total_words):
        mask = 1 << i
        # For each position
        for pos in range(n):
            current_digit = get_digit(i, pos)
            for new_digit in range(q):
                if new_digit == current_digit:
                    continue
                # Calculate new index
                # i + (new_digit - current_digit) * (q ** pos)
                diff = new_digit - current_digit
                new_idx = i + diff * (q ** pos)
                mask |= (1 << new_idx)
        ball_masks[i] = mask

    full_mask = (1 << total_words) - 1
    
    best_code = None
    best_size = float('inf')
    
    # Randomized Greedy
    # To add diversity, we can shuffle the order of candidates when counts are tied,
    # or just rely on the fact that if we always pick the first max, it's deterministic.
    # We will pick a random one from the set of max-count candidates.
    
    while time.time() < deadline:
        uncovered = full_mask
        code = []
        
        while uncovered:
            max_count = -1
            best_candidates = []
            
            # Find max count
            # Optimization: If we have many candidates, this loop is expensive.
            # But 2187 is small.
            for c in range(total_words):
                count = (ball_masks[c] & uncovered).bit_count()
                if count > max_count:
                    max_count = count
                    best_candidates = [c]
                elif count == max_count:
                    best_candidates.append(c)
            
            # Pick one randomly
            if best_candidates:
                chosen = random.choice(best_candidates)
            else:
                # Should not happen
                break
                
            code.append(chosen)
            uncovered &= ~ball_masks[chosen]
            
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
            # If we hit the theoretical lower bound or known best, we might stop?
            # Known best is 186. Lower bound 156.
            # If we find 186, it's likely optimal or near.
            # We can continue to see if we find 185, but it's rare.
            # Let's continue for a bit to ensure robustness, but maybe break if we hit 186?
            # Actually, finding 186 is great. Finding 185 is better.
            # Let's not break early unless we are very close to deadline.
            
    # Convert indices to words
    if best_code is None:
        # Fallback: return empty or trivial
        return []
        
    result = []
    for idx in best_code:
        word = []
        temp = idx
        for _ in range(n):
            word.append(temp % q)
            temp //= q
        word.reverse()
        result.append(word)
        
    return result
