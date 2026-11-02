import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    if n < 4:
        return [list(t) for t in itertools.product(range(q), repeat=n)]
    
    total_words = q ** n
    idx_to_word = []
    word_to_idx = {}
    idx = 0
    for w in itertools.product(range(q), repeat=n):
        word_to_idx[w] = idx
        idx_to_word.append(w)
        idx += 1
    
    # Precompute ball (sphere) for each word as a bitmask
    # For n=7, q=3, total_words=2187, we can use Python's big integers
    ball_masks = []
    for w_idx in range(total_words):
        w = idx_to_word[w_idx]
        mask = 1 << w_idx
        if radius >= 1:
            for i in range(n):
                for val in range(q):
                    if val == w[i]:
                        continue
                    new_w = list(w)
                    new_w[i] = val
                    mask |= 1 << word_to_idx[tuple(new_w)]
        if radius >= 2:
            for i in range(n):
                for j in range(i+1, n):
                    for vi in range(q):
                        if vi == w[i]:
                            continue
                        for vj in range(q):
                            if vj == w[j]:
                                continue
                            new_w = list(w)
                            new_w[i] = vi
                            new_w[j] = vj
                            mask |= 1 << word_to_idx[tuple(new_w)]
        ball_masks.append(mask)
    
    # Greedy initialization: start with empty code, add words that cover most uncovered space
    covered_mask = 0
    code = []
    
    # Precompute list of all word indices for random sampling
    all_indices = list(range(total_words))
    
    # First pass: greedy by coverage
    remaining = all_indices[:]
    while remaining:
        if time.time() > deadline - 60:
            break
        best_idx = -1
        best_new = 0
        # Sample a subset for speed if too many remaining
        if len(remaining) > 200:
            sample = random.sample(remaining, 200)
        else:
            sample = remaining
            
        for c_idx in sample:
            new_cov = (ball_masks[c_idx] & ~covered_mask).bit_count()
            if new_cov > best_new:
                best_new = new_cov
                best_idx = c_idx
        
        if best_idx == -1 or best_new == 0:
            break
            
        code.append(best_idx)
        covered_mask |= ball_masks[best_idx]
        remaining.remove(best_idx)
        
        if covered_mask == (1 << total_words) - 1:
            break
    
    # If greedy didn't cover everything, add random words until covered
    if covered_mask != (1 << total_words) - 1:
        random.shuffle(all_indices)
        for c_idx in all_indices:
            if c_idx in code:
                continue
            code.append(c_idx)
            covered_mask |= ball_masks[c_idx]
            if covered_mask == (1 << total_words) - 1:
                break
    
    best_code = code[:]
    best_size = len(code)
    
    # Local search: deletion and swap moves
    # Use bitmask for fast coverage check
    def get_covered_mask(code_list):
        m = 0
        for c in code_list:
            m |= ball_masks[c]
        return m
    
    # Check if a codeword can be removed (all its ball neighbors are still covered)
    def can_remove(c_idx, code_list):
        # Check if ball(c_idx) is subset of covered by other codewords
        others_mask = 0
        for c in code_list:
            if c != c_idx:
                others_mask |= ball_masks[c]
        return (ball_masks[c_idx] & ~others_mask) == 0
    
    # Aggressive local search
    while time.time() < deadline - 5:
        improved = False
        
        # Try deletions
        for i in range(len(code)):
            if time.time() > deadline - 5:
                break
            c_idx = code[i]
            if can_remove(c_idx, code):
                code.pop(i)
                improved = True
                if len(code) < best_size:
                    best_code = code[:]
                    best_size = len(code)
                break  # Restart deletion scan after a removal
        
        if not improved:
            # Try swap moves: remove one, add another
            for i in range(len(code)):
                if time.time() > deadline - 5:
                    break
                c_remove = code[i]
                
                # Try several random candidates to add
                for _ in range(50):
                    if time.time() > deadline - 5:
                        break
                    c_add = random.randint(0, total_words - 1)
                    if c_add in code:
                        continue
                    
                    # Check if adding c_add and removing c_remove still covers everything
                    # New covered = (covered - ball(c_remove) + ball(c_add))
                    # Simpler: build new code and check coverage
                    new_code = code[:i] + code[i+1:] + [c_add]
                    new_mask = 0
                    for c in new_code:
                        new_mask |= ball_masks[c]
                    
                    if new_mask == (1 << total_words) - 1:
                        code = new_code
                        improved = True
                        if len(code) < best_size:
                            best_code = code[:]
                            best_size = len(code)
                        break  # Break inner loop, restart outer scan
                
                if improved:
                    break
        
        if not improved:
            # No improvement found, try random restart with perturbation
            # Keep best code and continue
            if time.time() > deadline - 10:
                break
            # Perturb: randomly remove a few and re-add
            if len(code) > 2:
                for _ in range(min(3, len(code) - 1)):
                    if time.time() > deadline - 5:
                        break
                    idx_to_remove = random.randint(0, len(code) - 1)
                    removed = code.pop(idx_to_remove)
                    # Try to find a replacement
                    for _ in range(100):
                        if time.time() > deadline - 5:
                            break
                        c_new = random.randint(0, total_words - 1)
                        if c_new in code:
                            continue
                        code.append(c_new)
                        # Check coverage
                        m = 0
                        for c in code:
                            m |= ball_masks[c]
                        if m == (1 << total_words) - 1:
                            break
                        else:
                            code.pop()
    
    # Use best code found
    if best_code and len(best_code) <= len(code):
        code = best_code
    
    # Convert back to lists of ints
    return [list(idx_to_word[idx]) for idx in code]
