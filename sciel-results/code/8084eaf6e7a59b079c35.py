import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback
        # Simple greedy for general case
        total_words = q ** n
        full_mask = (1 << total_words) - 1
        def word_to_idx(w):
            idx = 0
            for d in w:
                idx = idx * q + d
            return idx
        def idx_to_word(idx):
            w = [0] * n
            for i in range(n - 1, -1, -1):
                w[i] = idx % q
                idx //= q
            return w
        all_indices = list(range(total_words))
        balls = [0] * total_words
        for i in all_indices:
            w = idx_to_word(i)
            mask = 1 << i
            for pos in range(n):
                for new_digit in range(q):
                    if new_digit != w[pos]:
                        nw = w[:]
                        nw[pos] = new_digit
                        j = word_to_idx(nw)
                        mask |= (1 << j)
            balls[i] = mask
        uncovered = full_mask
        code = []
        code_set = set()
        while uncovered:
            best_idx = -1
            max_covered = 0
            for c in all_indices:
                if c in code_set: continue
                covered = (balls[c] & uncovered).bit_count()
                if covered > max_covered:
                    max_covered = covered
                    best_idx = c
            if best_idx == -1: break
            code.append(best_idx)
            code_set.add(best_idx)
            uncovered &= ~balls[best_idx]
            if time.time() > deadline: break
        return [idx_to_word(c) for c in code]

    # Specific optimization for q=3, n=7, radius=1
    total_words = 2187
    full_mask = (1 << total_words) - 1
    
    def word_to_idx(w):
        idx = 0
        for d in w:
            idx = idx * 3 + d
        return idx

    def idx_to_word(idx):
        w = [0] * 7
        for i in range(6, -1, -1):
            w[i] = idx % 3
            idx //= 3
        return w

    all_indices = list(range(total_words))
    
    # Precompute balls
    balls = [0] * total_words
    for i in all_indices:
        w = idx_to_word(i)
        mask = 1 << i
        for pos in range(7):
            # Neighbors: change pos to (w[pos]+1)%3 and (w[pos]+2)%3
            n1 = (w[pos] + 1) % 3
            n2 = (w[pos] + 2) % 3
            nw1 = w[:]
            nw1[pos] = n1
            j1 = word_to_idx(nw1)
            mask |= (1 << j1)
            
            nw2 = w[:]
            nw2[pos] = n2
            j2 = word_to_idx(nw2)
            mask |= (1 << j2)
        balls[i] = mask

    best_code = None
    best_size = float('inf')
    
    def run_greedy_removal(randomize_greedy=False):
        uncovered = full_mask
        code = []
        code_set = set()
        
        while uncovered:
            best_idx = -1
            max_covered = 0
            candidates = []
            
            for c in all_indices:
                if c in code_set: 
                    continue
                covered = (balls[c] & uncovered).bit_count()
                if covered > max_covered:
                    max_covered = covered
                    best_idx = c
                    candidates = [c]
                elif covered == max_covered and max_covered > 0:
                    candidates.append(c)
            
            if best_idx == -1:
                break
                
            if randomize_greedy and len(candidates) > 1:
                chosen = random.choice(candidates)
            else:
                chosen = best_idx
                
            code.append(chosen)
            code_set.add(chosen)
            uncovered &= ~balls[chosen]
            
            if time.time() > deadline:
                return code

        # Removal Phase
        coverage = [0] * total_words
        for c in code:
            mask = balls[c]
            while mask:
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                coverage[idx] += 1
                mask ^= lsb

        removed_any = True
        while removed_any:
            removed_any = False
            # Randomize removal order
            indices = list(range(len(code)))
            random.shuffle(indices)
            for i in indices:
                if i >= len(code): continue # might have been removed
                c = code[i]
                can_remove = True
                mask = balls[c]
                while mask:
                    lsb = mask & -mask
                    idx = lsb.bit_length() - 1
                    if coverage[idx] == 1:
                        can_remove = False
                        break
                    mask ^= lsb
                
                if can_remove:
                    mask = balls[c]
                    while mask:
                        lsb = mask & -mask
                        idx = lsb.bit_length() - 1
                        coverage[idx] -= 1
                        mask ^= lsb
                    
                    code.pop(i)
                    code_set.discard(c)
                    removed_any = True
                    if time.time() > deadline:
                        return code
        
        return code

    # Run deterministic first
    code = run_greedy_removal(randomize_greedy=False)
    best_code = code
    best_size = len(code)
    
    # Run randomized restarts if time permits
    while time.time() < deadline - 5:
        # Randomize the initial choice or the process
        code = run_greedy_removal(randomize_greedy=True)
        if len(code) < best_size:
            best_code = code
            best_size = len(code)
            if best_size <= 186:
                break # Can't do better than known bound
        
        # Also try a "kick" on the best code found so far
        # Remove a random codeword from best_code, then re-greedy?
        # This is complex to implement safely within time. 
        # The randomized greedy above is a good proxy.

    return [idx_to_word(c) for c in best_code]
