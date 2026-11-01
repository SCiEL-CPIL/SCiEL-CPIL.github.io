"""Combined program for ternary covering code K_3(7,1)."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s limit
    
    # Specific optimization for the known task: q=3, n=7, radius=1
    if q == 3 and n == 7 and radius == 1:
        return _construct_3_7_1(deadline)
        
    # Fallback for other parameters: simple greedy + pruning
    return _construct_general(q, n, radius, deadline)

def _construct_3_7_1(deadline):
    N = 2187  # 3^7
    # Precompute powers of 3 for fast index conversion
    p3 = [1] * 7
    for i in range(1, 7):
        p3[i] = p3[i - 1] * 3

    # Precompute ball_indices: for each word index, list of indices within Hamming distance 1
    # We also precompute the bitset mask for each ball
    ball_masks = [0] * N
    
    for i in range(N):
        digits = [0] * 7
        tmp = i
        for pos in range(7):
            digits[pos] = tmp % 3
            tmp //= 3
        
        mask = 1 << i
        for pos in range(7):
            orig = digits[pos]
            for val in range(3):
                if val != orig:
                    new_i = i + (val - orig) * p3[pos]
                    mask |= (1 << new_i)
        ball_masks[i] = mask
    
    # Helper: Greedy Set Cover initialization with randomization
    def greedy_init(seed):
        rng = random.Random(seed)
        uncovered = (1 << N) - 1  # Full mask
        selected = []
        selected_set = set()
        
        while uncovered:
            best_idx = -1
            best_count = -1
            
            # To make it randomized, we shuffle the candidate order
            # This breaks ties randomly and explores different paths
            candidates = list(range(N))
            rng.shuffle(candidates)
            
            for i in candidates:
                if i in selected_set:
                    continue
                    
                # Count uncovered bits covered by i
                cnt = (uncovered & ball_masks[i]).bit_count()
                
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            
            if best_idx == -1 or best_count == 0:
                break # Should not happen if uncovered.any()
                
            selected.append(best_idx)
            selected_set.add(best_idx)
            uncovered &= ~ball_masks[best_idx]
            
            if time.time() > deadline - 5.0:
                break
                
        return selected

    # Helper: Local Search (Delete + Swap)
    def local_search(code, seed):
        rng = random.Random(seed)
        code_set = set(code)
        
        # cnt[i] is the number of codewords in 'code' that cover word i
        # We use a Python list of integers for speed, or numpy array?
        # N=2187 is small. A list of ints is very fast for random access.
        cnt = [0] * N
        for c in code:
            mask = ball_masks[c]
            # Iterate set bits to update cnt
            m = mask
            while m:
                # Get lowest set bit index
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cnt[idx] += 1
                m ^= lsb

        best_code = code.copy()
        best_size = len(code)
        
        while time.time() < deadline - 2.0:
            improved = False
            
            # Phase 1: Deletion
            # Try to remove any codeword that doesn't uniquely cover any word
            to_remove = []
            for c in code:
                mask = ball_masks[c]
                m = mask
                is_essential = False
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cnt[idx] == 1:
                        is_essential = True
                        break
                    m ^= lsb
                if not is_essential:
                    to_remove.append(c)
            
            if to_remove:
                for c in to_remove:
                    # Remove c from code
                    code.remove(c)
                    code_set.remove(c)
                    # Update counts
                    mask = ball_masks[c]
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cnt[idx] -= 1
                        m ^= lsb
                    
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code.copy()
                improved = True
                continue # Restart loop as counts changed significantly
            
            # Phase 2: Swap
            # Try to swap a codeword c with c'
            # c' must cover all words that c uniquely covers.
            
            # Pick a random codeword to potentially remove
            if not code:
                break
            c = rng.choice(code)
            
            # Find words uniquely covered by c
            unique_words = []
            mask = ball_masks[c]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if cnt[idx] == 1:
                    unique_words.append(idx)
                m ^= lsb
            
            if not unique_words:
                # Should have been caught in deletion phase, but just in case
                code.remove(c)
                code_set.remove(c)
                m = ball_masks[c]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    cnt[idx] -= 1
                    m ^= lsb
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code.copy()
                improved = True
                continue
            
            # Find candidates c' that cover all unique_words
            # Intersection of balls of unique_words
            cand_mask = ball_masks[unique_words[0]]
            for w in unique_words[1:]:
                cand_mask &= ball_masks[w]
                if not cand_mask:
                    break
            
            if not cand_mask:
                continue
            
            # Filter out candidates already in code
            # We need to find indices in cand_mask that are not in code_set
            # Iterate bits of cand_mask
            candidates = []
            m = cand_mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if idx not in code_set:
                    candidates.append(idx)
                m ^= lsb
            
            if not candidates:
                continue
            
            # Pick a random candidate
            c_prime = rng.choice(candidates)
            
            # Perform swap
            # 1. Remove c
            code.remove(c)
            code_set.remove(c)
            m = ball_masks[c]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cnt[idx] -= 1
                m ^= lsb
            
            # 2. Add c_prime
            code.append(c_prime)
            code_set.add(c_prime)
            m = ball_masks[c_prime]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cnt[idx] += 1
                m ^= lsb
            
            # Check if this swap actually helped (it should, by definition of swap, 
            # but we might want to accept it only if it leads to further improvements?
            # Actually, standard local search accepts any valid move.
            # A swap keeps the size constant. It is useful to escape local minima for deletion.
            # So we accept it immediately.
            improved = True
            
            # If we didn't improve size, we might loop forever swapping.
            # To prevent this, we can track recent moves or just rely on randomization.
            # Since we are just swapping, the size is constant.
            # The goal is to reach a state where deletion is possible.
            
            if not improved:
                # If we tried a swap and it didn't change the state (shouldn't happen with valid swap)
                # Or if we exhausted swaps?
                # Let's add a small penalty if we do many swaps without deletion
                pass

        return best_code

    # Main Loop: Multi-start
    best_solution = None
    best_size = float('inf')
    
    # Run several restarts
    # Time budget for each restart?
    # Let's just run until deadline.
    
    seed = 42
    while time.time() < deadline - 2.0:
        # Get initial greedy solution
        init_code = greedy_init(seed)
        
        if not init_code:
            break
            
        # Refine with local search
        refined_code = local_search(init_code, seed)
        
        if len(refined_code) < best_size:
            best_size = len(refined_code)
            best_solution = refined_code.copy()
            
        # Convert to list of lists for return format
        # all_words index to digits
        def idx_to_digits(idx):
            digits = [0] * 7
            tmp = idx
            for pos in range(7):
                digits[pos] = tmp % 3
                tmp //= 3
            return digits
        
        seed += 1
        
        # If we found a very good solution (close to lower bound 156), we might stop early?
        # Lower bound is 156. Best known 186.
        # If we hit 186, we can try to beat it.
        # If we hit 180, definitely keep going.
        
    if best_solution is None:
        # Fallback: return empty or a trivial valid code?
        # Must return a valid code.
        # If greedy failed, return all words (valid but huge).
        # But greedy should not fail.
        return []
    
    # Convert indices to digit lists
    result = []
    for idx in best_solution:
        digits = [0] * 7
        tmp = idx
        for pos in range(7):
            digits[pos] = tmp % 3
            tmp //= 3
        result.append(digits)
        
    return result

def _construct_general(q, n, radius, deadline):
    # Simple fallback for other parameters
    # Generate all words
    all_words = []
    for i in range(q**n):
        digits = [0] * n
        tmp = i
        for pos in range(n):
            digits[pos] = tmp % q
            tmp //= q
        all_words.append(tuple(digits))
    
    # Precompute distances or balls
    # For general case, we might not fit in bitsets if q^n is large.
    # But for this task, we assume q=3, n=7 is the main case.
    # If n is larger, this might be slow.
    
    # Simple Greedy
    covered = set()
    selected = []
    remaining = set(all_words)
    
    # Precompute ball for each word
    balls = []
    for w in all_words:
        ball = set()
        for v in all_words:
            dist = sum(1 for a, b in zip(w, v) if a != b)
            if dist <= radius:
                ball.add(v)
        balls.append(ball)
        
    while remaining:
        best_idx = -1
        best_count = -1
        for i, w in enumerate(all_words):
            if w in covered:
                continue
            count = len(balls[i] & remaining)
            if count > best_count:
                best_count = count
                best_idx = i
        
        if best_idx == -1:
            break
            
        selected.append(all_words[best_idx])
        covered.update(balls[best_idx])
        remaining -= set(balls[best_idx])
        
        if time.time() > deadline - 5.0:
            break
            
    return [list(w) for w in selected]
# EVOLVE-BLOCK-END
