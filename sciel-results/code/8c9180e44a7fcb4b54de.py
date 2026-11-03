import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    time_limit = 115.0 # Safety margin for 120s limit
    deadline = start_time + time_limit

    # Total words in space
    total_words = q ** n

    # Map word (tuple) to index
    def tuple_to_idx(t):
        idx = 0
        for val in t:
            idx = idx * q + val
        return idx
        
    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % q
            idx //= q
        return t

    # Generate all words
    all_words = [idx_to_tuple(i) for i in range(total_words)]
    
    # Precompute ball masks using Python integers as bitsets
    # Bit i is set if word i is covered.
    ball_masks = [0] * total_words
    for c_idx, c_word in enumerate(all_words):
        mask = 0
        mask |= (1 << c_idx)
        for pos in range(n):
            orig_val = c_word[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                neighbor = list(c_word)
                neighbor[pos] = new_val
                n_idx = tuple_to_idx(neighbor)
                mask |= (1 << n_idx)
        ball_masks[c_idx] = mask
        
    # Precompute list of indices for each mask to speed up coverage counting
    # mask_indices[i] is a list of word indices covered by codeword i
    mask_indices = []
    for i in range(total_words):
        mask = ball_masks[i]
        inds = []
        m = mask
        while m:
            # Get index of lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            inds.append(idx)
            m ^= lsb
        mask_indices.append(inds)

    def greedy_solve(random_seed):
        rng = random.Random(random_seed)
        uncovered = (1 << total_words) - 1
        selected = []
        
        while uncovered:
            best_gain = -1
            best_c_idx = -1
            # Random tie-breaking helps escape local optima
            for c_idx in range(total_words):
                gain = (uncovered & ball_masks[c_idx]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_c_idx = c_idx
                elif gain == best_gain and gain > 0:
                    # Random tie break
                    if rng.random() < 0.5:
                        best_c_idx = c_idx
                        
            if best_c_idx == -1 or best_gain == 0:
                break
                
            selected.append(best_c_idx)
            uncovered &= ~ball_masks[best_c_idx]
            
        # Local search reduction (Pruning)
        # We maintain cover_count: how many times each word is covered
        # If a codeword covers no word uniquely (count == 1), it is redundant.
        
        def reduce_code(code_list):
            if not code_list:
                return code_list
            
            # Initialize cover counts
            cover_count = [0] * total_words
            for c_idx in code_list:
                for w in mask_indices[c_idx]:
                    cover_count[w] += 1
            
            # Iteratively remove redundant codewords
            changed = True
            while changed and time.time() < deadline - 5:
                changed = False
                new_code = []
                for c_idx in code_list:
                    # Check if c_idx is essential
                    essential = False
                    for w in mask_indices[c_idx]:
                        if cover_count[w] == 1:
                            essential = True
                            break
                    
                    if essential:
                        new_code.append(c_idx)
                    else:
                        # Remove c_idx and update counts
                        for w in mask_indices[c_idx]:
                            cover_count[w] -= 1
                        changed = True
                
                code_list = new_code
                
            return code_list
        
        reduced_code = reduce_code(selected)
        return [all_words[i] for i in reduced_code]

    def local_search_improve(code_list, rng):
        # Try to swap one codeword for another if it allows removing others?
        # Or simply try random removals if possible?
        # The pruning above is exhaustive for "removable" codewords.
        # To improve further, we might need to replace a codeword with a different one
        # that covers the same unique words but allows more global redundancy.
        
        # Simple heuristic: Randomly shuffle and re-prune?
        # Or: Pick a codeword, try replacing it with a random other codeword,
        # then prune. If size decreases, keep.
        
        current_code = code_list[:]
        
        # Limit iterations to keep it fast
        max_iters = 20
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            if not current_code:
                break
                
            # Pick a random codeword to replace
            idx_to_replace = rng.randint(0, len(current_code) - 1)
            old_cw = current_code[idx_to_replace]
            
            # Try a random replacement
            new_cw = rng.randint(0, total_words - 1)
            if new_cw == old_cw:
                continue
                
            # Tentatively replace
            candidate_code = current_code[:]
            candidate_code[idx_to_replace] = new_cw
            
            # Prune
            pruned = reduce_code(candidate_code)
            
            if len(pruned) < len(current_code):
                current_code = pruned
            # If len is equal, we might still want to keep the new one if it's "better" positioned?
            # Hard to define. Let's stick to strict improvement for now.
            
        return current_code

    best_solution = None
    best_len = float('inf')
    
    seed = 0
    while time.time() < deadline:
        # Run greedy
        solution = greedy_solve(seed)
        
        # Local search improvement
        rng = random.Random(seed)
        improved_solution = local_search_improve(solution, rng)
        
        if len(improved_solution) < best_len:
            best_len = len(improved_solution)
            best_solution = improved_solution
            
        seed += 1
        # If we get a very good solution, we might want to spend more time?
        # Or just keep trying.
        
    if best_solution is None:
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return best_solution
