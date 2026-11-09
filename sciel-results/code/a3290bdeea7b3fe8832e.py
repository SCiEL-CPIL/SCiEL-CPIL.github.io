import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 60s limit
    
    N = 17
    MIN_D = 6
    TOTAL = 1 << N
    
    # Precompute weights using numpy for speed
    all_words = np.arange(TOTAL, dtype=np.uint32)
    # Compute popcount for all words
    weights = np.zeros(TOTAL, dtype=np.uint8)
    for i in range(TOTAL):
        weights[i] = bin(i).count('1')
    
    # 1. Generate the known 256-word linear code
    def generate_linear_code():
        basis = []
        current_code = [0]
        
        for g in range(1, TOTAL):
            if time.time() > deadline:
                break
            valid = True
            for c in current_code:
                if weights[g ^ c] < MIN_D:
                    valid = False
                    break
            if valid:
                basis.append(g)
                new_code = []
                for c in current_code:
                    new_code.append(c)
                    new_code.append(c ^ g)
                current_code = new_code
                if len(basis) >= 8:
                    break
        return current_code

    best_code = generate_linear_code()
    best_size = len(best_code)
    
    # Convert to numpy array for fast operations
    best_code_np = np.array(best_code, dtype=np.uint32)
    
    # Helper to check if a word can be added using numpy vectorization
    def can_add_fast(word, code_np):
        """Check if word has min distance >= MIN_D from all words in code_np"""
        if len(code_np) == 0:
            return True
        # Compute XOR distances
        xors = code_np ^ word
        # Compute popcounts
        dists = np.array([bin(x).count('1') for x in xors], dtype=np.uint8)
        return dists.min() >= MIN_D
    
    # Faster: precompute popcount lookup for 17-bit values
    popcount_lookup = np.array([bin(i).count('1') for i in range(TOTAL)], dtype=np.uint8)
    
    def can_add_fast2(word, code_np):
        """Check if word has min distance >= MIN_D from all words in code_np using lookup"""
        if len(code_np) == 0:
            return True
        xors = code_np ^ word
        dists = popcount_lookup[xors]
        return dists.min() >= MIN_D
    
    # Helper to perform a greedy pass
    def greedy_pass(order):
        code = []
        code_np = np.array([], dtype=np.uint32)
        for w in order:
            if can_add_fast2(w, code_np):
                code.append(w)
                code_np = np.append(code_np, w)
        return code
    
    # Local improvement: try to add words to an existing code
    def try_extend(code, remaining_candidates):
        """Try to add words from remaining_candidates to code"""
        code_set = set(code)
        code_np = np.array(code, dtype=np.uint32)
        extended = list(code)
        for w in remaining_candidates:
            if w in code_set:
                continue
            if can_add_fast2(w, code_np):
                extended.append(w)
                code_set.add(w)
                code_np = np.append(code_np, w)
        return extended
    
    # Swap-based local search: try replacing one word with another
    def local_search_swap(code, all_words_list, max_swaps=50):
        """Try to improve code by swapping words"""
        code_set = set(code)
        code_np = np.array(code, dtype=np.uint32)
        improved = True
        swap_count = 0
        while improved and swap_count < max_swaps:
            improved = False
            swap_count += 1
            if time.time() > deadline - 5:
                break
            # For each word in code, try to remove it and see if we can add more
            for i in range(len(code)):
                if time.time() > deadline - 5:
                    break
                removed = code[i]
                temp_code = code[:i] + code[i+1:]
                temp_set = set(temp_code)
                temp_np = np.array(temp_code, dtype=np.uint32)
                # Try to add words that were previously blocked
                added = 0
                for w in all_words_list:
                    if w in temp_set:
                        continue
                    if can_add_fast2(w, temp_np):
                        temp_code.append(w)
                        temp_set.add(w)
                        temp_np = np.append(temp_np, w)
                        added += 1
                        if added >= 2:  # Net gain of at least 1
                            break
                if len(temp_code) > len(code):
                    code = temp_code
                    code_set = temp_set
                    code_np = temp_np
                    improved = True
                    break
        return code
    
    # To speed up can_add, we can limit checks if we just want to find *any* code,
    # but we want the *largest*.
    
    # We will run multiple randomized greedy passes.
    # We can also try different initial seeds or orderings.
    
    # Strategy: 
    # - Shuffle all_words
    # - Run greedy
    # - Update best
    
    # To make it faster, we can sample a subset of words? No, we need all candidates.
    # But we can skip words that are close to existing ones quickly?
    # The O(N * |code|) check is the bottleneck.
    
    # Let's run a loop of randomized greedy passes.
    
    # We can also try to "repair" a code if it gets stuck?
    # For now, simple restarts are safer.
    
    # Optimization: 
    # If we find a code of size > best_size, we save it.
    # We can also try to add a few words to the linear code manually if we find a "gap"?
    # But random greedy from scratch is more likely to find the 258-word structure if it exists 
    # as a local optimum of the greedy landscape.
    
    # Let's add a slight bias: 
    # In each pass, we can choose the shuffle strategy.
    # Strategy 1: Uniform Random
    # Strategy 2: Weight Sorted (Randomized within weights)
    
    # We'll stick to Uniform Random for simplicity and robustness.
    
    # Precompute weight groups for weighted ordering
    weight_groups = {}
    for w in range(TOTAL):
        wt = weights[w]
        if wt not in weight_groups:
            weight_groups[wt] = []
        weight_groups[wt].append(w)
    
    # Create all_words as a list for shuffling
    all_words_list = list(range(TOTAL))
    
    pass_count = 0
    while time.time() < deadline - 10:  # Leave 10s for cleanup/return
        pass_count += 1
        
        # Alternate between strategies
        if pass_count % 3 == 0:
            # Weight-based ordering: shuffle within each weight class
            order = []
            for wt in sorted(weight_groups.keys()):
                group = weight_groups[wt].copy()
                random.shuffle(group)
                order.extend(group)
            # Also shuffle the weight order sometimes
            if random.random() < 0.5:
                weight_order = sorted(weight_groups.keys())
                random.shuffle(weight_order)
                order = []
                for wt in weight_order:
                    order.extend(weight_groups[wt])
        elif pass_count % 3 == 1:
            # Uniform random
            random.shuffle(all_words_list)
            order = all_words_list[:]
        else:
            # Start from best code and try to extend
            if len(best_code) > 0:
                # Create candidates not in best_code
                best_set = set(best_code)
                candidates = [w for w in range(TOTAL) if w not in best_set]
                random.shuffle(candidates)
                order = best_code + candidates
            else:
                random.shuffle(all_words_list)
                order = all_words_list[:]
        
        current_code = greedy_pass(order)
        
        if len(current_code) > best_size:
            best_code = current_code
            best_size = len(current_code)
            best_code_np = np.array(best_code, dtype=np.uint32)
        
        # Try local improvement on best code
        if len(best_code) > 0 and time.time() < deadline - 15:
            # Try to extend with words not in current best
            best_set = set(best_code)
            candidates = [w for w in range(TOTAL) if w not in best_set]
            random.shuffle(candidates)
            # Only try a subset to save time
            subset = candidates[:200]
            extended = try_extend(best_code, subset)
            if len(extended) > best_size:
                best_code = extended
                best_size = len(extended)
                best_code_np = np.array(best_code, dtype=np.uint32)
        
        # Try swap-based local search occasionally
        if len(best_code) > 0 and pass_count % 5 == 0 and time.time() < deadline - 20:
            # Use a smaller candidate set for swap search
            best_set = set(best_code)
            swap_candidates = [w for w in range(TOTAL) if w not in best_set][:100]
            random.shuffle(swap_candidates)
            improved_code = local_search_swap(best_code, swap_candidates, max_swaps=10)
            if len(improved_code) > best_size:
                best_code = improved_code
                best_size = len(improved_code)
                best_code_np = np.array(best_code, dtype=np.uint32)
    
    return best_code
