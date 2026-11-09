import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 60s limit
    
    N = 17
    MIN_D = 6
    TOTAL = 1 << N
    
    # Precompute popcount for all 17-bit values
    # Using a lookup table is fastest
    popcount_lookup = np.array([bin(i).count('1') for i in range(TOTAL)], dtype=np.uint8)
    
    # 1. Generate the known 256-word linear code
    # We can hardcode the generator or use a fast greedy.
    # The seed program finds it quickly. Let's keep that but make it faster.
    
    def generate_linear_code():
        # Use a known basis for [18,9,6] shortened or just greedy.
        # Greedy is fine for 256 words.
        code = [0]
        code_set = {0}
        # Try to add words in increasing order of weight? 
        # Linear codes have specific structure.
        # Let's just use the existing greedy but optimized.
        for g in range(1, TOTAL):
            if time.time() > deadline - 40:
                break
            # Check if g is far from all in code
            valid = True
            for c in code:
                if popcount_lookup[g ^ c] < MIN_D:
                    valid = False
                    break
            if valid:
                code.append(g)
                code_set.add(g)
                # Double the code
                new_words = [c ^ g for c in code[:-1]] # c ^ g for all previous
                # Actually, if we add g, we must add c^g for all c in code.
                # The loop above only checks g. We need to ensure c^g is also valid?
                # In a linear code construction, if g is far from all c, then g^c is far from all c'?
                # dist(g^c, c') = dist(g, c^c'). Since code is linear/closed?
                # This greedy builds a linear code if we always add g and c^g.
                # But checking g against all c is sufficient if we maintain the code as a subspace?
                # No, this greedy just finds a set.
                # To ensure it's linear, we should maintain basis.
                # But we just need a large set.
                # The standard greedy for linear codes:
                # If g is not in span(basis) and dist(g, span) >= d, add to basis.
                # This is harder to check quickly.
                
                # The current code in the prompt does:
                # if valid: basis.append(g); new_code = [c, c^g for c in current_code]
                # This ensures the set is closed under XOR with g.
                
                # Let's implement that properly.
                pass
        
        return code

    # The prompt's generate_linear_code is okay. Let's use it but with the fast popcount.
    def generate_linear_code_fast():
        code = [0]
        for g in range(1, TOTAL):
            if time.time() > deadline - 40:
                break
            valid = True
            for c in code:
                if popcount_lookup[g ^ c] < MIN_D:
                    valid = False
                    break
            if valid:
                # Add g and all c^g
                new_code = []
                for c in code:
                    new_code.append(c)
                    new_code.append(c ^ g)
                code = new_code
                if len(code) >= 256:
                    break
        return code

    best_code = generate_linear_code_fast()
    best_size = len(best_code)
    
    # Convert to list and set for fast manipulation
    best_set = set(best_code)
    
    # Helper to check if a word can be added
    def can_add(word, code_list, code_set):
        if word in code_set:
            return False
        for c in code_list:
            if popcount_lookup[word ^ c] < MIN_D:
                return False
        return True

    # Greedy Pass
    def greedy_pass(order, initial_code=None):
        if initial_code:
            code = list(initial_code)
            code_set = set(code)
        else:
            code = []
            code_set = set()
            
        for w in order:
            if w in code_set:
                continue
            if can_add(w, code, code_set):
                code.append(w)
                code_set.add(w)
                
        return code, code_set

    # Local Search: Try to swap or add
    # Strategy: 
    # 1. Try to add any word.
    # 2. If stuck, try to swap one word with another to enable addition.
    
    def local_improve(code, code_set, candidates, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try to add
            for w in candidates:
                if w in code_set:
                    continue
                if can_add(w, code, code_set):
                    code.append(w)
                    code_set.add(w)
                    improved = True
                    # Restart candidate check? Or just continue?
                    # If we add, the set of valid candidates changes.
                    # For simplicity, break and restart loop.
                    break
            
            if improved:
                continue
                
            # If no add, try swap
            # Pick a random word to remove
            if len(code) == 0:
                break
            idx = random.randint(0, len(code)-1)
            removed_word = code[idx]
            code.pop(idx)
            code_set.remove(removed_word)
            
            # Try to add 2 words to compensate
            added_any = False
            # We need to find words that are compatible with the new code
            # and each other.
            # This is expensive.
            # Instead, just try to add ONE word that is "better" or random.
            # If we can't add even one, revert.
            
            # Let's try to add the removed word back if we can't find a better config.
            # This is a "no-op" swap.
            
            # Better: Try to add a word that was blocked by removed_word.
            # We don't track blocked words.
            
            # Simple Swap: Try to replace removed_word with a random candidate.
            # If valid, keep.
            # This might not increase size, but might change structure.
            
            # For now, just revert if we didn't add.
            code.append(removed_word)
            code_set.add(removed_word)
            
            # To avoid infinite loop of no-ops, we should track failed swaps.
            # But for now, the "Add" phase is the main driver.
            
        return code, code_set

    # Main Loop
    all_words_list = list(range(TOTAL))
    
    # Pre-shuffle candidates for local search
    # We want to try words that are "likely" to fit.
    
    while time.time() < deadline - 5:
        # 1. Random Greedy Restart
        random.shuffle(all_words_list)
        # Bias towards middle weights?
        # For now, uniform.
        
        # Start from best code? Or from scratch?
        # Starting from best code and trying to extend is usually better.
        
        # Try extending best code
        candidates = [w for w in all_words_list if w not in best_set]
        random.shuffle(candidates)
        
        # Run greedy on candidates
        new_code, new_set = greedy_pass(candidates, initial_code=best_code)
        
        if len(new_code) > best_size:
            best_code = new_code
            best_set = new_set
            best_size = len(best_code)
        
        # Try local improvement on the current best
        if time.time() < deadline - 10:
            # Use a subset of candidates for speed
            # Pick candidates that are close to the code?
            # Or just random.
            local_candidates = random.sample(all_words_list, min(1000, len(all_words_list)))
            # Filter out those already in best
            local_candidates = [w for w in local_candidates if w not in best_set]
            
            improved_code, improved_set = local_improve(best_code, best_set, local_candidates, deadline - 10)
            
            if len(improved_code) > best_size:
                best_code = improved_code
                best_set = improved_set
                best_size = len(best_code)
                
    return best_code
