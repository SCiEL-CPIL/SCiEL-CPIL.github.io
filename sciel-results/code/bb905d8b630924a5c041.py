import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 90s hard limit

    N = 17
    MIN_D = 6
    TOTAL = 1 << N

    # Precompute popcount lookup for all 17-bit values (fast vectorized index)
    popcount_lookup = np.array([bin(i).count('1') for i in range(TOTAL)], dtype=np.uint8)
    
    # Precompute all words as numpy array
    all_words_np = np.arange(TOTAL, dtype=np.uint32)

    # ---- 1. Generate the known 256-word linear code [17,8,6] ----
    def generate_linear_code():
        basis = []
        current_code = [0]
        # Try to find a basis of size 8
        # We can just iterate through all words and greedily add if independent
        # This is fast enough for N=17
        for g in range(1, TOTAL):
            if time.time() > deadline:
                break
            # Check if g is at distance >= 6 from all current codewords
            # Vectorized check
            current_np = np.array(current_code, dtype=np.uint32)
            xors = current_np ^ g
            dists = popcount_lookup[xors]
            if dists.min() >= MIN_D:
                basis.append(g)
                # Expand code
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
    best_set = set(best_code)
    
    # Convert best_code to numpy for fast operations
    best_code_np = np.array(best_code, dtype=np.uint32)
    
    # Precompute a mask of words NOT in best_code
    # This is used for candidate selection
    in_code_mask = np.zeros(TOTAL, dtype=bool)
    in_code_mask[best_code_np] = True
    
    # ---- Vectorized Feasibility Check ----
    def check_feasibility(candidates_np, code_np):
        """
        Returns a boolean array of the same length as candidates_np.
        True if the candidate is at distance >= MIN_D from ALL codewords in code_np.
        """
        if code_np.size == 0:
            return np.ones(len(candidates_np), dtype=bool)
        
        # We need to check dist(cand, code) >= MIN_D for all code.
        # If we do candidates[:, None] ^ code[None, :], we get a (N_cand, N_code) matrix.
        # For N_cand=1000, N_code=256, this is 256k elements. Very fast.
        # For N_cand=10000, N_code=256, this is 2.5M elements. Still fast.
        
        # To save memory if candidates are huge, we can batch, but 131k * 256 is ~34M uint8s = 34MB.
        # This is acceptable. Let's try full vectorization for the whole candidate pool if possible.
        
        xors = candidates_np[:, None] ^ code_np[None, :]
        dists = popcount_lookup[xors]
        min_dists = dists.min(axis=1)
        return min_dists >= MIN_D

    # ---- Fast Greedy Pass ----
    def greedy_pass(order):
        code = []
        code_np = np.array([], dtype=np.uint32)
        for w in order:
            if time.time() > deadline - 5:
                break
            if code_np.size == 0:
                code.append(w)
                code_np = np.array([w], dtype=np.uint32)
            else:
                # Check w against code_np
                xors = code_np ^ w
                if popcount_lookup[xors].min() >= MIN_D:
                    code.append(w)
                    code_np = np.append(code_np, w)
        return code

    # ---- Vectorized Local Search (Swap) ----
    # Strategy:
    # 1. Pick a random codeword to remove.
    # 2. Check all candidates (not in code) against the remaining code.
    # 3. Add all feasible candidates (or a subset) to see if we improve size.
    # 4. If size improves, keep. Else, revert.
    # To make it faster, we can try to remove "central" codewords or just random ones.
    
    def local_search(code_list, code_np, in_code_mask):
        n = len(code_list)
        candidates_np = all_words_np[~in_code_mask]
        
        # We will try a number of swaps
        attempts = 0
        max_attempts = 5000 # Cap attempts based on time
        
        while time.time() < deadline - 10:
            attempts += 1
            if attempts > max_attempts:
                break
                
            # Pick a random index to remove
            idx = random.randint(0, n - 1)
            removed_word = code_list[idx]
            
            # Create new code list and np array
            new_code_list = code_list[:idx] + code_list[idx+1:]
            new_code_np = np.delete(code_np, idx)
            
            # Check feasibility of all current candidates against new_code_np
            # Note: candidates_np is fixed (words not in original code).
            # The removed word is not in candidates_np.
            # We can add the removed word back to candidates? No, we are removing it.
            # We want to add OTHER words.
            
            feas_mask = check_feasibility(candidates_np, new_code_np)
            
            # Count how many we can add
            # If we add ALL feasible ones, we might get a huge jump.
            # But adding them all changes the code, so we should check if the NEW code is valid?
            # Yes, because check_feasibility ensures each new word is far from OLD code.
            # But new words must be far from EACH OTHER.
            # So we can't just add all feasible ones. We need a greedy addition from the feasible set.
            
            feasible_candidates = candidates_np[feas_mask]
            
            if len(feasible_candidates) == 0:
                continue
                
            # Greedily add from feasible_candidates
            # Sort by weight? Or random? Random is good for diversity.
            # Let's just take them in order (which is index order) or shuffle.
            # Shuffling is expensive if done every time. 
            # Let's just take the first K feasible ones? 
            # Actually, since they are all feasible against the OLD code, 
            # we just need to ensure they are feasible against EACH OTHER.
            
            added = 0
            new_code_list_local = new_code_list
            new_code_np_local = new_code_np
            
            # We can vectorize this greedy addition if we are careful, 
            # but simple loop is okay if the number of additions is small.
            # However, if we can add 10 words, that's a big win.
            
            # Let's try to add as many as possible.
            # To speed up, we can check batches.
            
            for w in feasible_candidates:
                if time.time() > deadline - 10:
                    break
                # Check w against new_code_np_local
                xors = new_code_np_local ^ w
                if popcount_lookup[xors].min() >= MIN_D:
                    new_code_list_local.append(int(w))
                    new_code_np_local = np.append(new_code_np_local, w)
                    added += 1
                    # Stop if we added enough to improve? 
                    # We removed 1. If added >= 1, we are tied or better.
                    # If added > 1, we are better.
                    # Let's keep going to maximize gain.
            
            if len(new_code_list_local) > n:
                # Update best
                code_list = new_code_list_local
                code_np = new_code_np_local
                n = len(code_list)
                
                # Update in_code_mask
                in_code_mask = np.zeros(TOTAL, dtype=bool)
                in_code_mask[code_np] = True
                candidates_np = all_words_np[~in_code_mask]
                
                # Reset attempts counter? No, keep going.
                
        return code_list, code_np, in_code_mask

    # Main Loop
    # 1. Start with linear code
    # 2. Run Local Search
    # 3. If no improvement, try Greedy with different orders to escape local minima?
    #    Or just keep doing Local Search with different random removals.
    
    # Let's do a few rounds of Local Search.
    # If stuck, we can try a "Greedy Restart" from a perturbed version.
    
    code_list, code_np, in_code_mask = local_search(best_code, best_code_np, in_code_mask)
    
    if len(code_list) > best_size:
        best_size = len(code_list)
        best_code = code_list
        best_code_np = code_np
        best_set = set(code_list)

    # If we have time, try to break out of local minima by random restarts or large swaps
    while time.time() < deadline - 5:
        # Try a different strategy: 
        # Remove a batch of words and refill?
        # Or just continue local search.
        
        # Let's try to randomize the order of candidates in local search?
        # The current local search iterates candidates in index order.
        # This might be biased.
        
        # Let's try a "Simulated Annealing" style:
        # Accept worse moves? No, we want to maximize size.
        
        # Let's try to remove 2 words and add 3?
        # That's complex.
        
        # Let's just run more local search passes.
        code_list, code_np, in_code_mask = local_search(code_list, code_np, in_code_mask)
        
        if len(code_list) > best_size:
            best_size = len(code_list)
            best_code = code_list
            best_code_np = code_np
            best_set = set(code_list)
        else:
            # If no improvement, we might be stuck.
            # Let's try a greedy pass from scratch with a random seed to see if we find a larger code?
            # Unlikely to beat 256 if the linear code is good, but nonlinear codes exist.
            
            # Let's try to perturb the current code: remove 5 random words, then refill greedily.
            if len(code_list) > 10:
                perturb_idx = random.sample(range(len(code_list)), 5)
                perturb_set = set(perturb_idx)
                new_code_list = [w for i, w in enumerate(code_list) if i not in perturb_set]
                new_code_np = np.array(new_code_list, dtype=np.uint32)
                
                in_code_mask = np.zeros(TOTAL, dtype=bool)
                in_code_mask[new_code_np] = True
                candidates_np = all_words_np[~in_code_mask]
                
                # Greedily add from candidates
                # Shuffle candidates?
                random.shuffle(candidates_np.tolist()) # This is slow if list conversion is heavy
                # Better:
                indices = np.arange(len(candidates_np))
                np.random.shuffle(indices)
                candidates_np = candidates_np[indices]
                
                for w in candidates_np:
                    if time.time() > deadline - 5:
                        break
                    xors = new_code_np ^ w
                    if popcount_lookup[xors].min() >= MIN_D:
                        new_code_list.append(int(w))
                        new_code_np = np.append(new_code_np, w)
                        
                if len(new_code_list) > best_size:
                    best_size = len(new_code_list)
                    best_code = new_code_list
                    best_code_np = new_code_np
                    best_set = set(new_code_list)
                    # Update local search state
                    in_code_mask = np.zeros(TOTAL, dtype=bool)
                    in_code_mask[best_code_np] = True
                    candidates_np = all_words_np[~in_code_mask]

    return best_code
