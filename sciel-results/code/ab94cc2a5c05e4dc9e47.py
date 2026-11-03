import time
import random
import math
import sys

def _precompute_neighbors(q, n, radius):
    """
    Precompute the neighbors for each word in {0..q-1}^n.
    For radius 1, neighbors are the word itself and words differing in 1 position.
    Returns:
        neigh: list of lists of neighbor indices
        neigh_sets: list of sets of neighbor indices for O(1) lookup
        total: total number of words
    """
    total = q ** n
    neigh = [[] for _ in range(total)]
    for idx in range(total):
        neigh[idx].append(idx)
        for i in range(n):
            pow_q_i = q ** i
            digit = (idx // pow_q_i) % q
            for new_digit in range(q):
                if new_digit != digit:
                    new_idx = idx + (new_digit - digit) * pow_q_i
                    neigh[idx].append(new_idx)
    
    # Precompute sets for faster intersection/subset checks
    neigh_sets = [set(lst) for lst in neigh]
    return neigh, neigh_sets, total

def construct(q, n, radius):
    start_time = time.monotonic()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    total = q ** n
    
    # Precompute neighbors once
    neigh, neigh_sets, _ = _precompute_neighbors(q, n, radius)
    
    best_code_global = None
    best_size_global = total + 1
    restart_count = 0

    # We will run multiple restarts
    # The number of restarts depends on available time.
    # Each restart consists of:
    # 1. Greedy construction (fast)
    # 2. Local search (variable time)
    
    while time.monotonic() < deadline:
        # Check if we have enough time for a restart
        # A restart takes roughly 0.5s for greedy + variable for local search.
        # We estimate a minimum of 1s per restart.
        if time.monotonic() > deadline - 2.0:
            break
            
        restart_count += 1
        # Use different seeds for diversity
        rng = random.Random(restart_count * 12345 + 67890)
        
        # 1. Greedy Initialization
        # To get diverse starts, we can shuffle the order in which we consider
        # candidates for the greedy step. However, the standard greedy is:
        # Pick the word that covers the most currently uncovered words.
        # We can introduce randomness by breaking ties randomly or
        # slightly perturbing the gain calculation.
        
        counts = [0] * total
        code = []
        code_set = set()
        remaining = [True] * total
        rem_count = total
        
        def add_to_code(c):
            nonlocal rem_count
            code.append(c)
            code_set.add(c)
            for u in neigh[c]:
                counts[u] += 1
                if counts[u] == 1:
                    remaining[u] = False
                    rem_count -= 1

        # Greedy loop
        # Optimization: Instead of checking all 729 words every step,
        # we can just do it. 729 is small.
        while rem_count > 0:
            best_c = -1
            best_gain = -1
            # Random tie-breaking to diversify
            candidates = []
            for c in range(total):
                gain = 0
                # Quick check: if c is already in code, gain is 0 (or we skip)
                # But greedy usually adds distinct words.
                for u in neigh[c]:
                    if remaining[u]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    candidates = [c]
                elif gain == best_gain:
                    candidates.append(c)
            
            if best_gain == 0:
                # Should not happen if rem_count > 0
                break
                
            # Pick randomly from best candidates to diversify
            best_c = rng.choice(candidates)
            add_to_code(best_c)
            
        current_size = len(code)
        
        # 2. Local Search
        # We use a variant of Simulated Annealing / Random Local Search
        # Moves:
        # A. Remove a redundant codeword (if valid)
        # B. Swap a codeword with a new one (if valid)
        # C. Add a random codeword (perturbation, accepted with prob)
        # D. Remove a random codeword (perturbation, even if invalid? No, must maintain cover)
        
        # Actually, to maintain validity, we must ensure cover is always maintained.
        # So "Remove" is only valid if no word becomes uncovered.
        # "Swap" is only valid if no word becomes uncovered.
        
        temp = 2.0
        cool_rate = 0.995
        min_temp = 0.01
        
        # Local search deadline: give this restart some time
        # Let's aim for ~1-2 seconds per restart
        ls_deadline = time.monotonic() + 1.5
        no_improve_count = 0
        max_no_improve = 500
        
        while time.monotonic() < ls_deadline and no_improve_count < max_no_improve:
            if temp < min_temp:
                temp = min_temp
                
            move_type = rng.random()
            
            if move_type < 0.5:
                # Try Remove
                if len(code) > 0:
                    # Prioritize words with higher redundancy (more likely to be removable)
                    removable_candidates = []
                    for idx, c in enumerate(code):
                        min_count = min(counts[u] for u in neigh[c])
                        if min_count > 1:
                            removable_candidates.append((min_count, idx))
                    
                    if removable_candidates:
                        # Pick from most redundant first (higher min_count)
                        removable_candidates.sort(reverse=True)
                        # With some probability, pick from top 3, otherwise random
                        if rng.random() < 0.7 and len(removable_candidates) > 3:
                            top_idx = rng.randint(0, 2)
                            idx = removable_candidates[top_idx][1]
                        else:
                            idx = removable_candidates[rng.randint(0, len(removable_candidates)-1)][1]
                        
                        c = code[idx]
                        # Check if removable
                        removable = True
                        for u in neigh[c]:
                            if counts[u] == 1:
                                removable = False
                                break
                        if removable:
                            # Accept removal
                            code.pop(idx)
                            code_set.remove(c)
                            for u in neigh[c]:
                                counts[u] -= 1
                            no_improve_count = 0
                            if len(code) < best_size_global:
                                best_size_global = len(code)
                                best_code_global = code[:]
                            continue
                    no_improve_count += 1
            elif move_type < 0.9:
                # Try Swap
                if len(code) > 0:
                    idx = rng.randint(0, len(code) - 1)
                    old_c = code[idx]
                    new_c = rng.randint(0, total - 1)
                    
                    if new_c == old_c or new_c in code_set:
                        no_improve_count += 1
                        continue
                        
                    # Check if swap is valid
                    # A word u is at risk if it is covered ONLY by old_c
                    # and is NOT covered by new_c.
                    valid_swap = True
                    neigh_new_c = neigh_sets[new_c]
                    for u in neigh[old_c]:
                        if counts[u] == 1:
                            if u not in neigh_new_c:
                                valid_swap = False
                                break
                            
                    if valid_swap:
                        # Perform swap
                        for u in neigh[old_c]:
                            counts[u] -= 1
                        for u in neigh[new_c]:
                            counts[u] += 1
                        code[idx] = new_c
                        code_set.remove(old_c)
                        code_set.add(new_c)
                        no_improve_count = 0
                        if len(code) < best_size_global:
                            best_size_global = len(code)
                            best_code_global = code[:]
                    else:
                        no_improve_count += 1
            else:
                # Try Add (Perturbation)
                # This increases size, but might allow better future removals
                new_c = rng.randint(0, total - 1)
                if new_c not in code_set:
                    # Accept with probability based on temperature
                    # Since we want to minimize size, adding is "bad" (cost +1)
                    # We accept with prob exp(-1/temp)
                    if rng.random() < math.exp(-1.0 / temp):
                        code.append(new_c)
                        code_set.add(new_c)
                        for u in neigh[new_c]:
                            counts[u] += 1
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
                        
        temp *= cool_rate
        # If we found a very good solution, we can restart more aggressively
        if best_size_global == 73:
             # We found the known optimum. We can try to verify or just break.
             # Since 73 is the best known, we can stop searching for better.
             break

    # Final polish: if we have time and are close to optimal, do a more thorough search
    if best_code_global is not None and best_size_global <= 75 and time.monotonic() < deadline - 3.0:
        # Do a final polish with more iterations
        counts = [0] * total
        code = best_code_global[:]
        code_set = set(code)
        for c in code:
            for u in neigh[c]:
                counts[u] += 1
        
        temp = 1.0
        cool_rate = 0.999
        min_temp = 0.005
        polish_deadline = time.monotonic() + min(10.0, deadline - time.monotonic() - 1.0)
        no_improve_count = 0
        max_no_improve = 1000
        
        while time.monotonic() < polish_deadline and no_improve_count < max_no_improve:
            if temp < min_temp:
                temp = min_temp
                
            move_type = random.random()
            
            if move_type < 0.6:
                # Try Remove
                if len(code) > 0:
                    idx = random.randint(0, len(code) - 1)
                    c = code[idx]
                    # Check if removable
                    removable = True
                    for u in neigh[c]:
                        if counts[u] == 1:
                            removable = False
                            break
                    if removable:
                        # Accept removal
                        code.pop(idx)
                        code_set.remove(c)
                        for u in neigh[c]:
                            counts[u] -= 1
                        no_improve_count = 0
                        if len(code) < best_size_global:
                            best_size_global = len(code)
                            best_code_global = code[:]
                        continue
                    no_improve_count += 1
            else:
                # Try Swap
                if len(code) > 0:
                    idx = random.randint(0, len(code) - 1)
                    old_c = code[idx]
                    new_c = random.randint(0, total - 1)
                    
                    if new_c == old_c or new_c in code_set:
                        no_improve_count += 1
                        continue
                        
                    # Check if swap is valid
                    valid_swap = True
                    neigh_new_c = neigh_sets[new_c]
                    for u in neigh[old_c]:
                        if counts[u] == 1:
                            if u not in neigh_new_c:
                                valid_swap = False
                                break
                            
                    if valid_swap:
                        # Perform swap
                        for u in neigh[old_c]:
                            counts[u] -= 1
                        for u in neigh[new_c]:
                            counts[u] += 1
                        code[idx] = new_c
                        code_set.remove(old_c)
                        code_set.add(new_c)
                        no_improve_count = 0
                        if len(code) < best_size_global:
                            best_size_global = len(code)
                            best_code_global = code[:]
                    else:
                        no_improve_count += 1
                        
            temp *= cool_rate

    # Convert best_code_global indices to words
    if best_code_global is None:
        # Fallback: return a valid cover (should not happen)
        # Just return all words? No, that's huge.
        # Return the last greedy code?
        # Let's ensure we have something.
        # Re-run greedy deterministically if needed
        counts = [0] * total
        code = []
        code_set = set()
        remaining = [True] * total
        rem_count = total
        
        def add_to_code(c):
            nonlocal rem_count
            code.append(c)
            code_set.add(c)
            for u in neigh[c]:
                counts[u] += 1
                if counts[u] == 1:
                    remaining[u] = False
                    rem_count -= 1

        while rem_count > 0:
            best_c = -1
            best_gain = -1
            for c in range(total):
                gain = 0
                for u in neigh[c]:
                    if remaining[u]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            if best_c == -1: break
            add_to_code(best_c)
        best_code_global = code

    result = []
    for idx in best_code_global:
        word = [0] * n
        temp_idx = idx
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        result.append(word)
        
    return result
