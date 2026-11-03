import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Precompute neighbors and masks for the specific space
    # For K_3(7,1), total_space = 2187.
    total_space = q ** n
    
    # If the space is too large for bitmask approach, we might need a different strategy,
    # but for the given task (n=7, q=3), bitmasks are extremely efficient.
    if total_space > 50000:
        # Fallback: simple greedy (likely to be slow or memory heavy for huge spaces)
        # But the task is specifically K_3(7,1).
        pass

    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # neighbors[i]: list of indices within Hamming distance <= radius
    # For radius 1, this is 1 + n*(q-1) = 1 + 7*2 = 15 neighbors.
    neighbors = [[] for _ in range(total_space)]
    for i in range(total_space):
        neighbors[i].append(i)
        for pos in range(n):
            digit_val = (i // powers[pos]) % q
            for new_val in range(q):
                if new_val == digit_val:
                    continue
                diff = new_val - digit_val
                neighbors[i].append(i + diff * powers[pos])
                
    # Convert neighbors to a set for O(1) membership check in local search?
    # Actually, for the "unique coverage" check, we need to know if a specific neighbor
    # is covered ONLY by the current codeword. This is handled by counts.
    # However, for the "can w' replace w?" check, we need to verify if w' covers all uniques.
    # Precomputing a bitmask for each codeword is useful for fast intersection.
    
    masks = [0] * total_space
    for i in range(total_space):
        m = 0
        for nb in neighbors[i]:
            m |= (1 << nb)
        masks[i] = m
        
    full_mask = (1 << total_space) - 1
    
    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
        return res

    # --- Phase 1: Initial Solution via Exhaustive Weighted Greedy (EWG) ---
    # EWG is a variant of greedy that picks the codeword maximizing the number of 
    # newly covered points, but with a specific tie-breaking or weight update.
    # Standard Greedy: pick max gain.
    # EWG: Often implies checking all candidates and picking the one that maximizes 
    #       (newly covered / cost) or similar. Since cost is 1, it's max gain.
    # The key difference from simple greedy in literature is often the handling of 
    # "potential" or running it multiple times with different orderings/weights.
    
    def run_ewg():
        covered = 0
        selected = []
        remaining = full_mask
        
        while remaining:
            best_gain = -1
            best_word = -1
            candidates = []
            
            # Optimization: We can skip words that are already "dominated" or 
            # have 0 gain. But computing gain is O(1) with bit_count.
            for w in range(total_space):
                # Gain is number of bits in (mask[w] & remaining)
                gain = (masks[w] & remaining).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    candidates = [w]
                elif gain == best_gain and gain > 0:
                    candidates.append(w)
                    
            if not candidates:
                break
                
            # Random tie-breaking to allow exploration in ILS
            best_word = random.choice(candidates)
            
            selected.append(best_word)
            covered |= masks[best_word]
            remaining = full_mask ^ covered
            
            if time.time() > deadline - 5:
                break
                
        return selected

    # --- Phase 2: Local Search (Polish) ---
    # We maintain an array `counts` where counts[x] is the number of codewords 
    # in `selected` that cover word x.
    # A codeword w is "redundant" if for all x in neighbors[w], counts[x] > 1.
    # A swap (w -> w') is valid if w' covers all x such that counts[x] == 1 and x in neighbors[w].
    
    def polish(selected):
        # Build counts
        counts = [0] * total_space
        for w in selected:
            for nb in neighbors[w]:
                counts[nb] += 1
        
        # Check validity (should be valid if input was valid)
        # But if we start from a heuristic, it's valid.
        
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            
            # 1. Remove redundant codewords
            # We iterate through a shuffled list of indices to avoid bias
            indices = list(range(len(selected)))
            random.shuffle(indices)
            
            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]
                removable = True
                for nb in neighbors[w]:
                    if counts[nb] == 1:
                        removable = False
                        break
                
                if removable:
                    for nb in neighbors[w]:
                        counts[nb] -= 1
                    selected.pop(idx)
                    changed = True
                    # After removal, we should restart the scan or at least be careful.
                    # Breaking and restarting the while loop is safer for correctness of indices.
                    # But for speed, we can just continue if we handle indices carefully.
                    # Actually, popping shifts indices. Let's restart the inner loop.
                    break
            
            if not changed:
                continue
                
            # 2. Try Swaps
            # We attempt to swap a codeword w with another w' to reduce size or maintain size
            # while potentially allowing further reductions.
            # Strategy: Pick a random w. Find its "unique" neighbors.
            # Look for w' that covers all uniques.
            # If found, swap.
            
            attempts = 0
            max_attempts = len(selected) * 2
            while attempts < max_attempts and time.time() < deadline - 5:
                attempts += 1
                idx = random.randint(0, len(selected) - 1)
                w = selected[idx]
                
                # Find unique neighbors of w
                uniques = []
                for nb in neighbors[w]:
                    if counts[nb] == 1:
                        uniques.append(nb)
                        
                if not uniques:
                    # Should have been removed, but handle just in case
                    for nb in neighbors[w]:
                        counts[nb] -= 1
                    selected.pop(idx)
                    changed = True
                    break
                    
                # Find a replacement w'
                # w' must cover all uniques.
                # We prefer w' that is NOT in selected (to keep size same) or IS in selected (to reduce size).
                # Actually, if w' is in selected, we are just removing w (since w' already covers uniques).
                # If w' is not in selected, we replace w with w' (size unchanged).
                
                found_replacement = False
                best_w_prime = -1
                
                # Check existing codewords first (potential reduction)
                for w2 in selected:
                    if w2 == w:
                        continue
                    # Check if w2 covers all uniques
                    # Optimization: use bitmask intersection?
                    # unique_mask = 0
                    # for u in uniques: unique_mask |= (1 << u)
                    # if (masks[w2] & unique_mask) == unique_mask: ...
                    
                    ok = True
                    for u in uniques:
                        if u not in neighbors[w2]:
                            ok = False
                            break
                    if ok:
                        best_w_prime = w2
                        found_replacement = True
                        break
                        
                if not found_replacement:
                    # Check all other words in the space
                    # Optimization: We can restrict search to words that are "close" to uniques?
                    # For now, scan all. 2187 is small.
                    for w2 in range(total_space):
                        if w2 == w:
                            continue
                        ok = True
                        for u in uniques:
                            if u not in neighbors[w2]:
                                ok = False
                                break
                        if ok:
                            best_w_prime = w2
                            found_replacement = True
                            break
                            
                if found_replacement:
                    # Perform swap
                    # Remove w
                    for nb in neighbors[w]:
                        counts[nb] -= 1
                    selected.pop(idx)
                    
                    # Add w' if it wasn't already in selected
                    if best_w_prime not in selected:
                        selected.append(best_w_prime)
                        for nb in neighbors[best_w_prime]:
                            counts[nb] += 1
                            
                    changed = True
                    # Restart search after a successful move
                    break
                    
            if not changed:
                break
                
        return selected

    # --- Phase 3: Iterated Local Search (ILS) ---
    # 1. Generate initial solution (EWG)
    # 2. Polish
    # 3. Record best
    # 4. Perturb (randomly swap/add/remove)
    # 5. Polish
    # 6. Repeat until time limit
    
    best_size = total_space
    best_selected = []
    
    # Run a few EWG restarts to find a good starting point
    num_initial_restarts = 10
    for i in range(num_initial_restarts):
        if time.time() > deadline - 10:
            break
        sol = run_ewg()
        polished = polish(sol)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]
            
    # ILS Loop
    while time.time() < deadline - 5:
        # Perturb the current best (or a recent solution)
        # Let's perturb the best solution found so far.
        current = best_selected[:]
        
        # Perturbation: Randomly swap k codewords
        # Or simply: pick a random codeword, replace with random other.
        # Or: remove a random codeword, add a random codeword.
        # To maintain validity during perturbation is hard, so we allow invalid intermediate states
        # and rely on Polish to fix it? No, Polish assumes valid input (counts > 0).
        # If we make it invalid, Polish might fail or behave unexpectedly.
        
        # Safer Perturbation: 
        # 1. Pick a random codeword w.
        # 2. Pick a random codeword w'.
        # 3. Replace w with w'. 
        # This keeps the size same and likely keeps coverage high, but might create holes.
        # If holes are created, Polish (which only removes/swaps) cannot ADD coverage.
        # So we need a Polish that can REPAIR invalid codes?
        # Or, we stick to "valid" perturbations:
        # Only swap w with w' if w' covers all uniques of w (like the swap in polish).
        # But that's just the polish step itself.
        
        # Alternative: "Add and Remove"
        # 1. Add a random codeword (size +1).
        # 2. Run Polish (which will try to remove redundant ones, potentially the new one or others).
        # This explores the neighborhood of size K+1 and K.
        
        if random.random() < 0.5:
            # Add a random word
            new_word = random.randint(0, total_space - 1)
            if new_word not in current:
                current.append(new_word)
        else:
            # Remove a random word? Risky for validity.
            # Let's stick to Add + Polish.
            pass
            
        # Run Polish on the perturbed solution
        # Note: Polish assumes valid input. Adding a word keeps it valid.
        # So `current` is valid.
        polished = polish(current)
        
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]
            
        # Occasionally, restart from a fresh EWG solution to escape local minima
        if random.random() < 0.1:
            sol = run_ewg()
            polished = polish(sol)
            if len(polished) < best_size:
                best_size = len(polished)
                best_selected = polished[:]
                
    # Final verification and conversion
    # Ensure we return a valid list
    # The polish function ensures validity if input is valid.
    # EWG ensures validity.
    # So best_selected should be valid.
    
    # Just in case, a final quick check?
    # For 2187 words, checking coverage is fast.
    # But we trust the logic.
    
    return [int_to_tuple(v) for v in best_selected]

# Fallback for generic cases if the specific logic fails or for different q,n
# The problem statement implies we should handle the specific K_3(7,1) well,
# but the interface is generic.
# The code above is optimized for small total_space.
# If total_space is huge, the bitmask approach fails.
# However, the task contract says "The evaluator passes this task's alphabet size q, length n and covering radius."
# and "Score: ... upper bound on K_q(n, radius)".
# Given the "Football pool" context and the specific mention of K_3(7,1),
# it is highly likely the evaluator will call construct(3, 7, 1).
# If called with other parameters, the bitmask approach might be too slow/memory heavy.
# We add a check.

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    total_space = q ** n
    
    if total_space > 20000:
        # Generic Greedy Set Cover without bitmasks (slower, but works for larger spaces)
        # Precompute neighbors as sets
        neighbors = [set() for _ in range(total_space)]
        powers = [1] * n
        for i in range(1, n):
            powers[i] = powers[i-1] * q
            
        for i in range(total_space):
            neighbors[i].add(i)
            for pos in range(n):
                digit_val = (i // powers[pos]) % q
                for new_val in range(q):
                    if new_val == digit_val: continue
                    diff = new_val - digit_val
                    neighbors[i].add(i + diff * powers[pos])
                    
        covered = set()
        selected = []
        remaining = set(range(total_space))
        
        while remaining:
            best_gain = -1
            best_word = -1
            for w in range(total_space):
                gain = len(neighbors[w] & remaining)
                if gain > best_gain:
                    best_gain = gain
                    best_word = w
            if best_word == -1: break
            selected.append(best_word)
            covered |= neighbors[best_word]
            remaining -= neighbors[best_word]
            
        def int_to_tuple(val):
            res = [0] * n
            for i in range(n):
                res[i] = (val // powers[i]) % q
            return res
            
        return [int_to_tuple(v) for v in selected]
        
    # Small space: Use the optimized Bitmask + ILS approach
    # (Code from previous block, slightly refactored to fit here)
    
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    neighbors = [[] for _ in range(total_space)]
    for i in range(total_space):
        neighbors[i].append(i)
        for pos in range(n):
            digit_val = (i // powers[pos]) % q
            for new_val in range(q):
                if new_val == digit_val:
                    continue
                diff = new_val - digit_val
                neighbors[i].append(i + diff * powers[pos])
                
    masks = [0] * total_space
    for i in range(total_space):
        m = 0
        for nb in neighbors[i]:
            m |= (1 << nb)
        masks[i] = m
        
    full_mask = (1 << total_space) - 1
    
    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
        return res

    def run_ewg():
        covered = 0
        selected = []
        remaining = full_mask
        
        while remaining:
            best_gain = -1
            candidates = []
            for w in range(total_space):
                gain = (masks[w] & remaining).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    candidates = [w]
                elif gain == best_gain and gain > 0:
                    candidates.append(w)
                    
            if not candidates:
                break
                
            best_word = random.choice(candidates)
            selected.append(best_word)
            covered |= masks[best_word]
            remaining = full_mask ^ covered
            
            if time.time() > deadline - 5:
                break
                
        return selected

    def polish(selected):
        counts = [0] * total_space
        for w in selected:
            for nb in neighbors[w]:
                counts[nb] += 1
        
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            
            indices = list(range(len(selected)))
            random.shuffle(indices)
            
            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]
                removable = True
                for nb in neighbors[w]:
                    if counts[nb] == 1:
                        removable = False
                        break
                
                if removable:
                    for nb in neighbors[w]:
                        counts[nb] -= 1
                    selected.pop(idx)
                    changed = True
                    break
            
            if not changed:
                continue
                
            attempts = 0
            max_attempts = len(selected) * 2
            while attempts < max_attempts and time.time() < deadline - 5:
                attempts += 1
                idx = random.randint(0, len(selected) - 1)
                w = selected[idx]
                
                uniques = []
                for nb in neighbors[w]:
                    if counts[nb] == 1:
                        uniques.append(nb)
                        
                if not uniques:
                    for nb in neighbors[w]:
                        counts[nb] -= 1
                    selected.pop(idx)
                    changed = True
                    break
                    
                found_replacement = False
                best_w_prime = -1
                
                for w2 in selected:
                    if w2 == w:
                        continue
                    ok = True
                    for u in uniques:
                        if u not in neighbors[w2]:
                            ok = False
                            break
                    if ok:
                        best_w_prime = w2
                        found_replacement = True
                        break
                        
                if not found_replacement:
                    for w2 in range(total_space):
                        if w2 == w:
                            continue
                        ok = True
                        for u in uniques:
                            if u not in neighbors[w2]:
                                ok = False
                                break
                        if ok:
                            best_w_prime = w2
                            found_replacement = True
                            break
                            
                if found_replacement:
                    for nb in neighbors[w]:
                        counts[nb] -= 1
                    selected.pop(idx)
                    
                    if best_w_prime not in selected:
                        selected.append(best_w_prime)
                        for nb in neighbors[best_w_prime]:
                            counts[nb] += 1
                            
                    changed = True
                    break
                    
            if not changed:
                break
                
        return selected

    best_size = total_space
    best_selected = []
    
    num_initial_restarts = 10
    for i in range(num_initial_restarts):
        if time.time() > deadline - 10:
            break
        sol = run_ewg()
        polished = polish(sol)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]
            
    while time.time() < deadline - 5:
        current = best_selected[:]
        
        if random.random() < 0.5:
            new_word = random.randint(0, total_space - 1)
            if new_word not in current:
                current.append(new_word)
                
        polished = polish(current)
        
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]
            
        if random.random() < 0.1:
            sol = run_ewg()
            polished = polish(sol)
            if len(polished) < best_size:
                best_size = len(polished)
                best_selected = polished[:]
                
    return [int_to_tuple(v) for v in best_selected]

