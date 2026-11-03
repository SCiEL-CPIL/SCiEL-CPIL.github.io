import time
import random
import math
import sys

def construct(q, n, radius):
    start_time = time.monotonic()
    deadline = start_time + 115.0  # Safety margin for 120s/150s limit
    
    total = q ** n
    rng = random.Random(42)
    
    # Precompute neighbors (words within Hamming distance radius)
    # For radius 1, this is 1 + n*(q-1)
    # Each neighbor is an index 0..total-1
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
    
    # Convert to sets for fast intersection/union checks if needed, 
    # but lists are better for iteration in gain calculation.
    # We will use lists for gain, sets for membership in code.
    
    def greedy_init(seed):
        """Fast greedy set cover initialization."""
        r = random.Random(seed)
        counts = [0] * total
        code = []
        code_set = set()
        remaining = [True] * total
        rem_count = total
        
        # Precompute list of all indices for random selection
        all_indices = list(range(total))
        
        while rem_count > 0:
            if time.monotonic() > deadline - 5.0:
                break
            
            best_c = -1
            best_gain = -1
            
            # Find candidate with max new coverage
            # Optimization: Iterate through all candidates. 
            # N=729, K=13. 729*13 = 9477 ops per pass. 
            # We do this ~73 times. 73 * 9477 ~ 700k ops. Very fast.
            for c in range(total):
                if c in code_set:
                    continue
                gain = 0
                for u in neigh[c]:
                    if remaining[u]:
                        gain += 1
                        if gain > best_gain:
                            # Can't break early unless we know max possible gain
                            # Max gain is len(neigh[c]) = 13.
                            # If gain == 13, we can break inner loop
                            if gain == len(neigh[c]):
                                break
                
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            
            if best_c == -1 or best_gain == 0:
                # Fallback: pick any uncovered word's neighbor
                for u in range(total):
                    if remaining[u]:
                        for nc in neigh[u]:
                            if nc not in code_set:
                                best_c = nc
                                break
                        if best_c != -1:
                            break
                if best_c == -1:
                    break
            
            code.append(best_c)
            code_set.add(best_c)
            
            for u in neigh[best_c]:
                counts[u] += 1
                if counts[u] == 1:
                    remaining[u] = False
                    rem_count -= 1
        
        return code, counts

    def polish(code, counts, time_budget):
        """Local search to minimize code size, maintaining validity."""
        code_set = set(code)
        best_code = code[:]
        best_size = len(code)
        end_time = time.monotonic() + time_budget
        
        # Precompute neighbor sets for O(1) membership checks in swap validation
        neigh_sets = [set(neigh[i]) for i in range(total)]
        
        def is_removable(c):
            """Check if codeword c can be removed without losing coverage."""
            for u in neigh[c]:
                if counts[u] == 1:
                    return False
            return True

        # Phase 1: Greedy Removal
        changed = True
        while changed and time.monotonic() < end_time:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                if is_removable(code[i]):
                    c = code.pop(i)
                    code_set.discard(c)
                    for u in neigh[c]:
                        counts[u] -= 1
                    changed = True
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = code[:]
        
        # Phase 2: Local Search (Swap and 2-for-1) maintaining validity
        # We use a simple loop with random moves, only accepting if valid.
        # If we get stuck, we can try random restarts within the polish or just rely on multiple restarts in main.
        
        while time.monotonic() < end_time:
            move_type = rng.random()
            made_change = False
            
            if move_type < 0.4:
                # Try Remove
                if len(code) > 0:
                    idx = rng.randint(0, len(code) - 1)
                    c = code[idx]
                    if is_removable(c):
                        code.pop(idx)
                        code_set.discard(c)
                        for u in neigh[c]:
                            counts[u] -= 1
                        made_change = True
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code[:]
                            
            elif move_type < 0.8:
                # Try Swap (1-for-1) - Must remain valid
                if len(code) > 0:
                    idx = rng.randint(0, len(code) - 1)
                    old_c = code[idx]
                    # Choose new_c
                    # Bias towards neighbors of old_c to maintain structure
                    if rng.random() < 0.7:
                        new_c = neigh[old_c][rng.randint(1, len(neigh[old_c])-1)]
                    else:
                        new_c = rng.randint(0, total - 1)
                    
                    if new_c == old_c or new_c in code_set:
                        continue
                    
                    # Check validity: Does removing old_c and adding new_c leave any word uncovered?
                    # A word u is uncovered if counts[u] == 1 (only covered by old_c) and u not in neigh[new_c]
                    valid = True
                    new_c_set = neigh_sets[new_c]
                    for u in neigh[old_c]:
                        if counts[u] == 1 and u not in new_c_set:
                            valid = False
                            break
                    
                    if valid:
                        # Accept
                        for u in neigh[old_c]:
                            counts[u] -= 1
                        for u in neigh[new_c]:
                            counts[u] += 1
                        code[idx] = new_c
                        code_set.discard(old_c)
                        code_set.add(new_c)
                        made_change = True
                        # Swap doesn't change size, but might enable removals
                        # Check if any codeword is now removable
                        for i in range(len(code) - 1, -1, -1):
                            if is_removable(code[i]):
                                c = code.pop(i)
                                code_set.discard(c)
                                for u in neigh[c]:
                                    counts[u] -= 1
                                if len(code) < best_size:
                                    best_size = len(code)
                                    best_code = code[:]
                                break # Restart removal check

            else:
                # Try 2-for-1 Move (Remove 2, Add 1) - Must remain valid
                if len(code) >= 2:
                    i1 = rng.randint(0, len(code) - 1)
                    i2 = rng.randint(0, len(code) - 1)
                    if i1 == i2:
                        continue
                    
                    c1 = code[i1]
                    c2 = code[i2]
                    
                    # Find a codeword c_new that covers all critical words of c1 and c2
                    critical_words = set()
                    for u in neigh[c1]:
                        if counts[u] == 1:
                            critical_words.add(u)
                    for u in neigh[c2]:
                        if counts[u] == 1:
                            critical_words.add(u)
                            
                    if not critical_words:
                        # Both are removable individually
                        if is_removable(c1):
                            code.remove(c1)
                            code_set.discard(c1)
                            for u in neigh[c1]: counts[u] -= 1
                            if len(code) < best_size:
                                best_size = len(code)
                                best_code = code[:]
                        if c2 in code and is_removable(c2):
                            code.remove(c2)
                            code_set.discard(c2)
                            for u in neigh[c2]: counts[u] -= 1
                            if len(code) < best_size:
                                best_size = len(code)
                                best_code = code[:]
                        made_change = True
                        continue
                    
                    # Candidate c_new must cover all critical_words
                    crit_list = list(critical_words)
                    sample_u = rng.choice(crit_list)
                    
                    found_new_c = -1
                    for cand in neigh[sample_u]:
                        if cand in code_set:
                            continue
                        # Check if cand covers all critical_words
                        covers_all = True
                        cand_set = neigh_sets[cand]
                        for w in critical_words:
                            if w not in cand_set:
                                covers_all = False
                                break
                        if covers_all:
                            found_new_c = cand
                            break
                    
                    if found_new_c != -1:
                        # Perform the move: Remove c1, c2, Add found_new_c
                        # Check validity
                        valid = True
                        affected = set(neigh[c1]) | set(neigh[c2]) | set(neigh[found_new_c])
                        
                        for u in affected:
                            new_count = counts[u]
                            if u in neigh[c1]: new_count -= 1
                            if u in neigh[c2]: new_count -= 1
                            if u in neigh[found_new_c]: new_count += 1
                            
                            if new_count < 0:
                                valid = False
                                break
                            
                        if valid:
                            # Apply move
                            code.remove(c1)
                            code_set.discard(c1)
                            for u in neigh[c1]: counts[u] -= 1
                            if c2 in code:
                                code.remove(c2)
                                code_set.discard(c2)
                                for u in neigh[c2]: counts[u] -= 1
                            code.append(found_new_c)
                            code_set.add(found_new_c)
                            for u in neigh[found_new_c]: counts[u] += 1
                            
                            if len(code) < best_size:
                                best_size = len(code)
                                best_code = code[:]
                            made_change = True

            if not made_change and rng.random() < 0.01:
                # Random restart of local search state? No, just continue.
                # If stuck, the main loop restarts will handle it.
                pass

        # Final removal pass
        changed = True
        while changed and time.monotonic() < end_time:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                if is_removable(code[i]):
                    c = code.pop(i)
                    code_set.discard(c)
                    for u in neigh[c]:
                        counts[u] -= 1
                    changed = True
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
        
        return best_code, best_size

    # Multi-restart strategy
    overall_best_code = None
    overall_best_size = total + 1
    
    # Determine number of restarts
    # Each restart: Greedy (~0.1s) + Polish (~5-10s)
    num_restarts = 15
    
    for restart in range(num_restarts):
        if time.monotonic() > deadline - 15.0:
            break
        
        # Use different seeds
        seed = 42 + restart * 13
        code, counts = greedy_init(seed)
        
        if not code:
            continue
        
        # Polish
        remaining_time = deadline - time.monotonic()
        if remaining_time < 15.0:
            break
            
        # Allocate time based on remaining restarts
        # Give more time to later restarts? Or equal?
        # Let's give equal time, but ensure we don't exceed deadline
        time_per_polish = min(8.0, (remaining_time - 5.0) / (num_restarts - restart))
        
        polished_code, polished_size = polish(code, counts, time_per_polish)
        
        if polished_size < overall_best_size:
            overall_best_size = polished_size
            overall_best_code = polished_code[:]
            
            # If we hit the theoretical lower bound or known best (73), we can stop early?
            # Lower bound is 71. Known best is 73.
            # If we find 73, we can try a few more to see if we find 72 or 71, 
            # but it's unlikely. However, the task asks to minimize.
            # We will continue to search for 72/71 if time permits, 
            # but 73 is a very strong result.
            if overall_best_size <= 73:
                # We can reduce time for subsequent restarts or break if we are confident.
                # Let's just let the loop continue with tighter time budgets.
                pass

    # If we haven't found anything, do one more greedy
    if overall_best_code is None:
        code, counts = greedy_init(42)
        overall_best_code = code
        overall_best_size = len(code)
    
    # Repair the code to ensure validity
    # Re-calculate counts for overall_best_code
    final_counts = [0] * total
    final_code_set = set(overall_best_code)
    for c in overall_best_code:
        for u in neigh[c]:
            final_counts[u] += 1
            
    # Check for uncovered words and add codewords to cover them
    for u in range(total):
        if final_counts[u] == 0:
            # Find best codeword to cover u
            best_c = -1
            best_gain = -1
            for cand in neigh[u]:
                if cand in final_code_set:
                    continue
                gain = 0
                for v in neigh[cand]:
                    if final_counts[v] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_c = cand
            
            if best_c != -1:
                overall_best_code.append(best_c)
                final_code_set.add(best_c)
                for v in neigh[best_c]:
                    final_counts[v] += 1
            else:
                # Should not happen if neigh is correct, but fallback
                overall_best_code.append(u)
                final_code_set.add(u)
                for v in neigh[u]:
                    final_counts[v] += 1

    # Convert to words
    result = []
    for idx in overall_best_code:
        word = [0] * n
        temp_idx = idx
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        result.append(word)
    
    return result
