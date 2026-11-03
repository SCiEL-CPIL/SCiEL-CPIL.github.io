import time
import random
import math
import heapq

def _precompute_neighbors(q, n, radius):
    total = q ** n
    # For radius 1, neighbors are the word itself and words differing in 1 position
    # We store neighbors as a list of indices
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
    return neigh, total

def construct(q, n, radius):
    deadline = time.monotonic() + 115.0 # Safety margin
    total = q ** n
    rng = random.Random(42)
    
    # 1. Precompute neighbors
    neigh, _ = _precompute_neighbors(q, n, radius)
    
    # State variables
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

    def remove_from_code(c):
        nonlocal rem_count
        code.remove(c)
        code_set.remove(c)
        for u in neigh[c]:
            counts[u] -= 1
            if counts[u] == 0:
                remaining[u] = True
                rem_count += 1

    # 2. Greedy Initialization
    # We want to cover all 'remaining' words.
    # Gain of adding c is number of remaining words in neigh[c].
    while rem_count > 0:
        if time.monotonic() > deadline:
            break
        best_c = -1
        best_gain = -1
        # Optimization: Only check candidates that might have high gain.
        # A simple full scan is 729 * (1 + 15) ops ~ 11k ops per step. 729 steps ~ 8M ops. Fast enough.
        for c in range(total):
            gain = 0
            for u in neigh[c]:
                if remaining[u]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_c = c
        if best_c == -1 or best_gain == 0:
            # Should not happen if rem_count > 0
            break
        add_to_code(best_c)
        
    if rem_count > 0:
        # Fallback: brute force add
        for c in range(total):
            if rem_count == 0: break
            if c not in code_set:
                add_to_code(c)

    best_code = code[:]
    best_size = len(code)
    
    # 3. Local Search
    # We will run local search until deadline.
    # Moves:
    # 1. Try to remove a codeword. If valid, do it. (Strict improvement)
    # 2. Try to swap a codeword c with d such that the new code is valid. 
    #    Ideally, we want to find a state where we can remove more words.
    #    A common effective move is "2-opt": remove c, add d, if valid, keep. 
    #    But we also want to reduce size.
    
    # Let's use a strategy:
    # Loop:
    #   1. Attempt deletions. If any successful, reset search intensity.
    #   2. If no deletion possible, perform a perturbation (kick) and repair.
    
    stagnation_counter = 0
    max_stagnation = 5000 # Number of moves without improvement before kick
    
    while time.monotonic() < deadline:
        improved = False
        
        # Phase 1: Try deletions
        # Iterate through code, try to remove.
        # To be fast, we can just pick a random index and try to remove.
        # Or iterate all. Iterating all is 73 * 16 checks ~ 1k ops. Very fast.
        # Let's iterate all to ensure we catch all possible deletions.
        i = 0
        while i < len(code):
            c = code[i]
            # Check if removable
            removable = True
            for u in neigh[c]:
                if counts[u] == 1:
                    removable = False
                    break
            
            if removable:
                remove_from_code(c)
                improved = True
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code[:]
                # After removal, other words might become removable, so restart scan
                i = 0
            else:
                i += 1
                
        if improved:
            stagnation_counter = 0
            continue
            
        # Phase 2: No deletions possible. We are in a local minimum for deletions.
        # Perform a kick.
        stagnation_counter += 1
        if stagnation_counter > max_stagnation or rng.random() < 0.1:
            # Kick: Remove 1 or 2 random words, then repair greedily.
            # Save state
            saved_code = code[:]
            saved_counts = counts[:]
            saved_remaining = remaining[:]
            saved_rem_count = rem_count
            saved_size = len(code)
            saved_code_set = set(code)
            
            # Remove k random words
            k = 1 if rng.random() < 0.7 else 2
            if len(code) > k:
                indices_to_remove = rng.sample(range(len(code)), k)
                # Remove in reverse order to keep indices valid
                for idx in sorted(indices_to_remove, reverse=True):
                    c = code[idx]
                    code.pop(idx)
                    code_set.remove(c)
                    for u in neigh[c]:
                        counts[u] -= 1
                        if counts[u] == 0:
                            remaining[u] = True
                            rem_count += 1
                
                # Repair: Add words to cover remaining
                # Use greedy repair
                max_repair_steps = 10
                repair_steps = 0
                while rem_count > 0 and repair_steps < max_repair_steps:
                    best_c = -1
                    best_gain = -1
                    for c in range(total):
                        if c in code_set:
                            continue
                        gain = 0
                        for u in neigh[c]:
                            if remaining[u]:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_c = c
                    if best_c == -1:
                        break
                    add_to_code(best_c)
                    repair_steps += 1
                
                # Evaluate
                if rem_count == 0:
                    # Valid code
                    if len(code) < saved_size:
                        # Improvement
                        if len(code) < best_size:
                            best_size = len(code)
                            best_code = code[:]
                        stagnation_counter = 0
                    else:
                        # Same size or worse. 
                        # If same size, we might accept to explore, but usually we revert if not improved 
                        # unless we want to shuffle the local minimum.
                        # Let's accept same-size moves with some probability to escape local minima.
                        if len(code) == saved_size and rng.random() < 0.2:
                            # Accept
                            stagnation_counter = 0
                        else:
                            # Revert
                            code = saved_code
                            code_set = saved_code_set
                            counts = saved_counts
                            remaining = saved_remaining
                            rem_count = saved_rem_count
                else:
                    # Failed to repair, revert
                    code = saved_code
                    code_set = saved_code_set
                    counts = saved_counts
                    remaining = saved_remaining
                    rem_count = saved_rem_count

    # Convert best_code indices to words
    result = []
    for idx in best_code:
        word = [0] * n
        temp_idx = idx
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        result.append(word)
        
    return result
