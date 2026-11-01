import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 118.0  # Safety margin for 120s limit
    
    total = q ** n
    ball_size = 1 + n * (q - 1)
    
    # Precompute place values for index conversion
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q
        
    # Precompute all words as indices
    # words[i] is the list of digits for index i
    words = np.zeros((total, n), dtype=np.int32)
    for i in range(total):
        temp = i
        for j in range(n - 1, -1, -1):
            words[i, j] = temp % q
            temp //= q
            
    # Precompute ball_indices
    # ball_indices[i] is a list of indices of words within distance 1 of i
    ball_indices = [[] for _ in range(total)]
    for i in range(total):
        w = words[i]
        ball_indices[i].append(i)
        for coord in range(n):
            orig = w[coord]
            for v in range(q):
                if v == orig:
                    continue
                new_idx = i + (v - orig) * base[coord]
                ball_indices[i].append(new_idx)
                
    # Convert ball_indices to numpy arrays for faster iteration? 
    # Lists are usually fine for small N (15 elements).
    
    def greedy_build():
        """Greedy set cover initialization."""
        covered = np.zeros(total, dtype=bool)
        code = []
        # To add diversity, we can randomize the order of candidates or weights
        # Standard greedy: pick set with max new coverage
        while not np.all(covered):
            best_c = -1
            max_new = -1
            for c in range(total):
                # Optimization: if c is already in code, skip? 
                # Actually, greedy set cover doesn't add duplicates.
                # We can maintain a list of uncovered points to speed up?
                # For now, simple loop.
                new_cov = 0
                for x in ball_indices[c]:
                    if not covered[x]:
                        new_cov += 1
                if new_cov > max_new:
                    max_new = new_cov
                    best_c = c
                    if max_new == ball_size:
                        break # Can't get better
            if best_c == -1:
                break # Should not happen if radius is sufficient
            code.append(best_c)
            for x in ball_indices[best_c]:
                covered[x] = True
        return code

    def run_delete_phase(code_set, cnt):
        """Removes redundant codewords. Returns number of removed."""
        removed = 0
        # We iterate until no more removals are possible
        changed = True
        while changed:
            changed = False
            for c in list(code_set):
                # Check if min count in ball(c) is >= 2
                min_val = 255
                for x in ball_indices[c]:
                    if cnt[x] < min_val:
                        min_val = cnt[x]
                if min_val >= 2:
                    code_set.remove(c)
                    for x in ball_indices[c]:
                        cnt[x] -= 1
                    removed += 1
                    changed = True
        return removed

    def local_search(code_set, cnt, max_iterations=10000):
        """
        Local search with 1-for-1 swaps and deletions.
        Returns the best code found.
        """
        best_size = len(code_set)
        best_code = list(code_set)
        
        # Precompute list of all words for random selection
        all_words = list(range(total))
        
        for _ in range(max_iterations):
            if time.time() > deadline - 2.0:
                break
                
            # 1. Delete Phase
            run_delete_phase(code_set, cnt)
            if len(code_set) < best_size:
                best_size = len(code_set)
                best_code = list(code_set)
                
            # 2. Swap Phase
            # Try random swaps
            found_improvement = False
            # Limit swaps per iteration to avoid getting stuck in one spot
            for _ in range(50): 
                if time.time() > deadline - 2.0:
                    break
                if len(code_set) == 0:
                    break
                
                c = random.choice(list(code_set))
                c_prime = random.choice(all_words)
                
                if c_prime == c or c_prime in code_set:
                    continue
                
                # Simulate swap
                # We need to check if (code_set - {c} + {c_prime}) is valid.
                # Efficient check:
                # 1. Decrement counts for c
                # 2. Increment counts for c_prime
                # 3. Check min > 0
                # 4. Revert if invalid
                
                # Copy cnt? 2187 ints is small.
                # Or do in-place and revert.
                # In-place is faster.
                
                # Save state of affected points
                # Points in Ball(c) or Ball(c_prime)
                # We just need to check if any point in the union drops to 0.
                
                # Let's do a dry run on a small array or just check specific points.
                # Actually, we can just update cnt, check, and revert.
                
                # Update
                for x in ball_indices[c]:
                    cnt[x] -= 1
                for x in ball_indices[c_prime]:
                    cnt[x] += 1
                    
                # Check validity
                # We only need to check points in Ball(c) that might have dropped to 0.
                # Points in Ball(c_prime) only increase, so they are safe.
                valid = True
                for x in ball_indices[c]:
                    if cnt[x] == 0:
                        valid = False
                        break
                        
                if valid:
                    # Accept swap
                    code_set.remove(c)
                    code_set.add(c_prime)
                    found_improvement = True
                    # After a swap, run delete phase immediately
                    run_delete_phase(code_set, cnt)
                    if len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = list(code_set)
                else:
                    # Revert
                    for x in ball_indices[c]:
                        cnt[x] += 1
                    for x in ball_indices[c_prime]:
                        cnt[x] -= 1
                        
            if not found_improvement:
                # If no improvement, try perturbation
                # Randomly add a few codewords and re-optimize
                if random.random() < 0.1:
                    # Perturbation: add 3 random codewords
                    for _ in range(3):
                        c_rand = random.choice(all_words)
                        if c_rand not in code_set:
                            code_set.add(c_rand)
                            for x in ball_indices[c_rand]:
                                cnt[x] += 1
                    # Run delete phase
                    run_delete_phase(code_set, cnt)
                    # If it's worse, revert? 
                    # Actually, let's just let the local search handle it.
                    # If it doesn't improve, it will likely be removed by swaps/deletions later.
                    # But to be safe, if size increases significantly, we might want to revert.
                    # Let's keep it simple: the local search will try to reduce it.

        return best_code, best_size

    # Initial Greedy
    current_code = set(greedy_build())
    
    # Initialize counts
    cnt = np.zeros(total, dtype=np.int32)
    for c in current_code:
        for x in ball_indices[c]:
            cnt[x] += 1
            
    best_code = list(current_code)
    best_size = len(current_code)
    
    # Run Local Search
    # We can run multiple local searches with different perturbations
    while time.time() < deadline - 5.0:
        # Perturb the current solution if we are stuck?
        # The local_search function above includes perturbation.
        # Let's just run it.
        
        # To escape local minima, we can randomly modify the current solution before passing to local_search
        # Or just rely on the internal perturbation.
        
        # Let's try a few restarts of local search from the current best
        # Or just run one long local search with perturbation.
        
        # The local_search function returns the best found within its run.
        # We should update global best.
        
        # Let's run local_search on the current state
        # But local_search modifies code_set and cnt in place.
        # We need to be careful if we want to restart from best_code.
        
        # Strategy:
        # 1. Start with greedy.
        # 2. Run local search.
        # 3. If stuck, perturb and run again.
        
        # The local_search function above does perturbation internally.
        # Let's call it.
        
        # However, local_search might take too long if max_iterations is high.
        # Let's set max_iterations based on time.
        
        # Let's just run the loop with perturbation inside.
        
        # Actually, the previous code structure was better:
        # Main loop:
        #   Delete
        #   Swap
        #   Perturb if stuck
        
        # Let's implement a tighter main loop.
        
        pass # We will implement the main loop below

    # Let's re-implement the main loop to be more efficient and controllable
    
    # Reset to best code found so far? 
    # No, let's just run the local search loop directly.
    
    # Re-initialize cnt from best_code to be safe
    cnt[:] = 0
    for c in best_code:
        for x in ball_indices[c]:
            cnt[x] += 1
            
    current_code = set(best_code)
    
    # Main Search Loop
    no_improve_count = 0
    last_best_size = best_size
    
    while time.time() < deadline - 5.0:
        # 1. Delete Phase
        run_delete_phase(current_code, cnt)
        
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = list(current_code)
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # 2. Swap Phase
        # Try a few random swaps
        swaps_attempted = 0
        max_swaps = 100
        while swaps_attempted < max_swaps and time.time() < deadline - 5.0:
            swaps_attempted += 1
            if len(current_code) == 0:
                break
                
            # Pick random codeword to remove
            c = random.choice(list(current_code))
            
            # Pick random candidate to add
            c_prime = random.randint(0, total - 1)
            
            if c_prime == c:
                continue
            if c_prime in current_code:
                continue
                
            # Simulate swap
            # Decrement c, Increment c_prime
            for x in ball_indices[c]:
                cnt[x] -= 1
            for x in ball_indices[c_prime]:
                cnt[x] += 1
                
            # Check validity: min count in Ball(c) must be > 0
            # (Points in Ball(c_prime) only increase)
            valid = True
            for x in ball_indices[c]:
                if cnt[x] == 0:
                    valid = False
                    break
            
            if valid:
                # Accept
                current_code.remove(c)
                current_code.add(c_prime)
                
                # Run delete phase immediately
                run_delete_phase(current_code, cnt)
                
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = list(current_code)
                    no_improve_count = 0
            else:
                # Revert
                for x in ball_indices[c]:
                    cnt[x] += 1
                for x in ball_indices[c_prime]:
                    cnt[x] -= 1
                    
        # 3. Perturbation if stuck
        if no_improve_count > 20:
            # Randomly add 2-5 codewords
            num_add = random.randint(2, 5)
            for _ in range(num_add):
                c_rand = random.randint(0, total - 1)
                if c_rand not in current_code:
                    current_code.add(c_rand)
                    for x in ball_indices[c_rand]:
                        cnt[x] += 1
            # Run delete phase
            run_delete_phase(current_code, cnt)
            no_improve_count = 0

    # Final result conversion
    result = []
    for idx in best_code:
        w = []
        temp = idx
        for i in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)
        
    return result
