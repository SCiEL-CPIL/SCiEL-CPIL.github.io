import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    """
    Constructs a covering code using:
    1. Greedy Set Cover
    2. Local Search with Critical Position Optimization
    """
    start_time = time.time()
    timeout = 118  # Safety margin from 120s limit
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other cases
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return [list(word) for word in itertools.product(range(q), repeat=n)]
    
    total_words = 3 ** 7
    n_words = total_words
    
    # Precompute digits and balls
    digits = np.zeros((n_words, n), dtype=int)
    for i in range(n_words):
        temp = i
        for k in range(n):
            digits[i, k] = temp % 3
            temp //= 3
            
    # Precompute balls as lists of indices
    balls = []
    powers3 = [1, 3, 9, 27, 81, 243, 729]
    for i in range(n_words):
        ball = [i]
        for pos in range(n):
            orig_digit = digits[i, pos]
            p = powers3[pos]
            base_val = i - orig_digit * p
            for new_digit in range(3):
                if new_digit == orig_digit:
                    continue
                j = base_val + new_digit * p
                ball.append(j)
        balls.append(ball)
        
    # Precompute ball sets for faster membership testing
    ball_sets = [set(b) for b in balls]

    def greedy_cover(seed):
        """Run greedy set cover."""
        rng = random.Random(seed)
        uncovered = np.ones(n_words, dtype=bool)
        code = []
        
        while np.any(uncovered):
            if time.time() - start_time > timeout:
                break
            
            best_idx = -1
            max_cov = 0
            best_candidates = []
            
            for i in range(n_words):
                cov = 0
                for j in balls[i]:
                    if uncovered[j]:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    best_candidates = [i]
                    if max_cov == 15:
                        break
                elif cov == max_cov and max_cov > 0:
                    best_candidates.append(i)
            
            if not best_candidates:
                break
            best_idx = rng.choice(best_candidates)
            code.append(best_idx)
            for j in balls[best_idx]:
                uncovered[j] = False
                
        return code

    def local_search(code, seed, time_budget):
        """
        Local search using critical positions.
        """
        if time.time() - start_time > timeout:
            return code
            
        count = np.zeros(n_words, dtype=int)
        for c in code:
            for j in balls[c]:
                count[j] += 1
                
        code_set = set(code)
        rng = random.Random(seed)
        
        def try_remove():
            """Try to remove a redundant codeword. Returns True if removed."""
            nonlocal code, count, code_set
            # Find a codeword that is redundant
            # A codeword c is redundant if for all j in balls[c], count[j] > 1
            # i.e., critical set is empty
            removable_indices = []
            for idx, c in enumerate(code):
                # Check if any neighbor has count 1
                redundant = True
                for j in balls[c]:
                    if count[j] == 1:
                        redundant = False
                        break
                if redundant:
                    removable_indices.append(idx)
            
            if removable_indices:
                idx = rng.choice(removable_indices)
                c = code[idx]
                # Remove it
                for j in balls[c]:
                    count[j] -= 1
                code.pop(idx)
                code_set.discard(c)
                return True
            return False

        def try_swap():
            """Try to swap a codeword with another to maintain cover and potentially enable removals."""
            nonlocal code, count, code_set
            if not code:
                return False
            
            c_idx = rng.randint(0, len(code) - 1)
            c = code[c_idx]
            
            # Find critical positions of c: j in balls[c] where count[j] == 1
            critical = []
            for j in balls[c]:
                if count[j] == 1:
                    critical.append(j)
            
            # If no critical positions, we could just remove c, but try_remove handles that.
            # Here we assume try_remove failed, so critical is not empty.
            # We need to find w not in code such that critical is subset of balls[w].
            
            # Try a few random w's
            for _ in range(20):
                w = rng.randint(0, n_words - 1)
                if w == c or w in code_set:
                    continue
                
                # Check if critical is subset of balls[w]
                valid = True
                ball_w_set = ball_sets[w]
                for j in critical:
                    if j not in ball_w_set:
                        valid = False
                        break
                
                if valid:
                    # Perform swap
                    for j in balls[c]:
                        count[j] -= 1
                    for j in balls[w]:
                        count[j] += 1
                    code[c_idx] = w
                    code_set.discard(c)
                    code_set.add(w)
                    return True
            return False

        def try_add_remove():
            """Add a random word, then try to remove any word."""
            nonlocal code, count, code_set
            old_code = list(code)
            old_count = count.copy()
            old_set = set(code_set)
            
            w = rng.randint(0, n_words - 1)
            if w in code_set:
                return False
                
            # Add w
            code.append(w)
            code_set.add(w)
            for j in balls[w]:
                count[j] += 1
            
            # Try to remove any word
            removed = False
            # We try to remove multiple times
            for _ in range(10):
                if not try_remove():
                    break
            removed = len(code) < len(old_code)
            
            if removed:
                return True
            else:
                # Revert
                code = old_code
                count = old_count
                code_set = old_set
                return False

        # Main loop
        while time.time() - start_time < time_budget:
            # 1. Try to remove
            if try_remove():
                continue
            
            # 2. Try to swap
            if try_swap():
                continue
            
            # 3. Try add-remove
            if try_add_remove():
                continue
            
            # If nothing worked, we are stuck.
            # Maybe try a random restart of the local search phase?
            # Or just break if we've been stuck for a while?
            # For now, just keep trying random swaps/adds.
            
        return code

    best_code = None
    best_size = float('inf')
    
    num_trials = 20
    for trial in range(num_trials):
        if time.time() - start_time > timeout:
            break
            
        seed = 100 + trial
        code = greedy_cover(seed)
        
        if len(code) < best_size:
            best_code = list(code)
            best_size = len(code)
            
        # Run local search with remaining time for this trial
        # Allocate time based on remaining total time
        remaining = timeout - (time.time() - start_time)
        if remaining > 5:
            # Give each trial a portion of the time, but allow the best one to get more?
            # Simple strategy: give each trial 5 seconds, and if we have more, do more trials or extend last one.
            # Better: dynamic allocation.
            # Let's just run local search until time is up, but keep track of best.
            # Actually, if we run local search on a trial, it might take all the time.
            # We should limit local search per trial.
            trial_time = min(remaining, 5.0) # 5 seconds per trial max
            code = local_search(code, seed, start_time + trial_time)
            
            if len(code) < best_size:
                best_code = list(code)
                best_size = len(code)
                
    # Final polish on best code with remaining time
    remaining = timeout - (time.time() - start_time)
    if best_code is not None and remaining > 2:
        best_code = local_search(best_code, 99999, timeout - 1)
        best_size = len(best_code)
        
    if best_code is None:
        best_code = list(range(n_words))
        
    result = [list(digits[c]) for c in best_code]
    return result
