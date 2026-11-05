import numpy as np
import random
import time

def construct(q, n, radius):
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin for 150s limit

    # For the specific case of q=3, n=7, radius=1, we use specialized code
    if q == 3 and n == 7 and radius == 1:
        N = 3 ** n  # 2187
        ball_size = 1 + n * (q - 1)  # 15

        def int_to_base3(num, length):
            res = [0] * length
            for i in range(length - 1, -1, -1):
                res[i] = num % 3
                num //= 3
            return res

        def base3_to_int(digits):
            res = 0
            for d in digits:
                res = res * 3 + d
            return res

        # Precompute balls as a NumPy array for vectorized operations
        # balls[i] contains the indices of the ball centered at i
        balls = np.zeros((N, ball_size), dtype=np.int32)
        for i in range(N):
            digits = int_to_base3(i, n)
            neighbors = [i]
            for pos in range(n):
                original = digits[pos]
                for new_val in range(q):
                    if new_val != original:
                        new_digits = digits[:]
                        new_digits[pos] = new_val
                        neighbors.append(base3_to_int(new_digits))
            balls[i] = neighbors

        # Precompute cover_map: for each word w, which codewords cover it
        # This is useful for efficient updates if we were doing list-based, 
        # but with NumPy arrays for cover_count, we can just do:
        # cover_count[balls[c]] -= 1
        # So we don't strictly need cover_map for the vectorized approach, 
        # unless we want to optimize specific checks. 
        # Let's stick to the vectorized ball updates.

        def greedy_cover(seed):
            """Vectorized Greedy Set Cover."""
            rng = random.Random(seed)
            uncovered = np.ones(N, dtype=bool)
            code = []
            
            while np.any(uncovered):
                if time.time() > deadline - 5:
                    break
                
                # Calculate gain for all potential codewords
                # gains[i] = number of uncovered words in ball(i)
                gains = np.sum(uncovered[balls], axis=1)
                
                max_gain = np.max(gains)
                if max_gain == 0:
                    break # Should not happen if uncovered exists and balls are valid
                
                candidates = np.where(gains == max_gain)[0]
                c = candidates[rng.randint(0, len(candidates) - 1)]
                
                code.append(c)
                uncovered[balls[c]] = False

            return code

        def refine(code):
            """Remove redundant codewords."""
            if not code:
                return code

            cover_count = np.zeros(N, dtype=np.int32)
            for c in code:
                cover_count[balls[c]] += 1

            code_list = list(code)
            changed = True
            while changed:
                changed = False
                # Shuffle to break symmetry? Or just iterate.
                # Iterating is fine for small lists.
                for i in range(len(code_list)):
                    c = code_list[i]
                    # If min cover count in ball(c) is > 1, c is redundant
                    if np.min(cover_count[balls[c]]) > 1:
                        cover_count[balls[c]] -= 1
                        code_list.pop(i)
                        changed = True
                        break
            return code_list

        def local_search_kick(code, seed, time_limit):
            """Local search with kick-and-repair moves."""
            rng = random.Random(seed)
            code_list = list(code)
            if not code_list:
                return code_list

            cover_count = np.zeros(N, dtype=np.int32)
            for c in code_list:
                cover_count[balls[c]] += 1

            improved = True
            while improved and time.time() < time_limit:
                improved = False
                # Try kick sizes 1, 2, 3
                # Randomize order of kick sizes? Or start with 1?
                # Start with 1 (standard deletion/swap) then 2, 3.
                
                # We will try a few random kicks per iteration
                for _ in range(10):
                    if time.time() > time_limit:
                        break
                    
                    k = rng.choice([1, 1, 1, 2, 2, 3]) # Weight towards 1
                    if len(code_list) < k:
                        continue

                    # Select k distinct indices to remove
                    indices = rng.sample(range(len(code_list)), k)
                    removed = [code_list[i] for i in indices]
                    
                    # Temporarily remove
                    for c in removed:
                        cover_count[balls[c]] -= 1
                    
                    # Check for uncovered
                    uncovered_mask = (cover_count == 0)
                    num_uncovered = np.sum(uncovered_mask)
                    
                    if num_uncovered == 0:
                        # Perfect removal
                        for i in sorted(indices, reverse=True):
                            code_list.pop(i)
                        improved = True
                        break

                    # Try to repair with < k new codewords
                    # Greedy repair on the uncovered set
                    still_unc = uncovered_mask.copy()
                    added = []
                    added_set = set()
                    
                    while np.any(still_unc) and len(added) < k:
                        # Calculate gains for all candidates
                        # We only care about covering 'still_unc'
                        # gains[i] = number of still_unc words in ball(i)
                        gains = np.sum(still_unc[balls], axis=1)
                        
                        # Exclude already added or already in code?
                        # We can add any codeword, but if it's in code_list, 
                        # it would have been counted in cover_count. 
                        # Wait, if we remove 'removed', and then add 'c' which is in code_list \ removed,
                        # it's effectively a swap.
                        # If 'c' is not in code_list, it's an addition.
                        # The greedy repair just finds best cover for holes.
                        
                        max_gain = np.max(gains)
                        if max_gain == 0:
                            break # Cannot cover remaining holes
                        
                        candidates = np.where(gains == max_gain)[0]
                        # Avoid picking a codeword that was just removed? 
                        # If we pick one, it's a swap. That's fine.
                        # Avoid picking one already in 'added'?
                        # If we pick one already in 'added', gain would be 0 for those words 
                        # if we updated still_unc. We do update still_unc.
                        
                        c = candidates[rng.randint(0, len(candidates) - 1)]
                        
                        # Check if c is already in code_list (and not removed)
                        # If c is in code_list, it's a swap.
                        # If c is not in code_list, it's an addition.
                        # We don't strictly need to distinguish for the count, 
                        # but we need to update state correctly.
                        
                        added.append(c)
                        added_set.add(c)
                        
                        # Update cover_count and still_unc
                        cover_count[balls[c]] += 1
                        still_unc[balls[c]] = False # Words covered by c are no longer "still uncovered"
                        # Note: still_unc should be (cover_count == 0) & original_uncovered_mask?
                        # Actually, just setting to False where we cover is enough 
                        # because we only care about the specific holes we created.
                        # But wait, if c covers a word that wasn't a hole, that's fine.
                        # The condition `while np.any(still_unc)` ensures we stop when all holes are filled.

                    if len(added) < k:
                        # Successful kick
                        # Remove the 'removed' codewords from code_list
                        # Note: 'removed' are values, not indices. 
                        # We need to remove them from the list.
                        # Since we popped by index earlier? No, we just identified indices.
                        # We need to actually remove them from code_list.
                        
                        # Re-identify indices of 'removed' in current code_list?
                        # It's safer to work with indices directly.
                        
                        # Let's redo the removal logic to be robust.
                        # We have 'indices' (list of indices in code_list)
                        # We have 'added' (list of new codeword values)
                        
                        # 1. Remove codewords at 'indices'
                        # Sort indices descending to pop correctly
                        for i in sorted(indices, reverse=True):
                            code_list.pop(i)
                        
                        # 2. Add 'added' codewords
                        for c in added:
                            code_list.append(c)
                        
                        improved = True
                        break
                    else:
                        # Repair failed (needed >= k codewords)
                        # Restore cover_count for 'added'
                        for c in added:
                            cover_count[balls[c]] -= 1
                        # 'removed' are still subtracted, which is correct for the next iteration 
                        # or we need to restore them if we are rejecting the move.
                        # Since we are rejecting, we must restore 'removed' too.
                        for c in removed:
                            cover_count[balls[c]] += 1

            return code_list

        best_code = None
        best_len = float('inf')

        # Multiple restarts
        # The greedy is fast with NumPy. Let's do more restarts.
        max_restarts = 100
        for i in range(max_restarts):
            if time.time() > deadline - 20:
                break
            code = greedy_cover(i)
            refined = refine(code)
            if len(refined) < best_len:
                best_len = len(refined)
                best_code = refined
            # Early stop if we hit the theoretical lower bound or known best
            if best_len <= 186:
                # We can try to improve further or just break if we want to save time
                # But 186 is the best known. Let's see if we can go lower (unlikely).
                # Let's keep searching a bit more to be sure.
                pass

        if best_code is not None:
            # Local search on the best code found
            # Use a time limit for local search
            ls_time_limit = deadline - 5
            if time.time() < ls_time_limit - 5:
                ls_code = local_search_kick(best_code, 42, ls_time_limit)
                if len(ls_code) < best_len:
                    best_len = len(ls_code)
                    best_code = ls_code
            
            # Second round with different seed
            if time.time() < ls_time_limit - 5:
                ls_code2 = local_search_kick(best_code, 123, ls_time_limit)
                if len(ls_code2) < best_len:
                    best_len = len(ls_code2)
                    best_code = ls_code2

            final_code = refine(best_code)
        else:
            final_code = []

        result = []
        for c in final_code:
            result.append(int_to_base3(c, n))
        return result

    # Generic case (shouldn't be called for this task, but included for completeness)
    N = q ** n
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i - 1] * q

    def idx_to_word(idx):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n - 1 - pos]) % q
        return w

    nb_count = 1 + n * (q - 1)
    neighbors = np.empty((N, nb_count), dtype=np.int32)
    for idx in range(N):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n - 1 - pos]) % q
        nb_list = [idx]
        for pos in range(n):
            for new_d in range(q):
                if new_d == w[pos]:
                    continue
                delta = (new_d - w[pos]) * q_powers[n - 1 - pos]
                nb_list.append(idx + delta)
        neighbors[idx] = nb_list

    # Generic greedy
    uncovered = np.ones(N, dtype=bool)
    code = []
    while np.any(uncovered):
        if time.time() > deadline:
            break
        gains = np.sum(uncovered[neighbors], axis=1)
        max_gain = np.max(gains)
        if max_gain == 0:
            break
        candidates = np.where(gains == max_gain)[0]
        best_idx = candidates[0] # Deterministic for generic, or random
        code.append(best_idx)
        uncovered[neighbors[best_idx]] = False

    return [idx_to_word(i) for i in code]
