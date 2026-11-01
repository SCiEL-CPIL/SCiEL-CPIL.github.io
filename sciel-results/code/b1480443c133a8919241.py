import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    total_words = q ** n
    
    if total_words > 10000:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [[0] * n]

    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q
            
    def idx_to_tuple(idx):
        t = [0] * n
        rem = idx
        for i in range(n):
            t[i] = rem % q
            rem //= q
        return tuple(t)
        
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx += t[i] * powers[i]
        return idx

    # Precompute ball membership as a 2D boolean array: ball_matrix[c, w] = True if w is in ball(c)
    # Also store balls as lists for set operations
    balls = [None] * total_words
    ball_matrix = np.zeros((total_words, total_words), dtype=bool)
    
    for i in range(total_words):
        w = idx_to_tuple(i)
        ball_set = {i}
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    new_w = list(w)
                    new_w[pos] = new_val
                    new_idx = tuple_to_idx(tuple(new_w))
                    ball_set.add(new_idx)
        balls[i] = ball_set
        for w_idx in ball_set:
            ball_matrix[i, w_idx] = True

    def prune_cover(indices):
        indices = list(indices)
        cover_counts = np.zeros(total_words, dtype=np.int32)
        for idx in indices:
            cover_counts[np.where(ball_matrix[idx])[0]] += 1
        
        changed = True
        while changed:
            changed = False
            new_indices = []
            for idx in indices:
                ball_words = np.where(ball_matrix[idx])[0]
                if np.any(cover_counts[ball_words] == 1):
                    new_indices.append(idx)
                else:
                    cover_counts[ball_words] -= 1
                    changed = True
            indices = new_indices
        return indices

    def local_search(code_indices, time_limit):
        """Try to improve the code by swapping or removing codewords."""
        code_set = set(code_indices)
        cover_counts = np.zeros(total_words, dtype=np.int32)
        for idx in code_set:
            cover_counts[np.where(ball_matrix[idx])[0]] += 1
        
        improved = True
        while improved and (time.time() < time_limit):
            improved = False
            
            # Try removing any codeword that doesn't uniquely cover any word
            for idx in list(code_set):
                ball_words = np.where(ball_matrix[idx])[0]
                if not np.any(cover_counts[ball_words] == 1):
                    code_set.discard(idx)
                    cover_counts[ball_words] -= 1
                    improved = True
                    break
            
            # Try swapping: remove one, add another that covers more uncovered words
            if not improved and time.time() < time_limit:
                for remove_idx in list(code_set):
                    if time.time() >= time_limit:
                        break
                    # Temporarily remove
                    ball_remove = np.where(ball_matrix[remove_idx])[0]
                    cover_counts[ball_remove] -= 1
                    
                    # Find best addition
                    # Words that are now uncovered or were only covered by remove_idx
                    # Actually, we want to find a codeword that covers words with count 0 after removal
                    # But we need to be careful: words with count 0 after removal need to be covered
                    # by the new codeword.
                    
                    # Gain for adding codeword c = number of words w where cover_counts[w] == 0 
                    # (after removal) and ball_matrix[c, w] == True
                    # But we also need to make sure we don't lose coverage.
                    
                    # Simpler: just check if removing remove_idx still gives a valid cover
                    # If yes, we already handled it above.
                    # If no, try to find a replacement.
                    
                    uncovered_after_remove = np.where(cover_counts == 0)[0]
                    if len(uncovered_after_remove) > 0:
                        # Find codewords that cover the most of these uncovered words
                        # For each candidate c not in code_set, compute gain
                        # This is expensive, so sample candidates
                        candidates = random.sample(list(set(range(total_words)) - code_set), min(50, total_words - len(code_set)))
                        best_c = None
                        best_gain = -1
                        for c in candidates:
                            gain = np.sum(ball_matrix[c, uncovered_after_remove])
                            if gain > best_gain:
                                best_gain = gain
                                best_c = c
                        
                        if best_c is not None and best_gain == len(uncovered_after_remove):
                            # This codeword covers all previously uncovered words
                            code_set.discard(remove_idx)
                            code_set.add(best_c)
                            cover_counts[ball_remove] += 1  # restore
                            cover_counts[np.where(ball_matrix[best_c])[0]] += 1
                            improved = True
                            break
                        else:
                            # Restore
                            cover_counts[ball_remove] += 1
                    else:
                        # Should have been caught by removal check
                        cover_counts[ball_remove] += 1
        
        return list(code_set)

    best_code = None
    best_size = float('inf')
    
    # 1. Try Direct Product Construction (if applicable)
    if q == 3 and n >= 4 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        dp_indices = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                word = h + list(tail)
                dp_indices.append(tuple_to_idx(tuple(word)))
        dp_indices = prune_cover(dp_indices)
        if len(dp_indices) < best_size:
            best_size = len(dp_indices)
            best_code = dp_indices

    # 2. Try Randomized Greedy Multiple Times with vectorized gain computation
    num_restarts = 30
    
    for r in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
            
        uncovered = np.ones(total_words, dtype=bool)
        code_indices = []
        
        while np.any(uncovered):
            # Vectorized gain computation
            # gain[c] = number of uncovered words in ball(c)
            gains = np.sum(ball_matrix[:, uncovered], axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            top_candidates = np.where(gains == max_gain)[0]
            
            # Randomized selection
            if len(top_candidates) > 1 and random.random() < 0.3:
                # Pick from top few
                top_k = min(5, len(top_candidates))
                selected = random.sample(list(top_candidates), top_k)
                # Pick the one with highest gain among selected (they all have same gain)
                best_c = random.choice(selected)
            else:
                best_c = random.choice(list(top_candidates))
            
            code_indices.append(best_c)
            uncovered[ball_matrix[best_c]] = False
            
            if time.time() > deadline - 20.0:
                break
        
        if np.any(uncovered):
            continue
            
        # Prune the result
        pruned = prune_cover(code_indices)
        
        # Try local search to improve further
        local_time_limit = min(time.time() + 5.0, deadline - 10.0)
        if local_time_limit > time.time() + 1.0:
            pruned = local_search(pruned, local_time_limit)
        
        if len(pruned) < best_size:
            best_size = len(pruned)
            best_code = pruned

    if best_code is None:
        if q == 3 and n >= 4 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            dp_indices = []
            for h in tetracode:
                for tail in itertools.product(range(3), repeat=n - 4):
                    word = h + list(tail)
                    dp_indices.append(tuple_to_idx(tuple(word)))
            best_code = dp_indices
        else:
            best_code = list(range(total_words))

    result = [list(idx_to_tuple(c)) for c in best_code]
    
    return result
