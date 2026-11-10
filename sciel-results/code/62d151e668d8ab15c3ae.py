# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, product

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin vs 150s kill

    if q == 3 and n == 7 and radius == 1:
        return _ternary7_r1(deadline)

    # Generic fallback: greedy set cover with NumPy
    return _generic_greedy(q, n, radius, deadline)


def _ternary7_r1(deadline):
    n = 7
    q = 3
    total = q ** n  # 2187

    # Precompute all words as (2187, 7) array
    all_words = np.array(np.unravel_index(np.arange(total), [3] * n)).T

    # Precompute ball_indices: for each codeword index c, the 15 words in its ball (dist <= 1)
    # Ball size = 1 + 2*7 = 15
    ball_indices = np.empty((total, 15), dtype=np.int32)
    for c in range(total):
        w = all_words[c]
        ball = [c]
        for i in range(n):
            orig = w[i]
            for new_val in range(q):
                if new_val != orig:
                    w2 = w.copy()
                    w2[i] = new_val
                    idx = int(np.dot(w2, [3 ** (n - 1 - j) for j in range(n)]))
                    ball.append(idx)
        ball_indices[c] = ball

    # Greedy set cover with vectorized marginal gain
    covered = np.zeros(total, dtype=bool)
    cover_count = np.zeros(total, dtype=np.int32)
    chosen = []
    
    while True:
        if time.time() > deadline - 30:  # leave time for local search
            break
        if covered.all():
            break
        
        if len(chosen) == 0:
            # First codeword: all have gain 15, pick index 0
            chosen.append(0)
            cover_count[ball_indices[0]] += 1
            covered[ball_indices[0]] = True
            continue
        
        # Compute marginal gain for each unchosen codeword
        ball_cov = covered[ball_indices]  # (2187, 15)
        gains = (1 - ball_cov).sum(axis=1)  # (2187,)
        
        # Mask out already chosen
        for c in chosen:
            gains[c] = -1
        
        best = int(np.argmax(gains))
        if gains[best] <= 0:
            break
        
        chosen.append(best)
        cover_count[ball_indices[best]] += 1
        covered[ball_indices[best]] = True
    
    # Local search polish: remove redundant codewords, then try 1-swap improvements
    if time.time() < deadline - 5:
        chosen = _local_search_polish(chosen, ball_indices, cover_count, total, deadline)
    
    # Convert to list of codewords
    result = [all_words[c].tolist() for c in chosen]
    return result


def _local_search_polish(chosen, ball_indices, cover_count, total, deadline):
    chosen_list = list(chosen)
    chosen_set = set(chosen_list)
    
    # Phase 1: Remove redundant codewords
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(len(chosen_list)):
            if time.time() > deadline - 5:
                break
            c = chosen_list[i]
            ball = ball_indices[c]
            if np.all(cover_count[ball] > 1):
                # Safe to remove
                chosen_list.pop(i)
                chosen_set.discard(c)
                cover_count[ball] -= 1
                improved = True
                break  # restart scan since indices changed
    
    # Phase 2: Try 1-swap moves: remove codeword i, add codeword j
    # This can reduce size if j covers more new words than i lost
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(chosen_list)):
                if time.time() > deadline - 5:
                    break
                c_remove = chosen_list[i]
                ball_remove = ball_indices[c_remove]
                
                # Check which words would become uncovered if we remove c_remove
                # A word w in ball_remove becomes uncovered if cover_count[w] == 1
                uncovered_if_removed = ball_remove[cover_count[ball_remove] == 1]
                
                if len(uncovered_if_removed) == 0:
                    # Redundant, should have been caught in phase 1
                    # But double-check
                    if np.all(cover_count[ball_remove] > 1):
                        chosen_list.pop(i)
                        chosen_set.discard(c_remove)
                        cover_count[ball_remove] -= 1
                        improved = True
                        break
                    continue
                
                # Try to find a codeword j not in chosen that covers all uncovered_if_removed
                # and doesn't add too much overhead
                # For each candidate j, compute:
                #   new_gain = number of words in ball(j) that are currently covered by c_remove 
                #              but not by any other chosen codeword
                #   Actually simpler: after removing c_remove, how many of its uncovered words
                #   does j cover?
                
                # We want: for some j not in chosen_set, 
                #   all words in uncovered_if_removed are in ball(j)
                # AND the net effect is we can remove c_remove and add j, keeping size same or less
                # Since we're doing 1-swap, size stays same. We want to see if this helps
                # future removals. But for direct improvement, we need to find if there's
                # a j such that after swap, some codeword becomes redundant.
                
                # Simpler approach: just try all j and see if the resulting cover allows
                # a removal. This is expensive, so limit it.
                
                if time.time() > deadline - 8:
                    break
                
                # For efficiency, only try a subset of candidates
                # Compute for all j: how many of uncovered_if_removed are in ball(j)
                # We need this count == len(uncovered_if_removed) for j to be a valid replacement
                
                # Vectorized: for each j, count how many words in uncovered_if_removed
                # are in ball(j)
                # ball_indices is (2187, 15), uncovered_if_removed is a set of indices
                # For each j, we need: |ball_indices[j] ∩ uncovered_if_removed| == len(uncovered_if_removed)
                
                # Convert uncovered_if_removed to a boolean mask
                unc_mask = np.zeros(total, dtype=bool)
                unc_mask[uncovered_if_removed] = True
                
                # For each j, count = sum(unc_mask[ball_indices[j]])
                # This is (2187, 15) -> (2187,)
                # But we only care about j not in chosen_set
                
                # Compute for all j
                counts = (unc_mask[ball_indices]).sum(axis=1)  # (2187,)
                
                # Valid j: counts[j] == len(uncovered_if_removed) and j not in chosen_set
                valid_j = np.where((counts == len(uncovered_if_removed)))[0]
                
                for j in valid_j:
                    j = int(j)
                    if j in chosen_set:
                        continue
                    if j == c_remove:
                        continue
                    
                    # Try the swap: remove c_remove, add j
                    # First, simulate: what's the state after swap?
                    # cover_count after removing c_remove: cover_count[ball_remove] -= 1
                    # Then adding j: cover_count[ball_indices[j]] += 1
                    # Check if any codeword becomes redundant
                    
                    ball_add = ball_indices[j]
                    
                    # New cover count would be:
                    # For words in ball_remove only: old_count - 1
                    # For words in ball_add only: old_count + 1  
                    # For words in both: old_count - 1 + 1 = old_count
                    # For words in neither: old_count
                    
                    # A codeword k is redundant if all words in ball(k) have new_count > 1
                    # i.e., for all w in ball(k), new_count[w] > 1
                    
                    # new_count[w] = cover_count[w] + (1 if w in ball_add else 0) - (1 if w in ball_remove else 0)
                    
                    # This is expensive to check for all k. Let's just do the swap and check.
                    
                    # Actually, let's just do the swap and run a quick redundancy check
                    # on a few codewords
                    
                    # Perform swap
                    cover_count[ball_remove] -= 1
                    cover_count[ball_add] += 1
                    chosen_set.discard(c_remove)
                    chosen_set.add(j)
                    chosen_list[i] = j
                    
                    # Check if any codeword is now redundant
                    found_redundant = False
                    for k_idx in range(len(chosen_list)):
                        if time.time() > deadline - 8:
                            break
                        k = chosen_list[k_idx]
                        ball_k = ball_indices[k]
                        if np.all(cover_count[ball_k] > 1):
                            # Remove k
                            chosen_list.pop(k_idx)
                            chosen_set.discard(k)
                            cover_count[ball_k] -= 1
                            found_redundant = True
                            improved = True
                            break
                    
                    if not found_redundant:
                        # Revert the swap
                        cover_count[ball_add] -= 1
                        cover_count[ball_remove] += 1
                        chosen_set.discard(j)
                        chosen_set.add(c_remove)
                        chosen_list[i] = c_remove
                        # Don't break, try next i
                    else:
                        break  # Break inner loop, continue outer
    
    # Phase 3: Final redundancy pass
    if time.time() < deadline - 3:
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for i in range(len(chosen_list)):
                if time.time() > deadline - 3:
                    break
                c = chosen_list[i]
                ball = ball_indices[c]
                if np.all(cover_count[ball] > 1):
                    chosen_list.pop(i)
                    chosen_set.discard(c)
                    cover_count[ball] -= 1
                    improved = True
                    break
    
    return chosen_list


def _generic_greedy(q, n, radius, deadline):
    total = q ** n
    all_words = np.array(np.unravel_index(np.arange(total), [q] * n)).T
    covered = np.zeros(total, dtype=bool)
    chosen = []
    
    while not covered.all() and time.time() < deadline - 5:
        best_gain = -1
        best_c = -1
        chosen_set = set(chosen)
        for c in range(total):
            if c in chosen_set:
                continue
            ball = _ball_generic(c, all_words[c], q, n, radius)
            gain = sum(1 for w in ball if not covered[w])
            if gain > best_gain:
                best_gain = gain
                best_c = c
        if best_c == -1 or best_gain <= 0:
            break
        chosen.append(best_c)
        ball = _ball_generic(best_c, all_words[best_c], q, n, radius)
        for w in ball:
            covered[w] = True
    
    return [all_words[c].tolist() for c in chosen]


def _ball_generic(c_idx, word, q, n, radius):
    ball = [c_idx]
    for r in range(1, radius + 1):
        for positions in combinations(range(n), r):
            for values in product(range(q), repeat=r):
                w2 = list(word)
                for pos, val in zip(positions, values):
                    if val != w2[pos]:
                        w2[pos] = val
                idx = int(np.dot(w2, [q ** (n - 1 - j) for j in range(n)]))
                ball.append(idx)
    return ball
# EVOLVE-BLOCK-END
