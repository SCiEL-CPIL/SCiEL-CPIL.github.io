import time
import random
import sys
import numpy as np

def construct(q, n, radius):
    # Task: q=3, n=7, radius=1
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters (not expected in this task)
        # Simple greedy set cover
        total_words = q ** n
        def int_to_tuple(w):
            digits = []
            for _ in range(n):
                digits.append(w % q)
                w //= q
            return tuple(reversed(digits))
        
        def tuple_to_int(t):
            w = 0
            mult = 1
            for d in reversed(t):
                w += d * mult
                mult *= q
            return w

        neighbors = [None] * total_words
        for w in range(total_words):
            t = int_to_tuple(w)
            ball = {w}
            for i in range(n):
                orig = t[i]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    new_t = list(t)
                    new_t[i] = new_val
                    ball.add(tuple_to_int(tuple(new_t)))
            neighbors[w] = list(ball)
        
        covered = [False] * total_words
        uncovered_count = total_words
        codewords_list = []
        available_candidates = set(range(total_words))
        
        while uncovered_count > 0:
            best_candidate = -1
            best_score = -1
            for c in available_candidates:
                score = 0
                for w in neighbors[c]:
                    if not covered[w]:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_candidate = c
            if best_candidate == -1:
                break
            codewords_list.append(best_candidate)
            available_candidates.remove(best_candidate)
            for w in neighbors[best_candidate]:
                if not covered[w]:
                    covered[w] = True
                    uncovered_count -= 1
        
        result = [list(int_to_tuple(c)) for c in codewords_list]
        return result

    # Specialized solver for q=3, n=7, r=1
    total_words = 2187
    
    # Precompute neighbors
    powers = [1, 3, 9, 27, 81, 243, 729]
    
    # neighbors_list[w] is a list of integers
    neighbors_list = [None] * total_words
    for w in range(total_words):
        ball = [w]
        for i in range(n):
            p = powers[i]
            digit = (w // p) % 3
            if digit != 0:
                ball.append(w - digit * p)
            if digit != 1:
                ball.append(w + (1 - digit) * p)
            if digit != 2:
                ball.append(w + (2 - digit) * p)
        neighbors_list[w] = ball
        
    # Precompute covers matrix: covers[c, w] is True if c covers w
    # This allows fast vectorized checks: covers[c_prime, unique_words]
    covers = np.zeros((total_words, total_words), dtype=bool)
    for c in range(total_words):
        covers[c, neighbors_list[c]] = True

    def greedy_set_cover(random_seed=0):
        rng = random.Random(random_seed)
        covered = np.zeros(total_words, dtype=bool)
        uncovered_count = total_words
        codewords_list = []
        
        # To speed up, we can just iterate all candidates each time?
        # 2187 candidates * 180 steps * 15 neighbors = ~6M ops. Fast.
        # But we can remove candidates that are fully covered? No, they might cover new stuff later?
        # No, if a candidate's ball is fully covered, it can never add new coverage.
        # So we can maintain a list of active candidates.
        
        active = list(range(total_words))
        rng.shuffle(active)
        
        while uncovered_count > 0 and active:
            best_candidate = -1
            best_score = 0
            
            # Iterate through active candidates
            # We can't easily remove from list while iterating efficiently in pure python loop without index management
            # Let's just iterate all active.
            for c in active:
                # Count uncovered neighbors
                # neighbors_list[c] is a list of ints.
                # covered is a numpy bool array.
                # score = np.sum(~covered[neighbors_list[c]])
                # This is vectorized and fast.
                score = np.sum(~covered[neighbors_list[c]])
                
                if score > best_score:
                    best_score = score
                    best_candidate = c
                    if best_score == 15:
                        break
            
            if best_candidate == -1 or best_score == 0:
                break
                
            codewords_list.append(best_candidate)
            active.remove(best_candidate)
            
            # Update covered
            new_cov = ~covered[neighbors_list[best_candidate]]
            # We need to update uncovered_count
            # Actually, just update covered array
            covered[neighbors_list[best_candidate]] = True
            uncovered_count -= np.sum(new_cov)
            
        return codewords_list

    def local_search(code_list, time_limit, random_seed=0):
        rng = random.Random(random_seed)
        code_set = set(code_list)
        code_list = list(code_list)
        
        # Initialize cover_count
        cover_count = np.zeros(total_words, dtype=np.int8)
        for c in code_set:
            cover_count[neighbors_list[c]] += 1
            
        start_time = time.time()
        best_size = len(code_list)
        best_code = list(code_list)
        
        # We will alternate between removals and swaps.
        # To allow more search, we will try to be fast.
        
        while time.time() < time_limit:
            improved = False
            
            # 1. Try Removals
            # Shuffle to avoid bias
            indices = list(range(len(code_list)))
            rng.shuffle(indices)
            
            for idx in indices:
                if time.time() > time_limit:
                    break
                c = code_list[idx]
                # Check if any neighbor has cover_count 1
                # min_count = np.min(cover_count[neighbors_list[c]])
                # If min_count > 1, c is redundant.
                if np.min(cover_count[neighbors_list[c]]) > 1:
                    # Remove
                    del code_list[idx]
                    code_set.remove(c)
                    cover_count[neighbors_list[c]] -= 1
                    improved = True
                    if len(code_list) < best_size:
                        best_size = len(code_list)
                        best_code = list(code_list)
                    break # Restart scan after removal
            
            if improved:
                continue
                
            # 2. Try Swaps
            if not code_list:
                break
                
            # Pick a random codeword to try to swap out
            idx = rng.randint(0, len(code_list) - 1)
            c = code_list[idx]
            
            # Find unique words (cover_count == 1) for c
            # unique_words = [w for w in neighbors_list[c] if cover_count[w] == 1]
            # Vectorized:
            mask = (cover_count[neighbors_list[c]] == 1)
            unique_words = neighbors_list[c][mask]
            
            if len(unique_words) == 0:
                # Should have been removed, but maybe timing?
                # Force removal?
                del code_list[idx]
                code_set.remove(c)
                cover_count[neighbors_list[c]] -= 1
                improved = True
                if len(code_list) < best_size:
                    best_size = len(code_list)
                    best_code = list(code_list)
                continue

            # Find candidate c' not in code_set that covers all unique_words
            # And score it.
            # Score heuristic: We want c' to cover words that are "critical" for OTHER codewords.
            # i.e., words w where cover_count[w] == 1 and w is NOT in unique_words.
            # If c' covers such a w, then after swap, cover_count[w] becomes 2, and the other codeword covering w becomes less critical.
            
            # Pre-filter candidates?
            # A candidate c' must cover all w in unique_words.
            # We can check this with: np.all(covers[c_prime, unique_words])
            
            best_c_prime = -1
            best_score = -1
            
            # Iterate over all non-codewords?
            # 2187 - len(code_list) ~ 2000 candidates.
            # For each, check coverage of unique_words (size <= 15).
            # Then score.
            
            # To speed up, we can sample? No, 2000 * 15 is small.
            # Let's iterate all.
            
            # Optimization: Instead of iterating all 2187, we can just iterate all and skip if in code_set.
            # code_set is a set, so `in` is O(1).
            
            for c_prime in range(total_words):
                if c_prime in code_set:
                    continue
                
                # Check if c_prime covers all unique_words
                # covers[c_prime] is a bool array.
                # np.all(covers[c_prime, unique_words])
                if not np.all(covers[c_prime, unique_words]):
                    continue
                
                # Calculate Score
                # We want to maximize the number of "savings" or "flexibility".
                # Savings: c' covers words w (not in unique_words) that have cover_count == 1.
                # Let's get neighbors of c_prime.
                # n_prime = neighbors_list[c_prime]
                # We want to count how many w in n_prime have cover_count[w] == 1 AND w is NOT in unique_words.
                # Note: w in unique_words has cover_count 1, but it's "handled" by the swap requirement.
                # Actually, for w in unique_words, cover_count is 1 (from c). After swap, it's 1 (from c'). No change in count, but c is gone.
                # For w NOT in unique_words:
                # If cover_count[w] == 1, it's covered by some other c_other.
                # If c' also covers w, then after swap, cover_count[w] becomes 2.
                # This makes c_other potentially removable.
                # So we score +1 for each such w.
                
                # Also, if c' covers words with cover_count 0? Impossible, c' is a valid word, its neighbors are covered?
                # No, c' is not in the code, so its neighbors might have cover_count 0?
                # No, the code covers ALL words. So all words have cover_count >= 1.
                
                # Let's vectorize the score calculation.
                # n_prime = neighbors_list[c_prime]
                # counts = cover_count[n_prime]
                # We want to sum(1 for w in n_prime if cover_count[w]==1 and w not in unique_words)
                
                # unique_words is a small list/array.
                # We can create a mask for unique_words?
                # Or just subtract the count of unique_words that have cover_count 1 (which is all of them) from the total count of cover_count==1 in n_prime.
                
                total_ones = np.sum(cover_count[neighbors_list[c_prime]] == 1)
                # Subtract the ones that are in unique_words?
                # unique_words are a subset of neighbors_list[c] (the one being removed).
                # Are they also in neighbors_list[c_prime]?
                # Yes, because c_prime must cover all unique_words.
                # So all unique_words are in neighbors_list[c_prime].
                # And they all have cover_count 1.
                # So the number of "other" words with cover_count 1 is:
                # total_ones - len(unique_words)
                
                score = total_ones - len(unique_words)
                
                # Additional heuristic: Prefer c' that is "far" from other codewords?
                # Or just stick to this.
                
                if score > best_score:
                    best_score = score
                    best_c_prime = c_prime
                    if best_score > 10: # Good enough, break early?
                         # Maybe not, let's find the best.
                         pass

            if best_c_prime != -1:
                # Perform Swap
                del code_list[idx]
                code_set.remove(c)
                cover_count[neighbors_list[c]] -= 1
                
                code_list.append(best_c_prime)
                code_set.add(best_c_prime)
                cover_count[neighbors_list[best_c_prime]] += 1
                
                improved = True
                # We don't update best_code here unless size decreased?
                # Swap doesn't change size. But it might enable removals later.
                # So we don't update best_size immediately.
            else:
                # No valid swap found for this c.
                # We could try a different c, but to keep it simple, we just continue the loop.
                # If we get stuck, we might want to randomize the choice of idx more aggressively.
                pass

        # Final cleanup: try removals one more time
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            indices = list(range(len(code_list)))
            rng.shuffle(indices)
            for idx in indices:
                if time.time() > time_limit:
                    break
                c = code_list[idx]
                if np.min(cover_count[neighbors_list[c]]) > 1:
                    del code_list[idx]
                    code_set.remove(c)
                    cover_count[neighbors_list[c]] -= 1
                    improved = True
                    if len(code_list) < best_size:
                        best_size = len(code_list)
                        best_code = list(code_list)
                    break

        return best_code

    # Execution
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    best_overall_code = None
    best_overall_size = float('inf')
    
    # Run multiple iterations with different seeds to escape local minima
    # Greedy is fast (~0.1s), Local Search takes the rest.
    # Let's run 5 restarts if time permits.
    
    num_restarts = 5
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # 1. Greedy
        initial_code = greedy_set_cover(random_seed=i)
        
        # 2. Local Search
        # Allocate time for local search, leaving some buffer
        ls_deadline = min(deadline - 2.0, time.time() + time_per_restart)
        
        final_code = local_search(initial_code, ls_deadline, random_seed=i)
        
        # Update best
        if len(final_code) < best_overall_size:
            best_overall_size = len(final_code)
            best_overall_code = final_code
            
        # If we found a very small code (e.g., < 170), we can stop early?
        # No, let's use all time.
        
    if best_overall_code is None:
        # Fallback: just greedy
        best_overall_code = greedy_set_cover(random_seed=0)

    # Convert to list of lists
    result = []
    for c in best_overall_code:
        digits = []
        w = c
        for _ in range(n):
            digits.append(w % q)
            w //= q
        result.append(digits[::-1])
        
    return result

