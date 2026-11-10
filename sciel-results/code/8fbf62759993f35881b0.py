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
    # neighbors_set[w] is a set of integers (for O(1) lookup if needed, but we mostly iterate)
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

    # Precompute a numpy array for fast vectorized operations if needed
    # covers[c, w] is True if c covers w
    # This is a 2187x2187 boolean matrix. ~4.7 MB. Fast to create.
    covers = np.zeros((total_words, total_words), dtype=bool)
    for c in range(total_words):
        covers[c, neighbors_list[c]] = True

    def greedy_set_cover():
        covered = np.zeros(total_words, dtype=bool)
        uncovered_count = total_words
        codewords_list = []
        available = set(range(total_words))
        
        while uncovered_count > 0:
            best_candidate = -1
            best_score = -1
            # Randomize tie-breaking to explore different greedy paths
            candidates = list(available)
            random.shuffle(candidates)
            
            for c in candidates:
                # Count uncovered neighbors
                score = 0
                for w in neighbors_list[c]:
                    if not covered[w]:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_candidate = c
                    if best_score == 15:
                        break
            
            if best_candidate == -1:
                break
            codewords_list.append(best_candidate)
            available.remove(best_candidate)
            for w in neighbors_list[best_candidate]:
                if not covered[w]:
                    covered[w] = True
                    uncovered_count -= 1
        return codewords_list

    def local_search(code_list, time_limit):
        code_set = set(code_list)
        # cover_count[w] = number of codewords covering w
        cover_count = np.zeros(total_words, dtype=np.int16)
        for c in code_set:
            cover_count[neighbors_list[c]] += 1
            
        start_time = time.time()
        best_size = len(code_list)
        best_code = list(code_list)
        no_improvement_count = 0
        
        while time.time() < time_limit:
            improved = False
            
            # 1. Try Removals
            indices = list(range(len(code_list)))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() > time_limit:
                    break
                c = code_list[idx]
                # Check if any neighbor has cover_count 1
                # We can use numpy min on the slice
                if np.min(cover_count[neighbors_list[c]]) > 1:
                    # Remove
                    del code_list[idx]
                    code_set.remove(c)
                    cover_count[neighbors_list[c]] -= 1
                    improved = True
                    if len(code_list) < best_size:
                        best_size = len(code_list)
                        best_code = list(code_list)
                    no_improvement_count = 0
                    break # Restart scan
            
            if improved:
                continue
                
            # 2. Try Swaps
            if code_list:
                # Pick a random codeword to potentially swap out
                idx = random.randint(0, len(code_list) - 1)
                c = code_list[idx]
                
                # Find unique words (cover_count == 1) for c
                unique_words = [w for w in neighbors_list[c] if cover_count[w] == 1]
                
                if unique_words:
                    # Find candidate c' not in code_set that covers all unique_words
                    best_c_prime = -1
                    best_score = -1
                    
                    # To speed this up, we can pre-filter candidates?
                    # Or just scan all. 2187 is small.
                    # But checking `covers[c_prime, unique_words]` is fast with numpy.
                    
                    for c_prime in range(total_words):
                        if c_prime in code_set:
                            continue
                        
                        # Check if c_prime covers all unique_words
                        # unique_words is a list of ints. covers[c_prime] is a bool array.
                        if not np.all(covers[c_prime, unique_words]):
                            continue
                        
                        # Score the swap
                        # We want to increase the "redundancy" of other codewords.
                        # Words with cover_count 1 are "critical".
                        # If c_prime covers a word w (w not in unique_words) that has cover_count 1,
                        # then after swap, w will have cover_count 2 (since c_prime adds 1, and c removes 1? No, c doesn't cover w if w not in unique_words? Wait.)
                        
                        # Let's trace:
                        # Before swap:
                        # c covers unique_words. cover_count[w]=1 for w in unique_words.
                        # c covers other_words.
                        # Other codewords cover unique_words? No, count is 1, so only c covers them.
                        
                        # After swap (remove c, add c'):
                        # c' covers unique_words. cover_count[w] becomes 1 (was 1, -1 from c, +1 from c').
                        # For w in neighbors[c] \ unique_words:
                        #   cover_count[w] decreases by 1.
                        # For w in neighbors[c']:
                        #   cover_count[w] increases by 1.
                        
                        # We want to avoid creating new "critical" words (count 1) from the removal of c.
                        # Words in neighbors[c] \ unique_words with count 2 will become 1. This is BAD.
                        # So we should penalize c' if it doesn't cover those "at-risk" words?
                        # Or simply, we prefer c' that covers words with low cover_count to boost them.
                        
                        # Let's calculate the "damage" and "gain".
                        # Damage: words in neighbors[c] \ unique_words that have count 2. They will become 1.
                        # Gain: words in neighbors[c'] that have count 1 or 2. They will become 2 or 3.
                        
                        # Actually, a simpler heuristic:
                        # Score = (# of neighbors of c' with count 1) * 2 + (# of neighbors of c' with count 2) * 1
                        # Penalty = (# of neighbors of c (excluding unique) with count 2) * 2
                        # Net Score = Score - Penalty
                        
                        # Let's compute Penalty first
                        penalty = 0
                        for w in neighbors_list[c]:
                            if w not in unique_words and cover_count[w] == 2:
                                penalty += 2
                        
                        score = 0
                        for w in neighbors_list[c_prime]:
                            if cover_count[w] == 1:
                                score += 2
                            elif cover_count[w] == 2:
                                score += 1
                                
                        net_score = score - penalty
                        
                        if net_score > best_score:
                            best_score = net_score
                            best_c_prime = c_prime
                            if best_score > 10: # Good enough
                                break
                    
                    if best_c_prime != -1:
                        # Perform Swap
                        del code_list[idx]
                        code_set.remove(c)
                        cover_count[neighbors_list[c]] -= 1
                        
                        code_list.append(best_c_prime)
                        code_set.add(best_c_prime)
                        cover_count[neighbors_list[best_c_prime]] += 1
                        
                        improved = True
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                else:
                    # Should have been removed, but maybe floating point or logic error?
                    # If no unique words, it should be removable.
                    no_improvement_count += 1
            else:
                break
            
            # 3. Perturbation if stuck
            if no_improvement_count > 20 and code_list:
                # Remove a random few codewords
                num_to_remove = max(1, len(code_list) // 10)
                remove_indices = random.sample(range(len(code_list)), num_to_remove)
                remove_indices.sort(reverse=True)
                for idx in remove_indices:
                    c = code_list[idx]
                    del code_list[idx]
                    code_set.remove(c)
                    cover_count[neighbors_list[c]] -= 1
                
                # Re-fill greedily
                covered = (cover_count > 0)
                uncovered_count = np.sum(~covered)
                
                # Greedy fill
                available = set(range(total_words)) - code_set
                while uncovered_count > 0 and time.time() < time_limit:
                    best_candidate = -1
                    best_score = -1
                    candidates = list(available)
                    random.shuffle(candidates)
                    
                    for c in candidates:
                        score = 0
                        for w in neighbors_list[c]:
                            if not covered[w]:
                                score += 1
                        if score > best_score:
                            best_score = score
                            best_candidate = c
                            if best_score == 15:
                                break
                    
                    if best_candidate == -1:
                        break
                    code_list.append(best_candidate)
                    code_set.add(best_candidate)
                    for w in neighbors_list[best_candidate]:
                        if not covered[w]:
                            covered[w] = True
                            uncovered_count -= 1
                    cover_count[neighbors_list[best_candidate]] += 1
                    available.remove(best_candidate)
                
                no_improvement_count = 0

        return best_code

    # Execution
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Run multiple greedy passes and keep the best
    best_initial = None
    best_initial_size = float('inf')
    
    while time.time() < deadline - 10.0: # Leave 10s for local search on best
        initial_code = greedy_set_cover()
        if len(initial_code) < best_initial_size:
            best_initial_size = len(initial_code)
            best_initial = initial_code
            # If we find a very good one, we can break early?
            # 186 is the target. If we find < 190, it's good.
            if best_initial_size <= 190:
                break
        
    # 2. Local Search on the best initial code
    if best_initial is None:
        best_initial = greedy_set_cover()
        
    final_code = local_search(best_initial, deadline)
    
    # 3. Final Cleanup
    code_set = set(final_code)
    cover_count = np.zeros(total_words, dtype=np.int16)
    for c in code_set:
        cover_count[neighbors_list[c]] += 1
        
    final_list = list(code_set)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = list(range(len(final_list)))
        random.shuffle(indices)
        for idx in indices:
            if time.time() > deadline:
                break
            c = final_list[idx]
            if np.min(cover_count[neighbors_list[c]]) > 1:
                del final_list[idx]
                code_set.remove(c)
                cover_count[neighbors_list[c]] -= 1
                improved = True

    # Convert to list of lists
    result = []
    for c in final_list:
        digits = []
        w = c
        for _ in range(n):
            digits.append(w % q)
            w //= q
        result.append(digits[::-1])
        
    return result
