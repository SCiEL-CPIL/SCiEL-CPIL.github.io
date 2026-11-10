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
    powers = [1, 3, 9, 27, 81, 243, 729]
    
    # Precompute neighbors as lists and sets
    neighbors_list = [None] * total_words
    neighbors_set = [None] * total_words
    
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
        neighbors_set[w] = set(ball)
        
    # covers[c, w] is True if c covers w
    # We can use a list of sets for faster "is in" checks, but numpy is good for counts.
    # Let's stick to python sets for logic and numpy for bulk counting if needed.
    # Actually, for 2187 words, python sets are very fast.
    
    def greedy_set_cover():
        covered = [False] * total_words
        uncovered_count = total_words
        codewords_list = []
        available = set(range(total_words))
        
        while uncovered_count > 0:
            best_candidate = -1
            best_score = -1
            # Heuristic: prioritize candidates covering many uncovered words
            # Random shuffle to break ties and diversify
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
                    if best_score == 15: # Max possible
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
        cover_count = [0] * total_words
        for c in code_set:
            for w in neighbors_list[c]:
                cover_count[w] += 1
        
        start_time = time.time()
        best_size = len(code_list)
        best_code = list(code_list)
        no_improvement = 0
        
        while time.time() < time_limit:
            improved = False
            
            # 1. Reduction Phase: Remove redundant codewords
            # Repeat until no more can be removed
            changed = True
            while changed:
                if time.time() > time_limit:
                    return best_code
                changed = False
                # Shuffle to avoid bias
                indices = list(range(len(code_list)))
                random.shuffle(indices)
                for idx in indices:
                    c = code_list[idx]
                    # Check if any neighbor has cover_count 1
                    # If all neighbors have cover_count > 1, c is redundant
                    redundant = True
                    for w in neighbors_list[c]:
                        if cover_count[w] == 1:
                            redundant = False
                            break
                    if redundant:
                        # Remove c
                        del code_list[idx]
                        code_set.remove(c)
                        for w in neighbors_list[c]:
                            cover_count[w] -= 1
                        changed = True
                        improved = True
                        if len(code_list) < best_size:
                            best_size = len(code_list)
                            best_code = list(code_list)
                        break # Restart scan from beginning after change
            
            if improved:
                no_improvement = 0
                continue
            
            # 2. 2-Opt Move: Remove c1, c2, Add c_new
            # Find a pair (c1, c2) such that their unique coverage is covered by some c_new
            if len(code_list) >= 2:
                # Sample a few pairs to avoid O(N^2) full scan every step
                # N is ~180, N^2 is ~32k. 15 neighbors each.
                # Full scan might be okay if we are fast.
                # Let's try random sampling of pairs first.
                pair_found = False
                attempts = 50
                for _ in range(attempts):
                    if time.time() > time_limit:
                        break
                    i1, i2 = random.sample(range(len(code_list)), 2)
                    c1 = code_list[i1]
                    c2 = code_list[i2]
                    
                    # Identify words that become uncovered if c1, c2 are removed
                    # These are words w in N(c1) U N(c2) such that
                    # cover_count(w) == 1 (and w in N(c1) XOR N(c2) is not enough, 
                    # if w in N(c1) and cover_count[w]==1, it's unique to c1.
                    # if w in N(c1) U N(c2) and cover_count[w]==2, it's unique to {c1,c2}.
                    # if cover_count[w] > 2, it's covered by others.
                    
                    must_cover = set()
                    # Check neighbors of c1
                    for w in neighbors_list[c1]:
                        if cover_count[w] <= 2:
                            # If cover_count is 1, it must be unique to c1 (since c2 might not cover it)
                            # If cover_count is 2, it must be covered by c1 and c2 ONLY.
                            if cover_count[w] == 1:
                                must_cover.add(w)
                            elif cover_count[w] == 2:
                                # Check if c2 covers w
                                if w in neighbors_set[c2]:
                                    must_cover.add(w)
                                # Else, it's covered by c1 and someone else, so removing c1,c2 leaves it covered.
                    
                    # Check neighbors of c2 (only those not in c1 to avoid double work, but set handles it)
                    for w in neighbors_list[c2]:
                        if cover_count[w] <= 2:
                            if cover_count[w] == 1:
                                must_cover.add(w)
                            elif cover_count[w] == 2:
                                if w in neighbors_set[c1]:
                                    must_cover.add(w)
                    
                    if not must_cover:
                        # We can just remove c1 and c2? No, we want to maintain coverage.
                        # If must_cover is empty, it means all words covered by c1 or c2 are covered by others.
                        # Then we can remove c1 and c2 and add NOTHING?
                        # That would reduce size by 2.
                        # But we must ensure the code remains a covering code.
                        # If must_cover is empty, removing c1, c2 is safe.
                        # Let's just remove them one by one in the reduction phase?
                        # The reduction phase already handles single removals.
                        # If we can remove two, the reduction phase would have removed one, then the other.
                        # So if we reach here, single removals are blocked.
                        # Thus must_cover should not be empty if single removals are blocked.
                        # Unless c1 and c2 together cover some word w with count 2, 
                        # but individually they are not redundant.
                        # In that case, must_cover contains w.
                        pass

                    if must_cover:
                        # Find c_new that covers all words in must_cover
                        # c_new must not be in code_set
                        best_c_new = -1
                        best_score = -1
                        
                        # Heuristic: c_new should cover many other "critical" words to allow future reductions
                        # Score: number of words in N(c_new) with cover_count == 1 (excluding must_cover)
                        # This makes other codewords less critical.
                        
                        # To speed up, we can pre-filter candidates.
                        # c_new must cover at least one word in must_cover.
                        # Let's just scan all c_new.
                        
                        for c_new in range(total_words):
                            if c_new in code_set:
                                continue
                            
                            # Check if c_new covers all must_cover
                            # Using set intersection is fast
                            # if not must_cover.issubset(neighbors_set[c_new]): continue
                            
                            # Faster: check each w in must_cover
                            ok = True
                            for w in must_cover:
                                if not neighbors_set[c_new].__contains__(w):
                                    ok = False
                                    break
                            if not ok:
                                continue
                            
                            # Score: how many "critical" words (count==1) does c_new cover?
                            # This helps by increasing the count of those words to 2,
                            # potentially making their covering codewords removable.
                            score = 0
                            for w in neighbors_list[c_new]:
                                if cover_count[w] == 1:
                                    score += 1
                                elif cover_count[w] == 2:
                                    score += 0.5
                            
                            if score > best_score:
                                best_score = score
                                best_c_new = c_new
                                if best_score > 10: # Good enough
                                    break
                        
                        if best_c_new != -1:
                            # Perform move: Remove c1, c2, Add best_c_new
                            # Update cover_count
                            for w in neighbors_list[c1]:
                                cover_count[w] -= 1
                            for w in neighbors_list[c2]:
                                cover_count[w] -= 1
                            for w in neighbors_list[best_c_new]:
                                cover_count[w] += 1
                            
                            code_set.remove(c1)
                            code_set.remove(c2)
                            code_set.add(best_c_new)
                            
                            # Update code_list
                            # Remove c1 and c2 (indices might have changed, so use values)
                            # Since we are in a loop, it's safer to rebuild or carefully delete.
                            # Let's rebuild code_list from set for simplicity in this rare event?
                            # No, list order doesn't matter, but we need to remove specific items.
                            # We can just filter.
                            code_list = [c for c in code_list if c != c1 and c != c2]
                            code_list.append(best_c_new)
                            
                            improved = True
                            pair_found = True
                            if len(code_list) < best_size:
                                best_size = len(code_list)
                                best_code = list(code_list)
                            break
                            
                if pair_found:
                    no_improvement = 0
                    continue
            
            # 3. 1-Opt Move: Swap c1 with c_new
            if code_list:
                idx = random.randint(0, len(code_list) - 1)
                c1 = code_list[idx]
                
                # Find unique words for c1
                must_cover = set()
                for w in neighbors_list[c1]:
                    if cover_count[w] == 1:
                        must_cover.add(w)
                
                if must_cover:
                    best_c_new = -1
                    best_score = -1
                    
                    for c_new in range(total_words):
                        if c_new in code_set:
                            continue
                        
                        ok = True
                        for w in must_cover:
                            if not neighbors_set[c_new].__contains__(w):
                                ok = False
                                break
                        if not ok:
                            continue
                        
                        # Score: cover more critical words
                        score = 0
                        for w in neighbors_list[c_new]:
                            if cover_count[w] == 1:
                                score += 1
                            elif cover_count[w] == 2:
                                score += 0.5
                            
                        if score > best_score:
                            best_score = score
                            best_c_new = c_new
                            if best_score > 10:
                                break
                    
                    if best_c_new != -1:
                        # Swap
                        for w in neighbors_list[c1]:
                            cover_count[w] -= 1
                        for w in neighbors_list[best_c_new]:
                            cover_count[w] += 1
                        
                        code_set.remove(c1)
                        code_set.add(best_c_new)
                        
                        code_list[idx] = best_c_new
                        
                        improved = True
                        no_improvement = 0
                        continue
            
            # 4. Perturbation
            if no_improvement > 20:
                # Add some random codewords
                k = random.randint(3, 10)
                added = 0
                for _ in range(k):
                    c_new = random.randint(0, total_words - 1)
                    if c_new not in code_set:
                        code_list.append(c_new)
                        code_set.add(c_new)
                        for w in neighbors_list[c_new]:
                            cover_count[w] += 1
                        added += 1
                no_improvement = 0
                # We don't update best_code here as size increased
            else:
                no_improvement += 1
                
        return best_code

    # Execution
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_overall_code = []
    best_overall_size = float('inf')
    
    # Multiple restarts
    while time.time() < deadline:
        # 1. Greedy
        initial_code = greedy_set_cover()
        
        # 2. Local Search
        # Allocate remaining time, but keep a margin
        remaining = deadline - time.time()
        if remaining < 5:
            break
            
        # Use most of the time for local search, but allow for final check
        search_time = remaining - 2.0
        if search_time < 1.0:
            search_time = 1.0
            
        final_code = local_search(initial_code, time.time() + search_time)
        
        # 3. Final Cleanup (Reduction)
        code_set = set(final_code)
        cover_count = [0] * total_words
        for c in code_set:
            for w in neighbors_list[c]:
                cover_count[w] += 1
        
        final_list = list(code_set)
        improved = True
        cleanup_deadline = min(deadline, time.time() + 2.0)
        while improved and time.time() < cleanup_deadline:
            improved = False
            indices = list(range(len(final_list)))
            random.shuffle(indices)
            for idx in indices:
                if time.time() > cleanup_deadline:
                    break
                c = final_list[idx]
                redundant = True
                for w in neighbors_list[c]:
                    if cover_count[w] == 1:
                        redundant = False
                        break
                if redundant:
                    del final_list[idx]
                    code_set.remove(c)
                    for w in neighbors_list[c]:
                        cover_count[w] -= 1
                    improved = True

        if len(final_list) < best_overall_size:
            best_overall_size = len(final_list)
            best_overall_code = final_list
            # If we hit the lower bound or a very good number, we could stop, but let's try to beat 186.
            # 186 is the target.
            
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
