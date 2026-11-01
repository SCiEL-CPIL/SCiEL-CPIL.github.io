import itertools
import random
import time

def construct(q, n, radius):
    start_time = time.time()
    # Use 110s limit to be safe against 120s kill
    time_limit = 110.0
    
    total_words = q ** n
    
    # For very large problems, this heuristic might be too slow.
    # But the task is specifically q=3, n=7, radius=1.
    # We'll add a fallback if total_words is huge, but for this task it's fine.
    
    # Precomputation
    word_to_idx = {}
    idx_to_word = []
    for i in range(total_words):
        w = []
        val = i
        for _ in range(n):
            w.append(val % q)
            val //= q
        w = tuple(w)
        word_to_idx[w] = i
        idx_to_word.append(w)
        
    balls = [set() for _ in range(total_words)]
    
    for i, w in enumerate(idx_to_word):
        balls[i].add(i)
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                new_w = tuple(new_w)
                j = word_to_idx[new_w]
                balls[i].add(j)
                
    code = set()
    coverage_count = [0] * total_words
    uncovered = set(range(total_words))
    
    # 1. Greedy Initialization
    # We use a simple greedy: pick the word that covers the most uncovered words.
    while uncovered and time.time() - start_time < time_limit * 0.2:
        best_idx = -1
        max_covered = 0
        # To break ties randomly and potentially find better structures, 
        # we could collect all best candidates and pick one.
        best_candidates = []
        
        for i in range(total_words):
            if i in code:
                continue
            count = 0
            for w in balls[i]:
                if w in uncovered:
                    count += 1
            if count > max_covered:
                max_covered = count
                best_candidates = [i]
            elif count == max_covered and count > 0:
                best_candidates.append(i)
        
        if not best_candidates:
            break
            
        # Pick randomly from best candidates to avoid deterministic traps
        best_idx = random.choice(best_candidates)
        
        code.add(best_idx)
        for w in balls[best_idx]:
            coverage_count[w] += 1
            if coverage_count[w] == 1:
                uncovered.discard(w)
                
    # 2. Local Search
    improved = True
    # We track consecutive swaps without removal to avoid infinite loops
    swaps_without_removal = 0
    max_swaps_without_removal = 5000
    
    while improved and time.time() - start_time < time_limit:
        improved = False
        
        # Phase 1: Try to remove redundant codewords
        for idx in list(code):
            if time.time() - start_time > time_limit:
                break
            removable = True
            for w in balls[idx]:
                if coverage_count[w] <= 1:
                    removable = False
                    break
            if removable:
                code.discard(idx)
                for w in balls[idx]:
                    coverage_count[w] -= 1
                improved = True
                swaps_without_removal = 0 # Reset counter
                break # Restart to check for more removals
                
        if improved or time.time() - start_time > time_limit:
            continue
            
        # Phase 2: Try swaps to escape local minima
        # If we've done too many swaps without a removal, we might be stuck.
        # We can still try, but we might want to perturb more aggressively later.
        # For now, just try swaps.
        
        swap_found = False
        # Iterate over a copy of code
        for idx_i in list(code):
            if time.time() - start_time > time_limit:
                break
                
            # Get neighbors
            neighbors = list(balls[idx_i])
            random.shuffle(neighbors)
            
            for idx_j in neighbors:
                if idx_j in code:
                    continue
                
                # Check if swap is valid
                # Valid if all w in ball[idx_i] \ ball[idx_j] have coverage_count[w] > 1
                valid = True
                for w in balls[idx_i]:
                    if w not in balls[idx_j]:
                        if coverage_count[w] <= 1:
                            valid = False
                            break
                
                if valid:
                    # Perform swap
                    code.discard(idx_i)
                    for w in balls[idx_i]:
                        coverage_count[w] -= 1
                        
                    code.add(idx_j)
                    for w in balls[idx_j]:
                        coverage_count[w] += 1
                        
                    swap_found = True
                    swaps_without_removal += 1
                    break # Restart outer loop to check for removals
            if swap_found:
                break
        
        if swap_found:
            improved = True
            if swaps_without_removal > max_swaps_without_removal:
                # We are likely stuck in a cycle of swaps.
                # We could stop, or try a random perturbation.
                # Let's just stop to avoid wasting time.
                # Or we can accept that we found a local minimum.
                break
        else:
            # No valid swaps found? This means we are in a "trap" where no single swap
            # maintains coverage.
            # We might need to add a codeword and remove two, or just accept the current solution.
            # For simplicity, we stop.
            break

    result = [list(idx_to_word[i]) for i in code]
    return result
