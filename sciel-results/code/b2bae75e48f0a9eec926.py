import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code using Greedy Set Cover with bitset acceleration
    and Local Search (Redundancy Removal + Swap).
    """
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
             return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
             return [list(t) for t in itertools.product(range(q), repeat=n)]

    total_words = q ** n
    ball_size = 1 + n * (q - 1)  # 15 for q=3, n=7, r=1

    def int_to_digits(x):
        digits = []
        for _ in range(n):
            digits.append(x % q)
            x //= q
        return digits

    def digits_to_int(d):
        x = 0
        multiplier = 1
        for i in range(n):
            x += d[i] * multiplier
            multiplier *= q
        return x

    # Precompute neighbors (ball of radius 1) as sorted lists
    balls = []
    for i in range(total_words):
        digits = int_to_digits(i)
        ball_set = {i}
        for pos in range(n):
            original_val = digits[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                new_digits = digits[:]
                new_digits[pos] = new_val
                ball_set.add(digits_to_int(new_digits))
        balls.append(sorted(ball_set))

    # Precompute ball as numpy array for fast operations
    balls_np = [np.array(b, dtype=np.int32) for b in balls]

    # Greedy Set Cover with numpy bitset
    def greedy_cover(randomize=False, seed_offset=0):
        remaining = np.ones(total_words, dtype=np.int8)
        selected = []
        
        # Precompute for speed: for each candidate, its ball mask
        # We'll use numpy array indexing for score computation
        
        while np.any(remaining):
            best_score = -1
            best_candidate = -1
            max_possible = ball_size
            
            if randomize:
                candidates = list(range(total_words))
                random.shuffle(candidates)
            else:
                candidates = range(total_words)
            
            for c in candidates:
                # Count how many words in balls[c] are still remaining
                score = np.sum(remaining[balls_np[c]])
                
                if score > best_score:
                    best_score = score
                    best_candidate = c
                    if best_score == max_possible:
                        break
                elif score == best_score and score > 0 and randomize:
                    # Random tie break - accept with some probability
                    if random.random() < 0.3:
                        best_candidate = c

            if best_candidate == -1:
                break

            selected.append(best_candidate)
            remaining[balls_np[best_candidate]] = 0
            
        return selected

    def remove_redundants(selected):
        if not selected:
            return selected
        
        coverage = np.zeros(total_words, dtype=np.int8)
        for c in selected:
            coverage[balls_np[c]] += 1
        
        changed = True
        while changed:
            changed = False
            for i in range(len(selected) - 1, -1, -1):
                c = selected[i]
                if np.min(coverage[balls_np[c]]) > 1:
                    coverage[balls_np[c]] -= 1
                    selected.pop(i)
                    changed = True
        return selected

    def local_search_swap(selected, max_iterations=500):
        """Try to replace one codeword with another to reduce total size."""
        if not selected:
            return selected
        
        selected = list(selected)
        coverage = np.zeros(total_words, dtype=np.int8)
        for c in selected:
            coverage[balls_np[c]] += 1
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iterations:
            improved = False
            iter_count += 1
            
            # Try removing each codeword and see if we can find a better replacement
            for i in range(len(selected)):
                c = selected[i]
                # Temporarily remove c
                coverage[balls_np[c]] -= 1
                
                # Find words that are now uncovered (coverage == 0)
                uncovered = np.where(coverage == 0)[0]
                
                if len(uncovered) == 0:
                    # c was redundant, remove it
                    selected.pop(i)
                    improved = True
                    break
                else:
                    # Try to find a single codeword that covers all uncovered
                    # or reduces the count most
                    best_new = -1
                    best_gain = 0
                    
                    # Sample candidates for speed
                    if len(uncovered) <= ball_size:
                        # Check all candidates that cover at least one uncovered word
                        candidate_set = set()
                        for u in uncovered:
                            candidate_set.add(u)
                        # Also add neighbors of uncovered words
                        for u in uncovered:
                            for nb in balls_np[u]:
                                candidate_set.add(nb)
                        
                        for cand in candidate_set:
                            gain = np.sum(coverage[balls_np[cand]] == 0)
                            if gain > best_gain:
                                best_gain = gain
                                best_new = cand
                            if best_gain == len(uncovered):
                                break
                    
                    if best_new != -1 and best_gain == len(uncovered):
                        # Perfect replacement
                        selected[i] = best_new
                        coverage[balls_np[best_new]] += 1
                        improved = True
                        break
                    else:
                        # Restore c
                        coverage[balls_np[c]] += 1
        
        return selected

    best_solution = None
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # First run: Deterministic greedy + redundancy removal + local search
    sol = greedy_cover(randomize=False)
    sol = remove_redundants(sol)
    sol = local_search_swap(sol, max_iterations=200)
    if best_solution is None or len(sol) < len(best_solution):
        best_solution = list(sol)
    
    # Subsequent runs: Randomized greedy + redundancy removal + local search
    run_id = 0
    while time.time() < deadline:
        run_id += 1
        sol = greedy_cover(randomize=True, seed_offset=run_id)
        sol = remove_redundants(sol)
        sol = local_search_swap(sol, max_iterations=100)
        
        if best_solution is None or len(sol) < len(best_solution):
            best_solution = list(sol)
            # If we hit the theoretical best known (186), we could stop
            if len(best_solution) <= 186:
                break
    
    # Convert indices back to lists of digits
    return [int_to_digits(c) for c in best_solution]
