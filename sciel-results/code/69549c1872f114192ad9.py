import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code using Greedy Set Cover and Local Search (Redundancy Removal).
    """
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters, though the task specifies this one
        # Direct sum of tetracode for n=7, q=3, r=1 is 243.
        # For general case, a simple greedy is better than nothing.
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
             return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
             return [list(t) for t in itertools.product(range(q), repeat=n)]

    total_words = q ** n
    # Precompute balls
    # Each word is an integer 0..total_words-1
    # Convert int to tuple of digits
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

    # Precompute neighbors (ball of radius 1)
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
        balls.append(list(ball_set))
        
    # Greedy Set Cover
    def greedy_cover(randomize=False):
        remaining = np.ones(total_words, dtype=bool)
        selected = []
        # To handle tie-breaking with randomization, we can shuffle the candidate list
        candidates = list(range(total_words))
        if randomize:
            random.shuffle(candidates)
            
        while np.any(remaining):
            best_score = -1
            best_candidate = -1
            # We want to maximize the number of newly covered words
            # Score = sum(remaining[j] for j in balls[c])
            # To speed up, we can break early if we find a perfect score (size of ball)
            max_possible = len(balls[candidates[0]]) # All balls have same size
            for c in candidates:
                if not remaining[c]: # If the center is already covered, it might still be a good candidate?
                    # Actually, we just want to cover 'remaining' words.
                    # A candidate c covers balls[c]. The new coverage is intersection of balls[c] and remaining.
                    # We don't require c itself to be uncovered, just that it adds coverage.
                    pass
                
                # Count overlap
                # balls[c] is a list of indices
                # remaining is a bool array
                score = np.sum(remaining[balls[c]])
                
                if score > best_score:
                    best_score = score
                    best_candidate = c
                    if best_score == max_possible:
                        break
                elif score == best_score and randomize:
                    # Random tie break
                    if random.random() < 0.5:
                        best_score = score
                        best_candidate = c

            if best_candidate == -1:
                break # Should not happen

            selected.append(best_candidate)
            # Update remaining
            remaining[balls[best_candidate]] = False
            
        return selected

    def remove_redundants(selected):
        # Convert to set for fast lookup? No, we need to iterate.
        # We maintain a coverage count array
        coverage = np.zeros(total_words, dtype=int)
        for c in selected:
            coverage[balls[c]] += 1
        
        changed = True
        while changed:
            changed = False
            # Iterate in reverse to allow removal while iterating? 
            # Or just make a copy of the list to iterate over.
            for i in range(len(selected) - 1, -1, -1):
                c = selected[i]
                # Check if any word in balls[c] has coverage count == 1
                # If min(coverage[balls[c]]) > 1, then c is redundant
                if np.min(coverage[balls[c]]) > 1:
                    # Remove c
                    coverage[balls[c]] -= 1
                    selected.pop(i)
                    changed = True
        return selected

    best_solution = None
    start_time = time.time()
    deadline = start_time + 100 # 100s safety margin
    
    # First run: Deterministic
    sol = greedy_cover(randomize=False)
    sol = remove_redundants(sol)
    if best_solution is None or len(sol) < len(best_solution):
        best_solution = sol
        
    # Subsequent runs: Randomized
    while time.time() < deadline:
        sol = greedy_cover(randomize=True)
        sol = remove_redundants(sol)
        if best_solution is None or len(sol) < len(best_solution):
            best_solution = sol
            # If we hit the theoretical best known (186), we could stop, but let's try to go lower
            # if len(best_solution) <= 186: break

    # Convert indices back to lists of digits
    return [int_to_digits(c) for c in best_solution]
