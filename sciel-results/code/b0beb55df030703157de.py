<<<<<<< SEARCH
import time
import numpy as np
import random
import sys
=======
import time
import numpy as np
import random
import sys
from itertools import product
>>>>>>> REPLACE
<<<<<<< SEARCH
    # Only optimized for the specific task: q=3, n=8, r=1
    # For other parameters, fall back to a simple valid construction
    if q != 3 or n != 8 or radius != 1:
        # Simple valid code: all words (worst case) or greedy
        # Let's do a quick greedy for generality
        total = q ** n
        # Precompute balls
        balls = []
        for x in range(total):
            ball = [x]
            digits = []
            tmp = x
            for i in range(n):
                d = tmp % q
                digits.append(d)
                tmp //= q
            for i in range(n):
                old_d = digits[i]
                mult = q ** i
                for new_d in range(q):
                    if new_d != old_d:
                        y = x + (new_d - old_d) * mult
                        ball.append(y)
            balls.append(ball)
        
        covered = np.zeros(total, dtype=bool)
        counts = np.array([len(b) for b in balls], dtype=np.int32)
        selected = []
        
        while True:
            if time.time() > deadline - 5:
                break
            max_idx = np.argmax(counts)
            if counts[max_idx] == 0:
                break
            selected.append(max_idx)
            for y in balls[max_idx]:
                if not covered[y]:
                    covered[y] = True
                    for i in range(len(balls)):
                        if y in balls[i]:
                            counts[i] -= 1
            if np.all(covered):
                break
        
        powers = [1] * n
        for i in range(1, n):
            powers[i] = powers[i-1] * q
        result = []
        for idx in selected:
            digits = []
            tmp = idx
            for i in range(n):
                d = tmp % q
                digits.append(d)
                tmp //= q
            result.append(digits)
        return result
=======
    # Only optimized for the specific task: q=3, n=8, r=1
    # For other parameters, fall back to a simple valid construction
    if q != 3 or n != 8 or radius != 1:
        # Simple valid code: all words (worst case) or greedy
        # Let's do a quick greedy for generality
        total = q ** n
        # Precompute balls
        balls = []
        for x in range(total):
            ball = [x]
            digits = []
            tmp = x
            for i in range(n):
                d = tmp % q
                digits.append(d)
                tmp //= q
            for i in range(n):
                old_d = digits[i]
                mult = q ** i
                for new_d in range(q):
                    if new_d != old_d:
                        y = x + (new_d - old_d) * mult
                        ball.append(y)
            balls.append(ball)
        
        covered = np.zeros(total, dtype=bool)
        counts = np.array([len(b) for b in balls], dtype=np.int32)
        selected = []
        
        while True:
            if time.time() > deadline - 5:
                break
            max_idx = np.argmax(counts)
            if counts[max_idx] == 0:
                break
            selected.append(max_idx)
            for y in balls[max_idx]:
                if not covered[y]:
                    covered[y] = True
                    for i in range(len(balls)):
                        if y in balls[i]:
                            counts[i] -= 1
            if np.all(covered):
                break
        
        powers = [1] * n
        for i in range(1, n):
            powers[i] = powers[i-1] * q
        result = []
        for idx in selected:
            digits = []
            tmp = idx
            for i in range(n):
                d = tmp % q
                digits.append(d)
                tmp //= q
            result.append(digits)
        return result
>>>>>>> REPLACE
<<<<<<< SEARCH
    # Specific case: q=3, n=8, r=1
    total = 3 ** 8  # 6561
    ball_size = 1 + 8 * 2  # 17
    
    # Precompute balls as lists of integers
    balls = []
    for x in range(total):
        ball = [x]
        digits = []
        tmp = x
        for i in range(8):
            d = tmp % 3
            digits.append(d)
            tmp //= 3
        for i in range(8):
            old_d = digits[i]
            mult = 3 ** i
            for new_d in range(3):
                if new_d != old_d:
                    y = x + (new_d - old_d) * mult
                    ball.append(y)
        balls.append(ball)
    
    # Precompute inverse: for each y, list of x such that y in balls[x]
    # This is the same as balls[y] due to symmetry
    # So inv[y] = balls[y]
    
    # Bitset representation for uncovered set
    # total = 6561, so we need 6561 bits = 103 64-bit words
    num_words = (total + 63) // 64
    
    # Initialize uncovered bitset: all 1s
    uncovered = np.ones(num_words, dtype=np.uint64)
    
    # Precompute ball bitsets for fast intersection
    ball_bitsets = []
    for x in range(total):
        bs = np.zeros(num_words, dtype=np.uint64)
        for y in balls[x]:
            word_idx = y // 64
            bit_idx = y % 64
            bs[word_idx] |= (1 << bit_idx)
        ball_bitsets.append(bs)
    
    # Current best solution
    best_size = total
    best_solution = None
    
    # Branch and Bound
    # State: current set of chosen codewords, uncovered bitset
    # We use a recursive approach with a stack to avoid recursion limit issues
    # But depth is at most ~500, which might be too deep for Python recursion.
    # Let's use an iterative deepening or a simple DFS with a stack.
    # Actually, a simple greedy gives ~500. B&B might not reach optimal quickly.
    # Let's try a hybrid: Greedy to get a good upper bound, then B&B to improve.
    
    # 1. Greedy to get initial upper bound
    covered = np.zeros(total, dtype=bool)
    counts = np.array([len(b) for b in balls], dtype=np.int32)
    greedy_selected = []
    
    while True:
        if time.time() > deadline - 30:
            break
        max_idx = np.argmax(counts)
        if counts[max_idx] == 0:
            break
        greedy_selected.append(max_idx)
        for y in balls[max_idx]:
            if not covered[y]:
                covered[y] = True
                for i in range(total):
                    if y in balls[i]:
                        counts[i] -= 1
        if np.all(covered):
            break
    
    if len(greedy_selected) < best_size:
        best_size = len(greedy_selected)
        best_solution = greedy_selected[:]
    
    # 2. Local Search on greedy solution to improve further
    current_set = set(greedy_selected)
    cover_count = np.zeros(total, dtype=np.int32)
    for c in current_set:
        for y in balls[c]:
            cover_count[y] += 1
    
    def is_redundant(c):
        for y in balls[c]:
            if cover_count[y] == 1:
                return False
        return True
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        candidates = list(current_set)
        random.shuffle(candidates)
        for c in candidates:
            if c not in current_set:
                continue
            if is_redundant(c):
                for y in balls[c]:
                    cover_count[y] -= 1
                current_set.remove(c)
                improved = True
                if len(current_set) < best_size:
                    best_size = len(current_set)
                    best_solution = list(current_set)
    
    # 3. Try a few random restarts with different greedy heuristics
    for restart in range(5):
        if time.time() > deadline - 10:
            break
        # Weighted greedy: randomly perturb the counts
        counts = np.array([len(b) for b in balls], dtype=np.int32)
        # Add random noise to break ties
        noise = np.random.uniform(0, 1, size=total)
        counts_noisy = counts.astype(float) + noise * 0.1
        
        covered = np.zeros(total, dtype=bool)
        selected = []
        
        while True:
            if time.time() > deadline - 10:
                break
            max_idx = np.argmax(counts_noisy)
            if counts[max_idx] == 0:
                break
            selected.append(max_idx)
            for y in balls[max_idx]:
                if not covered[y]:
                    covered[y] = True
                    for i in range(total):
                        if y in balls[i]:
                            counts[i] -= 1
                            counts_noisy[i] -= 1  # Keep noise consistent? No, just recompute
            if np.all(covered):
                break
        
        # Local search on this new solution
        current_set = set(selected)
        cover_count = np.zeros(total, dtype=np.int32)
        for c in current_set:
            for y in balls[c]:
                cover_count[y] += 1
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            candidates = list(current_set)
            random.shuffle(candidates)
            for c in candidates:
                if c not in current_set:
                    continue
                if is_redundant(c):
                    for y in balls[c]:
                        cover_count[y] -= 1
                    current_set.remove(c)
                    improved = True
                    if len(current_set) < best_size:
                        best_size = len(current_set)
                        best_solution = list(current_set)
    
    # 4. If we have time, try a more aggressive local search with swaps
    if best_solution is not None and time.time() < deadline - 5:
        current_set = set(best_solution)
        cover_count = np.zeros(total, dtype=np.int32)
        for c in current_set:
            for y in balls[c]:
                cover_count[y] += 1
        
        # Try to remove any redundant
        for c in list(current_set):
            if is_redundant(c):
                for y in balls[c]:
                    cover_count[y] -= 1
                current_set.remove(c)
                if len(current_set) < best_size:
                    best_size = len(current_set)
                    best_solution = list(current_set)
    
    # Convert best_solution to list of digit lists
    if best_solution is None:
        # Fallback: return all words (invalid for score but valid structure)
        # Actually, we must return a valid cover.
        # If best_solution is None, use the greedy one from step 1
        best_solution = greedy_selected
    
    result = []
    for idx in best_solution:
        digits = []
        tmp = idx
        for i in range(8):
            d = tmp % 3
            digits.append(d)
            tmp //= 3
        result.append(digits)
    
    return result
=======
    # Specific case: q=3, n=8, r=1
    total = 3 ** 8  # 6561
    ball_size = 1 + 8 * 2  # 17
    
    # Precompute balls as numpy arrays for faster operations
    balls = []
    for x in range(total):
        ball = [x]
        tmp = x
        for i in range(8):
            d = tmp % 3
            tmp //= 3
            for new_d in range(3):
                if new_d != d:
                    y = x + (new_d - d) * (3 ** i)
                    ball.append(y)
        balls.append(np.array(ball, dtype=np.int32))
    
    # Precompute inverse: for each y, list of x such that y in balls[x]
    # Due to symmetry, inv[y] = balls[y], but we'll use the balls directly
    
    # Current best solution
    best_size = total
    best_solution = None
    
    # 1. Fast Greedy with vectorized updates
    def greedy_cover(heuristic="max"):
        """
        Greedy set cover with optional heuristic:
        - "max": pick codeword covering most uncovered points
        - "random": pick random codeword that covers at least one uncovered point
        """
        covered = np.zeros(total, dtype=bool)
        uncovered_count = np.array([len(b) for b in balls], dtype=np.int32)
        selected = []
        
        while True:
            if time.time() > deadline - 15:
                break
            if heuristic == "max":
                max_idx = np.argmax(uncovered_count)
            else:
                # Pick a random candidate that still covers something
                candidates = np.where(uncovered_count > 0)[0]
                if len(candidates) == 0:
                    break
                max_idx = candidates[random.randint(0, len(candidates) - 1)]
            
            if uncovered_count[max_idx] == 0:
                break
            
            selected.append(max_idx)
            
            # Vectorized update: find all words in ball[max_idx] that are not yet covered
            ball_words = balls[max_idx]
            new_covered = ball_words[~covered[ball_words]]
            
            # Update uncovered_count for all codewords that cover any of new_covered
            if len(new_covered) > 0:
                # For each newly covered word, decrement count for all codewords covering it
                # This is expensive, so we batch it
                for y in new_covered:
                    covered[y] = True
                    # Find all codewords whose ball contains y
                    # Since balls are symmetric, these are exactly the codewords in balls[y]
                    for c in balls[y]:
                        uncovered_count[c] -= 1
            
            if np.all(covered):
                break
        
        return selected
    
    # Run greedy with max heuristic
    greedy_selected = greedy_cover("max")
    
    if len(greedy_selected) < best_size:
        best_size = len(greedy_selected)
        best_solution = greedy_selected[:]
    
    # 2. Local Search: Remove redundant codewords
    def local_search_remove(selected):
        """Remove redundant codewords from the selected set."""
        current_set = set(selected)
        cover_count = np.zeros(total, dtype=np.int32)
        for c in current_set:
            cover_count[balls[c]] += 1
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            candidates = list(current_set)
            random.shuffle(candidates)
            for c in candidates:
                if c not in current_set:
                    continue
                # Check if c is redundant: all words in ball[c] are covered by others
                if np.all(cover_count[balls[c]] > 1):
                    cover_count[balls[c]] -= 1
                    current_set.remove(c)
                    improved = True
                    if len(current_set) < best_size:
                        nonlocal best_size, best_solution
                        best_size = len(current_set)
                        best_solution = list(current_set)
        
        return current_set
    
    # Apply local search to greedy solution
    current_set = local_search_remove(greedy_selected)
    
    # 3. Try multiple random restarts with different greedy strategies
    for restart in range(8):
        if time.time() > deadline - 10:
            break
        
        # Alternate between max and random greedy
        heuristic = "max" if restart % 2 == 0 else "random"
        selected = greedy_cover(heuristic)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_solution = selected[:]
        
        # Local search on this solution
        current_set = local_search_remove(selected)
    
    # 4. Aggressive local search with swap moves
    if best_solution is not None and time.time() < deadline - 5:
        current_set = set(best_solution)
        cover_count = np.zeros(total, dtype=np.int32)
        for c in current_set:
            cover_count[balls[c]] += 1
        
        # Try to remove any remaining redundant codewords
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            candidates = list(current_set)
            random.shuffle(candidates)
            for c in candidates:
                if c not in current_set:
                    continue
                if np.all(cover_count[balls[c]] > 1):
                    cover_count[balls[c]] -= 1
                    current_set.remove(c)
                    improved = True
                    if len(current_set) < best_size:
                        best_size = len(current_set)
                        best_solution = list(current_set)
    
    # Convert best_solution to list of digit lists
    if best_solution is None:
        best_solution = greedy_selected
    
    result = []
    for idx in best_solution:
        digits = []
        tmp = idx
        for i in range(8):
            d = tmp % 3
            digits.append(d)
            tmp //= 3
        result.append(digits)
    
    return result
>>>>>>> REPLACE
