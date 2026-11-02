<<<<<<< SEARCH
import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)

    # Fallback for other parameters
    # Simple exhaustive or greedy approach for small cases
    import itertools
    if total_words := q ** n <= 10000:
        # Greedy set cover
        all_words = [list(c) for c in itertools.product(range(q), repeat=n)]
        uncovered = set(range(len(all_words)))
        codewords = []
        
        # Precompute coverage
        def hamming_dist(a, b):
            return sum(1 for x, y in zip(a, b) if x != y)
        
        while uncovered:
            if time.time() > deadline - 5.0:
                break
            best_idx = -1
            best_count = -1
            for i in range(len(all_words)):
                count = 0
                for j in uncovered:
                    if hamming_dist(all_words[i], all_words[j]) <= radius:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_idx = i
            
            if best_idx == -1:
                break
                
            codewords.append(all_words[best_idx])
            # Remove covered
            new_uncovered = set()
            for j in uncovered:
                if hamming_dist(all_words[best_idx], all_words[j]) > radius:
                    new_uncovered.add(j)
            uncovered = new_uncovered
            
        return codewords
    else:
        # Return a valid but large code (exhaustive is too big, so just return all? No, must be valid)
        # For large cases, this might timeout or be invalid if not covering.
        # But the task is specific to q=3, n=8.
        return [list(c) for c in itertools.product(range(q), repeat=n)]
=======
import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s limit

    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)

    # Fallback for other parameters
    # Simple greedy set cover
    if q ** n <= 10000:
        all_words = [list(c) for c in itertools.product(range(q), repeat=n)]
        uncovered = set(range(len(all_words)))
        codewords = []
        
        def hamming_dist(a, b):
            return sum(1 for x, y in zip(a, b) if x != y)
        
        while uncovered:
            if time.time() > deadline - 5.0:
                break
            best_idx = -1
            best_count = -1
            for i in range(len(all_words)):
                count = 0
                for j in uncovered:
                    if hamming_dist(all_words[i], all_words[j]) <= radius:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_idx = i
            
            if best_idx == -1:
                break
                
            codewords.append(all_words[best_idx])
            new_uncovered = set()
            for j in uncovered:
                if hamming_dist(all_words[best_idx], all_words[j]) > radius:
                    new_uncovered.add(j)
            uncovered = new_uncovered
            
        return codewords
    else:
        return [list(c) for c in itertools.product(range(q), repeat=n)]
>>>>>>> REPLACE

