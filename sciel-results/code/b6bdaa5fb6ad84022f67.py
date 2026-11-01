import numpy as np
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    total = q ** n
    ball_size = 1 + n * (q - 1)
    
    def int_to_digits(x, q, n):
        digits = [0] * n
        for i in range(n - 1, -1, -1):
            digits[i] = x % q
            x //= q
        return digits
    
    def digits_to_int(digits, q):
        x = 0
        for d in digits:
            x = x * q + d
        return x
    
    all_digits = [int_to_digits(i, q, n) for i in range(total)]
    
    # Precompute neighbors
    neighbors = []
    for i in range(total):
        d = all_digits[i]
        nbrs = [i]
        for pos in range(n):
            orig = d[pos]
            for val in range(q):
                if val == orig:
                    continue
                new_d = d[:]
                new_d[pos] = val
                nbrs.append(digits_to_int(new_d, q))
        neighbors.append(nbrs)
    
    uncovered = [True] * total
    uncovered_count = total
    codewords = []
    
    random.seed(42)
    
    # Greedy phase
    while uncovered_count > 0:
        if time.time() > deadline:
            break
        
        uncovered_indices = [i for i in range(total) if uncovered[i]]
        if not uncovered_indices:
            break
            
        sample_size = min(200, len(uncovered_indices))
        candidates = random.sample(uncovered_indices, sample_size)
        
        best_word = None
        best_count = -1
        for word in candidates:
            count = 0
            for nb in neighbors[word]:
                if uncovered[nb]:
                    count += 1
            if count > best_count:
                best_count = count
                best_word = word
        
        if best_word is None:
            best_word = uncovered_indices[0]
        
        codewords.append(all_digits[best_word])
        for nb in neighbors[best_word]:
            if uncovered[nb]:
                uncovered[nb] = False
                uncovered_count -= 1
    
    # Local improvement: try removing redundant codewords
    if time.time() < deadline - 10:
        for i in range(len(codewords)):
            if time.time() > deadline:
                break
            # Check if removing codeword i still covers everything
            # This is expensive, so only do it for a subset
            if i % 10 != 0:
                continue
            
            # Temporarily remove and check coverage
            remaining = codewords[:i] + codewords[i+1:]
            if not remaining:
                continue
            
            # Check if all words are covered by remaining
            covered = [False] * total
            for cw in remaining:
                idx = digits_to_int(cw, q)
                for nb in neighbors[idx]:
                    covered[nb] = True
            
            if all(covered):
                codewords = remaining
                break  # Restart improvement
    
    return codewords
