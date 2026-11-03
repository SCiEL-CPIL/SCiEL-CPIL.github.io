import time
import random
from math import log
import itertools

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return 0.0
    sums = set()
    diffs = set()
    for i in range(n):
        for j in range(i, n):
            sums.add(U[i] + U[j])
        for j in range(i + 1, n):
            d = U[j] - U[i]
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return -1
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def local_search(U, deadline):
    current = sorted(set(U))
    current_score = score_set(current)
    if current_score < 0:
        return current, current_score
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(current)
        for i in range(1, n): # Skip 0
            old_val = current[i]
            # Try a wider range of nearby values
            for delta in range(-5, 6):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val <= 0:
                    continue
                # Ensure uniqueness and ordering
                new_set = [current[j] if j != i else new_val for j in range(n)]
                new_set = sorted(set(new_set))
                if len(new_set) != n:
                    continue
                new_score = score_set(new_set)
                if new_score > current_score:
                    current = new_set
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
        
        # Try adding an element - search wider range
        if time.time() < deadline and n < 18:
            max_val = max(current)
            # Try values near max and also some random positions
            candidates_to_try = []
            for new_val in range(max_val + 1, max_val + 20):
                candidates_to_try.append(new_val)
            # Also try inserting between existing elements
            for i in range(1, n):
                if current[i] - current[i-1] > 2:
                    candidates_to_try.append(current[i-1] + 1)
                    candidates_to_try.append(current[i] - 1)
            random.shuffle(candidates_to_try)
            for new_val in candidates_to_try:
                if new_val <= 0 or new_val in current:
                    continue
                new_set = sorted(current + [new_val])
                new_score = score_set(new_set)
                if new_score > current_score:
                    current = new_set
                    current_score = new_score
                    improved = True
                    break
            
        # Try removing an element
        if time.time() < deadline and n > 4:
            for i in range(1, n):
                new_set = [current[j] for j in range(n) if j != i]
                new_score = score_set(new_set)
                if new_score > current_score:
                    current = new_set
                    current_score = new_score
                    improved = True
                    break

    return current, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 100 # Safety margin to avoid timeout
    
    best_set = [0, 1, 4, 9, 11]
    best_score = score_set(best_set)
    
    # Generate base candidates
    candidates = []
    
    # 1. Exponential growth (2^k - 1)
    for k in range(1, 15):
        U = [0]
        val = 1
        for _ in range(k):
            U.append(val)
            val = 2 * val + 1
        if time.time() < deadline:
            candidates.append(U)
            
    # 2. Powers of 2
    for k in range(1, 15):
        U = [0]
        val = 1
        for _ in range(k):
            U.append(val)
            val = 2 * val
        if time.time() < deadline:
            candidates.append(U)
            
    # 3. Fibonacci
    a, b = 0, 1
    U = [0, 1]
    while b < 10000:
        U.append(b)
        a, b = b, a + b
    if time.time() < deadline:
        candidates.append(U)
    
    # 4. Squares
    U = [i*i for i in range(12)]
    if time.time() < deadline:
        candidates.append(U)
    
    # 5. Triangular numbers
    U = [i*(i+1)//2 for i in range(12)]
    if time.time() < deadline:
        candidates.append(U)
    
    # 6. Randomized exponential with different seeds
    for seed in [42, 123, 456, 789, 1000]:
        random.seed(seed)
        for _ in range(5):
            U = [0]
            val = random.randint(1, 5)
            for _ in range(random.randint(3, 10)):
                U.append(val)
                val = random.randint(int(val * 1.3), int(val * 2.8))
            if time.time() < deadline:
                candidates.append(U)
    
    # 7. Geometric sequences with various ratios
    for ratio in [1.5, 1.7, 1.9, 2.2, 2.5, 3.0]:
        U = [0]
        val = 1
        for _ in range(10):
            U.append(val)
            val = int(val * ratio)
        if time.time() < deadline:
            candidates.append(U)
    
    # 8. Sparse random sets
    random.seed(999)
    for _ in range(10):
        n_elem = random.randint(5, 12)
        max_val = random.randint(50, 500)
        U = [0] + sorted(random.sample(range(1, max_val), n_elem - 1))
        if time.time() < deadline:
            candidates.append(U)

    # Evaluate and refine
    for U in candidates:
        if time.time() > deadline - 15:
            break
        refined, score = local_search(U, deadline)
        if score > best_score:
            best_score = score
            best_set = refined

    return best_set
