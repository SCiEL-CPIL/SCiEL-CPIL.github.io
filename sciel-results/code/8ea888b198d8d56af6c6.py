import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second margin
    
    def calc_score(U):
        if not U:
            return 0.0
        U = sorted(set(U))
        if 0 not in U:
            return 0.0
        max_u = max(U)
        if max_u == 0:
            return 1.0
        sum_set = set()
        for a in U:
            for b in U:
                sum_set.add(a + b)
        diff_set = set()
        for a in U:
            for b in U:
                diff_set.add(a - b)
        
        size_sum = len(sum_set)
        size_diff = len(diff_set)
        
        if size_sum == 0:
            return 0.0
            
        ratio = size_diff / size_sum
        if ratio <= 1.0:
            return 1.0
        
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 1.0
            
        return 1.0 + math.log(ratio) / denom

    best_set = [0, 1, 3]
    best_score = calc_score(best_set)
    
    # Initial candidates
    candidates = []
    # Powers of 2
    for k in range(2, 10):
        candidates.append([2**i for i in range(k)])
    # Arithmetic progressions
    for n in range(2, 10):
        for step in [1, 2, 3, 5, 7]:
            candidates.append([i * step for i in range(n)])
    
    # Random small sets
    for _ in range(100):
        n = random.randint(3, 8)
        max_val = random.randint(10, 50)
        s = random.sample(range(1, max_val), n-1)
        s.append(0)
        candidates.append(s)
        
    # Evaluate initial candidates
    for c in candidates:
        if time.time() > deadline:
            break
        s = calc_score(c)
        if s > best_score:
            best_score = s
            best_set = c[:]

    # Local Search
    # Try to add, remove, or modify elements
    for _ in range(5000):
        if time.time() > deadline:
            break
            
        current = best_set[:]
        # Random mutation
        if random.random() < 0.3 and len(current) > 1:
            # Remove a random element (keep 0)
            idx = random.randint(1, len(current) - 1)
            current.pop(idx)
        elif random.random() < 0.3:
            # Add a random element
            max_val = max(current) + random.randint(1, 10)
            current.append(random.randint(1, max_val))
        else:
            # Modify a random element
            idx = random.randint(1, len(current) - 1)
            current[idx] += random.randint(-5, 5)
            if current[idx] < 0:
                current[idx] = 1
            # Ensure uniqueness
            if len(set(current)) != len(current):
                continue
                
        s = calc_score(current)
        if s > best_score:
            best_score = s
            best_set = current[:]

    return best_set
