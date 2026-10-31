import math
import random
import time

def score(U):
    if not U:
        return 0
    if 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return 1.0
    return 1 + math.log(ratio) / math.log(2 * max_u + 1)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Seeds: Known Golomb rulers and some random small sets
    seeds = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 20],
        [0, 1, 4, 10, 12, 17, 20, 24],
        [0, 1, 4, 10, 12, 17, 20, 24, 28],
        [0, 1, 4, 10, 12, 17, 20, 24, 28, 31],
        [0, 1, 4, 10, 12, 17, 20, 24, 28, 31, 34], # n=11 approx
        [0, 2, 3, 5, 8, 13, 21, 34, 55, 89] # Fibonacci-like?
    ]
    
    # Add some random seeds with small max
    for _ in range(20):
        n = random.randint(5, 15)
        L = random.randint(n, 50)
        U = {0}
        while len(U) < n:
            U.add(random.randint(1, L))
        seeds.append(sorted(U))
        
    best_set = seeds[0]
    best_score = score(best_set)
    
    # Evaluate all seeds to find the best starting point
    for s in seeds:
        sc = score(s)
        if sc > best_score:
            best_score = sc
            best_set = list(s)
            
    current_set = list(best_set)
    current_score = best_score
    
    # Simulated Annealing / Hill Climbing
    T = 1.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Generate neighbor
        neighbor = list(current_set)
        move = random.choice(['add', 'remove', 'swap'])
        
        if move == 'add':
            max_val = max(neighbor)
            # Try adding small numbers to keep L small
            # Bias towards small values
            candidates = [i for i in range(1, max_val + 10) if i not in neighbor]
            if candidates:
                # Weight towards smaller numbers
                weights = [1.0 / (1.0 + c) for c in candidates]
                neighbor.append(random.choices(candidates, weights=weights)[0])
        elif move == 'remove':
            if len(neighbor) > 1:
                rem = random.choice([x for x in neighbor if x != 0])
                neighbor.remove(rem)
        else:
            # Swap a random element with a new one
            if len(neighbor) > 1:
                idx = random.choice([i for i, x in enumerate(neighbor) if x != 0])
                old_val = neighbor[idx]
                # Try to swap with a value that might improve ratio
                # Or just random in a range
                new_val = random.randint(1, max(neighbor) + 5)
                if new_val not in neighbor:
                    neighbor[idx] = new_val
        
        neighbor.sort()
        new_score = score(neighbor)
        
        delta = new_score - current_score
        if delta > 0 or random.random() < math.exp(delta / T):
            current_set = neighbor
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = list(current_set)
                
        T *= cooling
        if T < 0.001:
            T = 0.001

    return best_set
