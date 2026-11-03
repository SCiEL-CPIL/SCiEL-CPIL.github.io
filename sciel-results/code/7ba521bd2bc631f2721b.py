import time
import random
from math import log
import sys

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

def fast_score(U, max_val_limit=10000):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return 0.0
    max_u = max(U)
    if max_u > max_val_limit:
        return score_set(U)
    
    # Use boolean arrays for speed if max_u is small enough
    # Sums range: 0 to 2*max_u
    # Diffs range: -max_u to max_u
    sum_size = 2 * max_u + 1
    diff_size = 2 * max_u + 1
    
    sum_flags = [False] * sum_size
    diff_flags = [False] * diff_size
    
    # Offset for diffs is max_u
    offset = max_u
    
    sum_count = 0
    diff_count = 0
    
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sj = ui + U[j]
            if not sum_flags[sj]:
                sum_flags[sj] = True
                sum_count += 1
            
            if i != j:
                dj = U[j] - ui
                idx1 = dj + offset
                idx2 = -dj + offset
                if not diff_flags[idx1]:
                    diff_flags[idx1] = True
                    diff_count += 1
                if not diff_flags[idx2]:
                    diff_flags[idx2] = True
                    diff_count += 1
    
    # Add 0 to diffs
    if not diff_flags[offset]:
        diff_flags[offset] = True
        diff_count += 1
    
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
    
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def simulated_annealing(initial_set, deadline, start_temp=10.0, end_temp=0.1, cooling=0.999):
    current = sorted(set(initial_set))
    current_score = fast_score(current)
    if current_score < 0:
        current_score = score_set(current)
    
    best = current[:]
    best_score = current_score
    
    temp = start_temp
    n = len(current)
    
    while time.time() < deadline and temp > end_temp:
        # Generate a neighbor
        move_type = random.random()
        new_set = current[:]
        
        if move_type < 0.7:
            # Replace an element
            idx = random.randint(1, n - 1) # Keep 0 fixed? Or allow moving 0? 0 is required.
            old_val = new_set[idx]
            # Try values in a range around old_val, and some random jumps
            # Range based on magnitude
            jump = max(1, int(abs(old_val) * 0.2))
            new_val = old_val + random.randint(-jump, jump)
            if new_val <= 0:
                new_val = random.randint(1, max(2, old_val * 2))
            
            # Check for duplicates
            if new_val in new_set and new_val != old_val:
                # If duplicate, skip or remove old
                pass 
            else:
                new_set[idx] = new_val
                new_set = sorted(set(new_set))
                if len(new_set) != n:
                    # If we created a duplicate, we lost an element. 
                    # This is a valid move (shrinking set), but we might want to maintain size.
                    # For now, accept if score improves.
                    pass
        elif move_type < 0.85:
            # Add an element
            if n < 30:
                max_val = max(current)
                new_val = random.randint(1, max_val * 2 + 10)
                if new_val not in current:
                    new_set.append(new_val)
                    new_set.sort()
        else:
            # Remove an element
            if n > 3:
                idx = random.randint(1, n - 1)
                new_set.pop(idx)
        
        if not new_set or 0 not in new_set:
            continue
            
        new_score = fast_score(new_set)
        if new_score < 0:
            new_score = score_set(new_set)
        
        if new_score > current_score:
            current = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best = current[:]
        else:
            # Accept with probability
            delta = current_score - new_score
            if delta < 0:
                prob = 1.0
            else:
                prob = pow(2.71828, -delta / temp)
            if random.random() < prob:
                current = new_set
                current_score = new_score
        
        temp *= cooling

    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    best_set = [0, 1, 3, 7, 15]
    best_score = fast_score(best_set)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Geometric sequences (2^k - 1, 3^k, etc)
    for base in [2, 3, 4, 5]:
        for k in range(1, 12):
            U = [0]
            val = 1
            for _ in range(k):
                U.append(val)
                val = val * base
            candidates.append(U)
            
    # 2. Exponential growth 2^k - 1
    for k in range(1, 12):
        U = [0]
        val = 1
        for _ in range(k):
            U.append(val)
            val = 2 * val + 1
        candidates.append(U)
        
    # 3. Random sparse sets
    random.seed(42)
    for _ in range(20):
        n_elem = random.randint(4, 15)
        max_val = random.randint(50, 500)
        U = [0] + sorted(random.sample(range(1, max_val), n_elem - 1))
        candidates.append(U)
        
    # 4. Structured: Pairs of close numbers
    for k in range(3, 8):
        U = [0]
        val = 1
        for i in range(k):
            U.append(val)
            U.append(val + 1)
            val = val * 3 + 1
        candidates.append(U)

    # Evaluate initial scores
    scored_candidates = []
    for U in candidates:
        s = fast_score(U)
        if s > 0:
            scored_candidates.append((s, U))
    scored_candidates.sort(reverse=True)
    
    # Run simulated annealing on top candidates
    for i, (s, U) in enumerate(scored_candidates[:20]):
        if time.time() > deadline - 10:
            break
        
        # Refine
        refined, refined_score = simulated_annealing(U, deadline)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined
            
        # Also try a quick hill climb on the best so far if time permits
        if i % 5 == 0 and time.time() < deadline - 5:
            refined_best, refined_best_score = simulated_annealing(best_set, deadline)
            if refined_best_score > best_score:
                best_score = refined_best_score
                best_set = refined_best

    return best_set
