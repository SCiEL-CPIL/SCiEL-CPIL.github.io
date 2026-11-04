import time
from math import log, floor, sqrt
import random
import numpy as np

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    sums = set()
    diffs = set()
    
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def generate_candidates(U, limit_val):
    """
    Generate candidate elements for addition based on sums, differences, and scaling.
    Strictly arithmetic to avoid noise (based on observation).
    """
    U_set = set(U)
    candidates = set()
    
    for a in U:
        for b in U:
            s = a + b
            d = abs(a - b)
            if 0 < s <= limit_val and s not in U_set:
                candidates.add(s)
            if 0 < d <= limit_val and d not in U_set:
                candidates.add(d)
            
            # Scaling factors to explore geometric/binary structures
            for factor in [2, 3, 1/2, 1/3]:
                val = int(a * factor)
                if val > 0 and val <= limit_val and val not in U_set:
                    candidates.add(val)
                # Also try a + a*factor
                val2 = a + int(a * (factor - 1))
                if val2 > 0 and val2 <= limit_val and val2 not in U_set:
                    candidates.add(val2)
            
    return list(candidates)

def local_search_dynamic(U, deadline, max_val_limit=2000, max_n_limit=35):
    """
    Local search with dynamic candidate generation.
    """
    U = sorted(set(U))
    current_score = score_set(U)
    best_local_score = current_score
    best_local_set = list(U)
    
    # Initial candidate pool
    candidates = generate_candidates(U, max_val_limit)
    
    no_improvement_count = 0
    max_no_improvement = 50 # Allow more exploration
    
    while time.time() < deadline - 5:
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        move_type = random.random()
        
        # Update candidate pool periodically or if empty
        if not candidates or random.random() < 0.05:
            current_max = max(U)
            new_limit = max(max_val_limit, current_max * 2)
            candidates = generate_candidates(U, new_limit)
            
        # Shuffle to ensure diversity
        random.shuffle(candidates)
        
        if move_type < 0.4:
            # Try to ADD an element
            if len(U) < max_n_limit and candidates:
                for c in candidates[:15]:
                    if c in U:
                        continue
                    new_U = U + [c]
                    new_U.sort()
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
        elif move_type < 0.6:
            # Try to REMOVE an element
            if len(U) > 2:
                removable = [x for x in U if x != 0]
                if removable:
                    c = random.choice(removable)
                    new_U = [x for x in U if x != c]
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates(U, max_val_limit)
        elif move_type < 0.8:
            # Try to REPLACE an element
            if len(U) > 1 and candidates:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                for new_val in candidates[:15]:
                    if new_val == old_val or new_val in U:
                        continue
                    new_U = U.copy()
                    new_U[idx] = new_val
                    new_U.sort()
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
        else:
            # Try SWAP: Replace one with another from pool, or swap two existing
            if len(U) > 2 and candidates:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                for new_val in candidates[:10]:
                    if new_val == old_val or new_val in U:
                        continue
                    new_U = U.copy()
                    new_U[idx] = new_val
                    new_U.sort()
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
            
        if improved:
            no_improvement_count = 0
            if current_score > best_local_score:
                best_local_score = current_score
                best_local_set = list(U)
        else:
            no_improvement_count += 1
                
    return best_local_score, best_local_set

def generate_geometric_set(b, m):
    U = [0]
    val = 1.0
    for k in range(1, m + 1):
        U.append(int(val))
        val *= b
    return sorted(set(U))

def generate_digit_subset(base, allowed_digits, length, num_samples=50):
    """
    Generate sets where elements are formed by choosing digits from allowed_digits
    in a base 'base' representation of fixed length.
    """
    sets = []
    max_val = base ** length
    for _ in range(num_samples):
        U = {0}
        for _ in range(random.randint(2, 10)): # 2 to 10 elements
            digits = [random.choice(allowed_digits) for _ in range(length)]
            val = 0
            for d in reversed(digits):
                val = val * base + d
            if val > 0 and val < max_val:
                U.add(val)
        if len(U) >= 2:
            sets.append(sorted(U))
    return sets

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    best_set = [0, 1, 4, 9, 11]
    best_score = score_set(best_set)

    seeds = []

    # 1. Geometric Seeds
    for b in [1.5, 1.8, 2.0, 2.2, 2.5, 3.0, 4.0]:
        for m in range(4, 16):
            U = generate_geometric_set(b, m)
            if len(U) < 2: continue
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U
            if s > 1.0:
                seeds.append(U)

    # 2. Digit Subset Seeds (Binary-like, etc.)
    # Base 2, digits {0,1} -> Powers of 2 subsets
    # Base 3, digits {0,1}
    # Base 10, digits {0,1}
    for base, digits in [(2, [0,1]), (3, [0,1]), (10, [0,1]), (2, [0,1,2]), (3, [0,1,2])]:
        if time.time() > deadline - 60: break
        for length in [6, 7, 8, 9, 10, 11, 12]:
            if time.time() > deadline - 50: break
            candidates = generate_digit_subset(base, digits, length, num_samples=20)
            for U in candidates:
                s = score_set(U)
                if s > best_score:
                    best_score = s
                    best_set = U
                if s > 1.0:
                    seeds.append(U)

    # Ensure best_set is in seeds
    if best_set not in seeds:
        seeds.append(best_set)
    else:
        seeds.remove(best_set)
        seeds.insert(0, best_set)

    # 3. Local Search
    max_val_limit = 3000
    max_n_limit = 35
    
    restarts = 0
    while time.time() < deadline - 20:
        restarts += 1
        # Cycle through seeds, prioritizing best ones
        base = seeds[restarts % len(seeds)]
        U = list(base)
        
        local_score, local_set = local_search_dynamic(U, deadline, max_val_limit, max_n_limit)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set
            # Update seeds: replace worst seed or add new best
            if len(seeds) < 8:
                seeds.append(local_set)
            else:
                # Replace the last one (which might be worse)
                seeds[-1] = local_set

    # 4. Final Polish
    if time.time() < deadline - 2:
        current = list(best_set)
        current_score = best_score
        improved = True
        iteration = 0
        while improved and time.time() < deadline - 1 and iteration < 200:
            improved = False
            iteration += 1
            n = len(current)
            # Try replacing each element with nearby values
            for i in range(1, n):
                if time.time() > deadline - 1:
                    break
                old_val = current[i]
                # Wider delta range for final polish
                for delta in range(-15, 16):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val <= 0:
                        continue
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
            
            # Try adding an element
            if not improved and n < 30 and time.time() < deadline - 1:
                max_val = max(current)
                for new_val in range(max_val - 15, max_val + 16):
                    if new_val <= 0:
                        continue
                    if new_val in current:
                        continue
                    new_set = sorted(current + [new_val])
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

            # Try removing an element
            if not improved and n > 3 and time.time() < deadline - 1:
                for i in range(1, n):
                    new_set = [current[j] for j in range(n) if j != i]
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

        if current_score > best_score:
            best_score = current_score
            best_set = current

    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
