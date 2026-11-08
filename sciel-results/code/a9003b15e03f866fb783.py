import time
import random
import math
from itertools import combinations

def get_score(U):
    if not U:
        return 0.0
    U = set(U)
    if 0 not in U:
        return 0.0
    # Ensure non-negative
    if any(x < 0 for x in U):
        return 0.0
    
    sums = set()
    diffs = set()
    
    # Optimization: Use list for faster iteration if U is small
    U_list = list(U)
    n = len(U_list)
    
    for a in U_list:
        for b in U_list:
            sums.add(a + b)
            diffs.add(a - b)
            
    sum_size = len(sums)
    diff_size = len(diffs)
    
    if sum_size == 0:
        return 0.0
        
    ratio = diff_size / sum_size
    max_u = max(U)
    denom = math.log(2 * max_u + 1)
    
    if denom == 0:
        return 1.0
        
    score = 1.0 + math.log(ratio) / denom
    return score

def mian_chowla(n):
    seq = [0, 1]
    for i in range(2, n):
        last = seq[-1]
        candidate = last + 1
        while True:
            ok = True
            # Check if candidate creates duplicate sums
            # Sums of existing pairs + candidate
            # We need to ensure all sums are distinct
            # For Mian-Chowla, we just need to check against existing sums
            existing_sums = set()
            for a in seq:
                for b in seq:
                    existing_sums.add(a + b)
            
            # Check sums involving candidate
            for a in seq:
                s = a + candidate
                if s in existing_sums:
                    ok = False
                    break
            if not ok:
                candidate += 1
                continue
            
            # Also need to ensure candidate + candidate doesn't clash? 
            # The standard definition ensures all sums x+y are distinct for x<=y.
            # The above check covers a+candidate for a in seq.
            # We also need candidate+candidate.
            if candidate + candidate in existing_sums:
                ok = False
            
            if ok:
                seq.append(candidate)
                break
            else:
                candidate += 1
    return seq

def score_set(U):
    """Compute the score for a set U. Same as get_score but returns -1 for invalid."""
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
        
    return 1 + math.log(ratio) / math.log(D)

def geometric_set(b, m):
    """Generate U = {floor(b^k) for k=0..m}, ensuring 0 is included and all are nonnegative distinct integers."""
    U = [0]
    seen = {0}
    for k in range(1, m + 1):
        val = int(b ** k)
        if val <= 0:
            val = 1
        if val not in seen:
            U.append(val)
            seen.add(val)
    return sorted(U)

def generate_candidates(U, limit_val):
    """
    Generate candidate elements for addition based on sums and differences of existing elements.
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
            
    return list(candidates)

def local_search_dynamic(U, deadline, max_val_limit=1000, max_n_limit=40):
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
    max_no_improvement = 30
    
    while time.time() < deadline - 2:
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        move_type = random.random()
        
        # Update candidate pool periodically or if empty
        if not candidates or random.random() < 0.1:
            # Expand limit if we are growing
            current_max = max(U)
            new_limit = max(max_val_limit, current_max * 2)
            candidates = generate_candidates(U, new_limit)
        
        if move_type < 0.5:
            # Try to ADD an element
            if len(U) < max_n_limit and candidates:
                # Try a few random candidates
                random.shuffle(candidates)
                for c in candidates[:10]:
                    if c in U:
                        continue
                    new_U = U + [c]
                    new_U.sort()
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        # Regenerate candidates immediately as structure changed significantly
                        candidates = generate_candidates(U, max(max_val_limit, max(U)*2))
                        break
        elif move_type < 0.7:
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
        else:
            # Try to REPLACE an element
            if len(U) > 1 and candidates:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                # Pick a random candidate
                random.shuffle(candidates)
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

def exhaustive_search(n, max_val):
    """Find best set of size n (including 0) with elements <= max_val."""
    best_s = -1
    best_c = None
    # We pick n-1 elements from 1..max_val
    range_vals = range(1, max_val + 1)
    for comb in combinations(range_vals, n - 1):
        U = [0] + list(comb)
        s = score_set(U)
        if s > best_s:
            best_s = s
            best_c = U
    return best_s, best_c

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    best_set = [0, 1, 4, 9, 11]
    best_score = score_set(best_set)
    
    # 1. Structured Candidates
    structured_candidates = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81], # Squares
        [0, 1, 2, 4, 8, 16, 32, 64, 128], # Powers of 2
        [0, 1, 3, 7, 15, 31, 63, 127], # Mersenne-like
        [0, 1, 4, 10, 20, 35, 56, 84], # Triangular
        [0, 1, 3, 4, 9, 10, 12, 13], # Small Sidon-like
        [0, 1, 4, 10, 18, 28, 40, 54], # Quadratic-ish
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256], # Larger powers of 2
        [0, 1, 3, 9, 27, 81, 243], # Powers of 3
        [0, 1, 4, 9, 11], # Known good
        [0, 1, 2, 5, 7],
        [0, 1, 3, 7, 9],
    ]
    
    for cand in structured_candidates:
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
    
    # 2. Exhaustive Search for Small n
    # This is a powerful strategy for small sets where local search might get stuck in local minima.
    
    if time.time() < deadline - 50:
        # n=3
        s, c = exhaustive_search(3, 50)
        if s > best_score:
            best_score = s
            best_set = c
        
        # n=4
        s, c = exhaustive_search(4, 40)
        if s > best_score:
            best_score = s
            best_set = c
            
        # n=5
        s, c = exhaustive_search(5, 30)
        if s > best_score:
            best_score = s
            best_set = c
            
        # n=6
        s, c = exhaustive_search(6, 20)
        if s > best_score:
            best_score = s
            best_set = c

    # 3. Local Search from Best Seed
    if time.time() < deadline - 5:
        ls_score, ls_set = local_search_dynamic(best_set, deadline)
        if ls_score > best_score:
            best_score = ls_score
            best_set = ls_set

    # 4. Random Restarts and Local Search
    # Use remaining time for random restarts
    while time.time() < deadline - 2:
        # Generate random set
        # Size between 3 and 10
        target_n = random.randint(3, 10)
        # Max value around 50-100
        max_val = random.randint(50, 100)
        
        rand_set = set([0])
        while len(rand_set) < target_n:
            rand_set.add(random.randint(1, max_val))
        rand_list = list(rand_set)
        
        s = score_set(rand_list)
        if s > best_score:
            best_score = s
            best_set = rand_list[:]
            
        # Local search on random start
        # Only do a quick local search if we have time
        if time.time() < deadline - 5:
            ls_score, ls_set = local_search_dynamic(rand_list, min(deadline - 2, time.time() + 2))
            if ls_score > best_score:
                best_score = ls_score
                best_set = ls_set

    return best_set

if __name__ == "__main__":
    # Quick self-test
    U = construct()
    print("Best set:", U)
    print("Score:", get_score(U))
