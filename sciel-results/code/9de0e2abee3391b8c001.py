import time
from math import log, floor
import random
import numpy as np
from itertools import combinations

def score_set(U):
    """Compute the score for a set U."""
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

def local_search_dynamic(U, deadline, max_val_limit=2000, max_n_limit=50):
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
    max_no_improvement = 20
    
    while time.time() < deadline - 2:
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        move_type = random.random()
        
        # Update candidate pool periodically or if empty
        if not candidates or random.random() < 0.1:
            current_max = max(U)
            new_limit = max(max_val_limit, current_max * 2)
            candidates = generate_candidates(U, new_limit)
        
        if move_type < 0.5:
            # Try to ADD an element
            if len(U) < max_n_limit and candidates:
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

def digit_based_local_search(U, deadline):
    """
    Local search that treats the problem as a discrete spectral optimization.
    Parameterizes the set U via a base-b expansion with variable digit sets.
    We try to perturb individual elements in a way that respects the "spectral" nature
    by looking at the base-b representation.
    """
    U = sorted(set(U))
    current_score = score_set(U)
    best_score = current_score
    best_set = list(U)
    
    # Determine a base b based on the current max element
    if len(U) == 1:
        return best_score, best_set
    
    max_val = max(U)
    # Choose base such that digits are small but cover the range
    # If max_val is small, use base 2 or 3. If large, use larger base.
    b = max(2, int(log(max_val, 10)) + 2) if max_val > 10 else 2
    
    # Precompute powers of b
    powers = []
    p = 1
    while p <= max_val * 2:
        powers.append(p)
        p *= b
        
    no_improvement = 0
    max_no_improvement = 30
    
    while time.time() < deadline - 5:
        if no_improvement > max_no_improvement:
            break
            
        improved = False
        # Pick a random element to perturb
        idx = random.randint(1, len(U) - 1)
        old_val = U[idx]
        
        # Generate new values by perturbing digits in base b
        # Get digits of old_val
        digits = []
        temp = old_val
        for p in powers:
            digits.append(temp % b)
            temp //= b
            if temp == 0:
                break
        
        # Try a few perturbations
        for _ in range(5):
            new_digits = digits[:]
            if not new_digits:
                new_digits = [0]
            
            # Perturb one digit
            d_idx = random.randint(0, len(new_digits) - 1)
            # Change digit to something close or random
            if random.random() < 0.5:
                new_digits[d_idx] = max(0, min(b-1, new_digits[d_idx] + random.randint(-2, 2)))
            else:
                new_digits[d_idx] = random.randint(0, b-1)
                
            # Reconstruct value
            new_val = 0
            for i, d in enumerate(new_digits):
                if i < len(powers):
                    new_val += d * powers[i]
                else:
                    new_val += d * (powers[-1] * b)
                    
            if new_val <= 0 or new_val in U:
                continue
                
            # Ensure we don't make the set too large or small
            if new_val > max_val * 3:
                continue
                
            new_U = U.copy()
            new_U[idx] = new_val
            new_U = sorted(set(new_U))
            
            if len(new_U) != len(U):
                continue
                
            new_score = score_set(new_U)
            if new_score > current_score:
                U = new_U
                current_score = new_score
                improved = True
                break
        else:
            # Try a random swap or replacement as fallback
            if len(U) > 1:
                idx2 = random.randint(1, len(U) - 1)
                if idx2 != idx:
                    new_U = U.copy()
                    new_U[idx], new_U[idx2] = new_U[idx2], new_U[idx]
                    new_U = sorted(set(new_U))
                    if len(new_U) == len(U):
                        new_score = score_set(new_U)
                        if new_score > current_score:
                            U = new_U
                            current_score = new_score
                            improved = True
        
        if improved:
            no_improvement = 0
            if current_score > best_score:
                best_score = current_score
                best_set = list(U)
        else:
            no_improvement += 1
            
    return best_score, best_set

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

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
        [0, 1, 4, 9, 11], # Known good small set
    ]
    
    for cand in structured_candidates:
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # 2. Exhaustive Search for Small n
    # This is a powerful strategy for small sets where local search might get stuck in local minima.
    
    def exhaustive_search(n, max_val):
        """Find best set of size n (including 0) with elements <= max_val."""
        best_s = -1
        best_c = None
        range_vals = range(1, max_val + 1)
        for comb in combinations(range_vals, n - 1):
            U = [0] + list(comb)
            s = score_set(U)
            if s > best_s:
                best_s = s
                best_c = U
        return best_s, best_c

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

    # 3. Geometric Search
    # Quick coarse grid to find a good geometric seed
    b_candidates = [b_int / 10.0 for b_int in range(12, 101)]
    best_b = None
    best_m = None
    best_grid_score = -1

    for b in b_candidates:
        if time.time() > deadline - 40:
            break
        for m in range(2, 21):
            U = [0]
            seen = {0}
            for k in range(1, m + 1):
                val = floor(b ** k)
                if val <= 0:
                    val = 1
                if val not in seen:
                    U.append(val)
                    seen.add(val)
            U.sort()
            s = score_set(U)
            if s > best_grid_score:
                best_grid_score = s
                best_b = b
                best_m = m
                best_set = U
                best_score = s

    # 4. Local Search from Best Seeds
    search_seeds = [best_set]
    
    # Add top structured candidates as additional seeds
    for cand in structured_candidates:
        if cand != best_set:
            search_seeds.append(cand)
            
    # Add a few random perturbations of the best set
    for _ in range(5):
        pert = best_set[:]
        if len(pert) > 1:
            idx = random.randint(1, len(pert)-1)
            pert[idx] += random.randint(-5, 5)
            if pert[idx] > 0:
                pert = sorted(set(pert))
                if len(pert) >= 2 and 0 in pert:
                    search_seeds.append(pert)

    # 5. Hybrid Local Search Phase
    # Alternate between standard local search and digit-based local search
    for seed in search_seeds:
        if time.time() > deadline - 10:
            break
        
        # Run standard local search
        local_score, local_set = local_search_dynamic(seed, deadline)
        if local_score > best_score:
            best_score = local_score
            best_set = local_set
            
        # Run digit-based local search on the result of standard search
        if time.time() < deadline - 10:
            digit_score, digit_set = digit_based_local_search(local_set, deadline)
            if digit_score > best_score:
                best_score = digit_score
                best_set = digit_set

    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
