import time
from math import log, floor
import random
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
    
    U_list = U
    n_len = n
    
    for i in range(n_len):
        ui = U_list[i]
        for j in range(i, n_len):
            sums.add(ui + U_list[j])
        for j in range(i + 1, n_len):
            d = U_list[j] - ui
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
        
    max_u = U_list[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def geometric_set(b, m):
    """Generate U = {floor(b^k) for k=0..m}, ensuring 0 is included and all are nonnegative distinct integers."""
    U = [0]
    seen = {0}
    for k in range(1, m + 1):
        val = floor(b ** k)
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
    max_no_improvement = 40
    
    while time.time() < deadline - 2:
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        move_type = random.random()
        
        if not candidates or random.random() < 0.05:
            current_max = max(U)
            new_limit = max(max_val_limit, current_max * 2)
            candidates = generate_candidates(U, new_limit)
        
        if move_type < 0.5:
            if len(U) < max_n_limit and candidates:
                random.shuffle(candidates)
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
        elif move_type < 0.7:
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
        elif move_type < 0.85:
            if len(U) > 1 and candidates:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                random.shuffle(candidates)
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
            if len(U) > 2 and candidates:
                idx1 = random.randint(1, len(U) - 1)
                idx2 = random.randint(1, len(U) - 1)
                while idx2 == idx1 and len(U) > 2:
                    idx2 = random.randint(1, len(U) - 1)
                
                val1 = U[idx1]
                val2 = U[idx2]
                
                pool = [c for c in candidates if c != val1 and c != val2 and c not in U]
                if len(pool) >= 2:
                    random.shuffle(pool)
                    new_val1 = pool[0]
                    new_val2 = pool[1]
                    
                    if new_val1 == new_val2:
                        if len(pool) >= 3:
                            new_val2 = pool[2]
                        else:
                            continue

                    new_U = U.copy()
                    new_U[idx1] = new_val1
                    new_U[idx2] = new_val2
                    new_U = sorted(set(new_U))
                    
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        candidates = generate_candidates(U, max(max_val_limit, max(U)*2))

        if improved:
            no_improvement_count = 0
            if current_score > best_local_score:
                best_local_score = current_score
                best_local_set = list(U)
        else:
            no_improvement_count += 1
                
    return best_local_score, best_local_set

def digit_based_search(U, deadline, max_val_limit=5000):
    """
    Search using digit-based parameterization.
    Represents elements in mixed-radix form and perturbs individual digits.
    """
    U = sorted(set(U))
    current_score = score_set(U)
    best_score = current_score
    best_set = list(U)
    
    # Determine number of digits needed
    if not U or max(U) == 0:
        return current_score, best_set
    
    max_val = max(U)
    # Use base 2 for maximum flexibility
    base = 2
    num_digits = max_val.bit_length()
    
    # Represent each element as a list of digits
    def to_digits(val, num_digits, base):
        digits = [0] * num_digits
        for i in range(num_digits):
            digits[i] = val % base
            val //= base
        return digits
    
    def from_digits(digits, base):
        val = 0
        for i in range(len(digits) - 1, -1, -1):
            val = val * base + digits[i]
        return val
    
    # Initialize digit representations
    current_digits = [to_digits(x, num_digits, base) for x in U]
    
    no_improvement = 0
    max_no_improvement = 60
    
    while time.time() < deadline - 3 and no_improvement < max_no_improvement:
        improved = False
        
        # Strategy 1: Perturb a single digit of one element
        if random.random() < 0.4:
            elem_idx = random.randint(1, len(U) - 1)  # Skip index 0 (which is 0)
            digit_idx = random.randint(0, min(num_digits - 1, 12))
            old_digit = current_digits[elem_idx][digit_idx]
            new_digit = (old_digit + random.choice([-1, 1])) % base
            
            if new_digit != old_digit:
                current_digits[elem_idx][digit_idx] = new_digit
                new_val = from_digits(current_digits[elem_idx], base)
                
                if new_val > 0 and new_val <= max_val_limit:
                    new_U = list(U)
                    new_U[elem_idx] = new_val
                    new_U = sorted(set(new_U))
                    
                    # Check if size changed (collision)
                    if len(new_U) == len(U):
                        new_score = score_set(new_U)
                        if new_score > current_score:
                            U = new_U
                            current_score = new_score
                            current_digits = [to_digits(x, num_digits, base) for x in U]
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_set = list(U)
                    else:
                        # Revert if collision
                        current_digits[elem_idx][digit_idx] = old_digit
        else:
            # Strategy 2: Perturb multiple digits simultaneously
            num_perturb = random.randint(1, 3)
            perturbed = False
            
            for _ in range(num_perturb):
                elem_idx = random.randint(1, len(U) - 1)
                digit_idx = random.randint(0, min(num_digits - 1, 12))
                old_digit = current_digits[elem_idx][digit_idx]
                new_digit = (old_digit + random.choice([-1, 1])) % base
                
                if new_digit != old_digit:
                    current_digits[elem_idx][digit_idx] = new_digit
                    perturbed = True
            
            if perturbed:
                # Check all elements
                valid = True
                new_U = [0]
                for i in range(1, len(U)):
                    new_val = from_digits(current_digits[i], base)
                    if new_val <= 0 or new_val > max_val_limit or new_val in new_U:
                        valid = False
                        break
                    new_U.append(new_val)
                
                if valid and len(new_U) == len(U):
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_set = list(U)
                    else:
                        # Revert all perturbations
                        for i in range(1, len(U)):
                            current_digits[i] = to_digits(U[i], num_digits, base)
        
        if improved:
            no_improvement = 0
        else:
            no_improvement += 1
    
    return best_score, best_set

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    best_set = [0, 1, 4, 9, 11]
    best_score = score_set(best_set)

    # 1. Structured Candidates
    structured_candidates = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81],
        [0, 1, 2, 4, 8, 16, 32, 64, 128],
        [0, 1, 3, 7, 15, 31, 63, 127],
        [0, 1, 4, 10, 20, 35, 56, 84],
        [0, 1, 3, 4, 9, 10, 12, 13],
        [0, 1, 4, 10, 18, 28, 40, 54],
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256],
        [0, 1, 3, 9, 27, 81, 243],
        [0, 1, 2, 5, 13, 34, 89],
        [0, 1, 5, 13, 25, 41, 61],
    ]
    
    for cand in structured_candidates:
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # 2. Exhaustive Search for Small n
    def exhaustive_search(n, max_val):
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

    if time.time() < deadline - 60:
        s, c = exhaustive_search(3, 60)
        if s > best_score:
            best_score = s
            best_set = c
        
        s, c = exhaustive_search(4, 50)
        if s > best_score:
            best_score = s
            best_set = c
            
        s, c = exhaustive_search(5, 40)
        if s > best_score:
            best_score = s
            best_set = c

    # 3. Geometric Search
    b_candidates = [b_int / 10.0 for b_int in range(12, 101)]
    best_b = None
    best_m = None
    best_grid_score = -1

    for b in b_candidates:
        if time.time() > deadline - 45:
            break
        for m in range(2, 21):
            U = geometric_set(b, m)
            if len(U) < 2:
                continue
            s = score_set(U)
            if s > best_grid_score:
                best_grid_score = s
                best_b = b
                best_m = m
                if s > best_score:
                    best_score = s
                    best_set = U

    # 4. Dynamic Local Search
    seeds = [best_set]
    for cand in structured_candidates[:3]:
        if cand != best_set:
            seeds.append(cand)
    
    max_val_limit = 800
    max_n_limit = 35
    
    restarts = 0
    while time.time() < deadline - 30:
        restarts += 1
        base = seeds[restarts % len(seeds)]
        U = list(base)
        
        local_score, local_set = local_search_dynamic(U, deadline, max_val_limit, max_n_limit)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set
            if len(seeds) < 6:
                seeds.append(local_set)
            else:
                seeds[-1] = local_set

    # 5. Digit-based search (new approach)
    if time.time() < deadline - 10:
        digit_score, digit_set = digit_based_search(best_set, deadline, max_val_limit=5000)
        if digit_score > best_score:
            best_score = digit_score
            best_set = digit_set

    # 6. Final Polish
    if time.time() < deadline - 2:
        current = list(best_set)
        current_score = best_score
        improved = True
        iteration = 0
        while improved and time.time() < deadline - 1 and iteration < 500:
            improved = False
            iteration += 1
            n = len(current)
            for i in range(1, n):
                if time.time() > deadline - 1:
                    break
                old_val = current[i]
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
            
            if not improved and n < 30 and time.time() < deadline - 1:
                max_val = max(current)
                for new_val in range(max_val - 20, max_val + 21):
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
