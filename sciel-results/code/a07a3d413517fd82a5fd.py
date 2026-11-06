import time
from math import gcd
from itertools import combinations

# Precompute prime factors for 1..511
def get_prime_factors(n):
    factors = set()
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

# Precompute for all numbers
PRIME_FACTORS = [set() for _ in range(512)]
for i in range(1, 512):
    PRIME_FACTORS[i] = get_prime_factors(i)

# Precompute values
VALUES = [0.0] * 512
for i in range(1, 512):
    VALUES[i] = 1.0 / (512 - i)

# Precompute conflict masks (bitmasks for speed)
# There are 511 numbers. We'll use a list of sets of conflicting numbers.
CONFLICTS = [set() for _ in range(512)]
for i in range(1, 512):
    for j in range(i + 1, 512):
        if gcd(i, j) > 1:
            CONFLICTS[i].add(j)
            CONFLICTS[j].add(i)

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin for safety
    
    # 1. Greedy Initialization
    # Sort candidates by value descending
    candidates = list(range(1, 512))
    candidates.sort(key=lambda x: VALUES[x], reverse=True)
    
    chosen = []
    prime_counts = {}
    
    for a in candidates:
        valid = True
        for p in PRIME_FACTORS[a]:
            if prime_counts.get(p, 0) > 0:
                valid = False
                break
        if valid:
            chosen.append(a)
            for p in PRIME_FACTORS[a]:
                prime_counts[p] = prime_counts.get(p, 0) + 1
                
    current_score = sum(VALUES[a] for a in chosen)
    best_set = set(chosen)
    best_prime_counts = dict(prime_counts)
    best_score = current_score
    
    def get_score(s):
        return sum(VALUES[a] for a in s)
    
    def is_valid_set(s):
        # Check pairwise coprime
        for i in range(len(s)):
            for j in range(i+1, len(s)):
                if gcd(s[i], s[j]) > 1:
                    return False
        return True
        
    def try_add(a, set_to_check, counts_to_check):
        for p in PRIME_FACTORS[a]:
            if counts_to_check.get(p, 0) > 0:
                return None
        return VALUES[a]
        
    def try_swap_1_1(a_out, b_in, set_to_check, counts_to_check):
        for p in PRIME_FACTORS[a_out]:
            counts_to_check[p] -= 1
            if counts_to_check[p] == 0:
                del counts_to_check[p]
                
        gain = try_add(b_in, set_to_check - {a_out}, counts_to_check)
        
        for p in PRIME_FACTORS[a_out]:
            counts_to_check[p] = counts_to_check.get(p, 0) + 1
            
        if gain is not None:
            return gain - VALUES[a_out]
        return None
        
    def try_swap_1_2(a_out, b_in, c_in, set_to_check, counts_to_check):
        for p in PRIME_FACTORS[a_out]:
            counts_to_check[p] -= 1
            if counts_to_check[p] == 0:
                del counts_to_check[p]
                
        gain_b = try_add(b_in, set_to_check - {a_out}, counts_to_check)
        if gain_b is None:
            for p in PRIME_FACTORS[a_out]:
                counts_to_check[p] = counts_to_check.get(p, 0) + 1
            return None
            
        for p in PRIME_FACTORS[b_in]:
            counts_to_check[p] = counts_to_check.get(p, 0) + 1
            
        gain_c = try_add(c_in, (set_to_check - {a_out}) | {b_in}, counts_to_check)
        
        for p in PRIME_FACTORS[b_in]:
            counts_to_check[p] -= 1
            if counts_to_check[p] == 0:
                del counts_to_check[p]
        for p in PRIME_FACTORS[a_out]:
            counts_to_check[p] = counts_to_check.get(p, 0) + 1
            
        if gain_c is not None:
            return gain_b + gain_c - VALUES[a_out]
        return None
        
    def try_swap_2_1(a_out1, a_out2, b_in, set_to_check, counts_to_check):
        for p in PRIME_FACTORS[a_out1]:
            counts_to_check[p] -= 1
            if counts_to_check[p] == 0:
                del counts_to_check[p]
        for p in PRIME_FACTORS[a_out2]:
            counts_to_check[p] -= 1
            if counts_to_check[p] == 0:
                del counts_to_check[p]
                
        gain = try_add(b_in, set_to_check - {a_out1, a_out2}, counts_to_check)
        
        for p in PRIME_FACTORS[a_out1]:
            counts_to_check[p] = counts_to_check.get(p, 0) + 1
        for p in PRIME_FACTORS[a_out2]:
            counts_to_check[p] = counts_to_check.get(p, 0) + 1
            
        if gain is not None:
            return gain - VALUES[a_out1] - VALUES[a_out2]
        return None

    # Local Search Loop
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
            
        current_set = set(best_set)
        current_counts = dict(best_prime_counts)
        current_score_val = best_score
        
        # 1. Try adding a single element
        best_add = None
        best_add_gain = 0
        for a in range(1, 512):
            if a not in current_set:
                gain = try_add(a, current_set, current_counts)
                if gain is not None and gain > best_add_gain:
                    best_add_gain = gain
                    best_add = a
                    
        if best_add is not None:
            current_set.add(best_add)
            for p in PRIME_FACTORS[best_add]:
                current_counts[p] = current_counts.get(p, 0) + 1
            current_score_val += best_add_gain
            if current_score_val > best_score:
                best_score = current_score_val
                best_set = current_set
                best_prime_counts = current_counts
            improved = True
            continue
            
        # 2. Try swapping 1 element with 1 element
        best_swap = None
        best_swap_gain = 0
        for a_out in list(current_set):
            for b_in in range(1, 512):
                if b_in not in current_set:
                    gain = try_swap_1_1(a_out, b_in, current_set, current_counts)
                    if gain is not None and gain > best_swap_gain:
                        best_swap_gain = gain
                        best_swap = (a_out, b_in)
                        
        if best_swap is not None:
            a_out, b_in = best_swap
            current_set.remove(a_out)
            for p in PRIME_FACTORS[a_out]:
                current_counts[p] -= 1
                if current_counts[p] == 0:
                    del current_counts[p]
            current_set.add(b_in)
            for p in PRIME_FACTORS[b_in]:
                current_counts[p] = current_counts.get(p, 0) + 1
            current_score_val += best_swap_gain
            if current_score_val > best_score:
                best_score = current_score_val
                best_set = current_set
                best_prime_counts = current_counts
            improved = True
            continue
            
        # 3. Try swapping 1 element with 2 elements (limited candidates)
        candidates_sorted = sorted(range(1, 512), key=lambda x: VALUES[x], reverse=True)
        top_candidates = candidates_sorted[:150]
        
        best_1_for_2 = None
        best_1_for_2_gain = 0
        for a_out in list(current_set):
            for i, b_in in enumerate(top_candidates):
                if b_in in current_set: continue
                for c_in in top_candidates[i+1:]:
                    if c_in in current_set: continue
                    gain = try_swap_1_2(a_out, b_in, c_in, current_set, current_counts)
                    if gain is not None and gain > best_1_for_2_gain:
                        best_1_for_2_gain = gain
                        best_1_for_2 = (a_out, b_in, c_in)
                        
        if best_1_for_2 is not None:
            a_out, b_in, c_in = best_1_for_2
            current_set.remove(a_out)
            for p in PRIME_FACTORS[a_out]:
                current_counts[p] -= 1
                if current_counts[p] == 0:
                    del current_counts[p]
            current_set.add(b_in)
            for p in PRIME_FACTORS[b_in]:
                current_counts[p] = current_counts.get(p, 0) + 1
            current_set.add(c_in)
            for p in PRIME_FACTORS[c_in]:
                current_counts[p] = current_counts.get(p, 0) + 1
            current_score_val += best_1_for_2_gain
            if current_score_val > best_score:
                best_score = current_score_val
                best_set = current_set
                best_prime_counts = current_counts
            improved = True
            continue
            
        # 4. Try swapping 2 elements with 1 element
        best_2_for_1 = None
        best_2_for_1_gain = 0
        set_list = list(current_set)
        for i in range(len(set_list)):
            for j in range(i+1, len(set_list)):
                a_out1 = set_list[i]
                a_out2 = set_list[j]
                for b_in in range(1, 512):
                    if b_in in current_set: continue
                    gain = try_swap_2_1(a_out1, a_out2, b_in, current_set, current_counts)
                    if gain is not None and gain > best_2_for_1_gain:
                        best_2_for_1_gain = gain
                        best_2_for_1 = (a_out1, a_out2, b_in)
                        
        if best_2_for_1 is not None:
            a_out1, a_out2, b_in = best_2_for_1
            current_set.remove(a_out1)
            for p in PRIME_FACTORS[a_out1]:
                current_counts[p] -= 1
                if current_counts[p] == 0:
                    del current_counts[p]
            current_set.remove(a_out2)
            for p in PRIME_FACTORS[a_out2]:
                current_counts[p] -= 1
                if current_counts[p] == 0:
                    del current_counts[p]
            current_set.add(b_in)
            for p in PRIME_FACTORS[b_in]:
                current_counts[p] = current_counts.get(p, 0) + 1
            current_score_val += best_2_for_1_gain
            if current_score_val > best_score:
                best_score = current_score_val
                best_set = current_set
                best_prime_counts = current_counts
            improved = True
            continue

        # 5. Perturbation: Random restarts
        if len(current_set) > 5:
            to_remove = random.sample(list(current_set), 5)
            for a in to_remove:
                current_set.remove(a)
                for p in PRIME_FACTORS[a]:
                    current_counts[p] -= 1
                    if current_counts[p] == 0:
                        del current_counts[p]
                        
            for a in candidates_sorted:
                if a not in current_set:
                    gain = try_add(a, current_set, current_counts)
                    if gain is not None:
                        current_set.add(a)
                        for p in PRIME_FACTORS[a]:
                            current_counts[p] = current_counts.get(p, 0) + 1
                            
            current_score_val = get_score(current_set)
            if current_score_val > best_score:
                best_score = current_score_val
                best_set = current_set
                best_prime_counts = dict(current_counts)
            improved = True
            continue

        break
        
    return list(best_set)

import random
