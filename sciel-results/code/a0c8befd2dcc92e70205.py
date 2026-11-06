import random
import math
import time
from itertools import combinations

def is_pairwise_coprime(nums):
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(nums[i], nums[j]) != 1:
                return False
    return True

def evaluate_pair(A, B):
    """Return (cardinality, is_valid) for a pair of 3-element lists."""
    sums = [a + b for a in A for b in B]
    distinct_sums = list(set(sums))
    curr_cardinality = len(distinct_sums)
    if curr_cardinality < 6:
        return curr_cardinality, False
    if is_pairwise_coprime(distinct_sums):
        return curr_cardinality, True
    return curr_cardinality, False

def local_search(A, B, deadline, best_card):
    """Hill-climb from (A, B) by swapping one element at a time."""
    current_A = list(A)
    current_B = list(B)
    curr_card, curr_valid = evaluate_pair(current_A, current_B)
    
    all_nums = list(range(1, 129))
    no_improve = 0
    
    # Determine parity strategy: if current is parity-mismatched, enforce it
    # A all odd, B all even OR A all even, B all odd
    a_parity = [x % 2 for x in current_A]
    b_parity = [x % 2 for x in current_B]
    
    enforce_parity = False
    if len(set(a_parity)) == 1 and len(set(b_parity)) == 1 and a_parity[0] != b_parity[0]:
        enforce_parity = True
        target_a_parity = a_parity[0]
        target_b_parity = b_parity[0]
    
    while time.time() < deadline:
        if curr_valid and curr_card == 9:
            return list(current_A), list(current_B)
        
        improved = False
        
        # Filter candidates based on parity if enforcing
        if enforce_parity:
            cand_A = [x for x in all_nums if x % 2 == target_a_parity]
            cand_B = [x for x in all_nums if x % 2 == target_b_parity]
        else:
            cand_A = all_nums
            cand_B = all_nums

        # Try swapping each element of A with a random alternative
        for _ in range(50):
            idx = random.randint(0, 2)
            old_val = current_A[idx]
            # Choose new val from candidates, ensuring distinctness in A
            candidates = [x for x in cand_A if x != old_val and x not in (current_A[(idx+1)%3], current_A[(idx+2)%3])]
            if not candidates: continue
            new_val = random.choice(candidates)
            test_A = list(current_A)
            test_A[idx] = new_val
            card, valid = evaluate_pair(test_A, current_B)
            if valid and card > curr_card:
                current_A = test_A
                curr_card, curr_valid = card, True
                improved = True
                no_improve = 0
                if curr_card == 9:
                    return list(current_A), list(current_B)
                break
            elif valid and card == curr_card and random.random() < 0.1:
                current_A = test_A
                curr_card, curr_valid = card, True
                improved = True
                no_improve = 0
                break
        
        if not improved:
            no_improve += 1
        
        # Try swapping each element of B with a random alternative
        for _ in range(50):
            idx = random.randint(0, 2)
            old_val = current_B[idx]
            candidates = [x for x in cand_B if x != old_val and x not in (current_B[(idx+1)%3], current_B[(idx+2)%3])]
            if not candidates: continue
            new_val = random.choice(candidates)
            test_B = list(current_B)
            test_B[idx] = new_val
            card, valid = evaluate_pair(current_A, test_B)
            if valid and card > curr_card:
                current_B = test_B
                curr_card, curr_valid = card, True
                improved = True
                no_improve = 0
                if curr_card == 9:
                    return list(current_A), list(current_B)
                break
            elif valid and card == curr_card and random.random() < 0.1:
                current_B = test_B
                curr_card, curr_valid = card, True
                improved = True
                no_improve = 0
                break
        
        if not improved:
            no_improve += 1
        
        if no_improve >= 10:
            break
    
    return list(current_A), list(current_B)

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    # Fallback best
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_card = 5

    # Structural Change: Targeted Exhaustive Search on Small Ranges
    # Hypothesis: Solutions with cardinality 9 are likely to use small numbers
    # because small primes are abundant and coprime.
    
    # Stage 1: Very small range (fast)
    # A: odds 1-30, B: evens 2-30
    odds_1 = [i for i in range(1, 31) if i % 2 != 0]
    evens_1 = [i for i in range(2, 31) if i % 2 == 0]
    
    for A in combinations(odds_1, 3):
        if time.time() > deadline - 10: break
        for B in combinations(evens_1, 3):
            sums = [a + b for a in A for b in B]
            distinct_sums = set(sums)
            if len(distinct_sums) == 9:
                d_list = list(distinct_sums)
                if is_pairwise_coprime(d_list):
                    return [list(A), list(B)]
            # Track best even if not 9
            card, valid = evaluate_pair(A, B)
            if valid and card > best_card:
                best_A, best_B = list(A), list(B)
                best_card = card
                if best_card == 9:
                    return [best_A, best_B]

    # Stage 2: Medium range
    # A: odds 1-50, B: evens 2-50
    # Only run if time permits
    if time.time() < deadline - 15:
        odds_2 = [i for i in range(1, 51) if i % 2 != 0]
        evens_2 = [i for i in range(2, 51) if i % 2 == 0]
        
        # To speed up, we can skip combinations that are subsets of previous? 
        # Or just run it. ~5 million iterations.
        # Optimization: Break early if time is tight.
        iter_count = 0
        for A in combinations(odds_2, 3):
            if time.time() > deadline - 5: break
            for B in combinations(evens_2, 3):
                iter_count += 1
                if iter_count % 10000 == 0 and time.time() > deadline - 5:
                    break
                
                sums = [a + b for a in A for b in B]
                distinct_sums = set(sums)
                if len(distinct_sums) == 9:
                    d_list = list(distinct_sums)
                    if is_pairwise_coprime(d_list):
                        return [list(A), list(B)]
                
                card, valid = evaluate_pair(A, B)
                if valid and card > best_card:
                    best_A, best_B = list(A), list(B)
                    best_card = card
                    if best_card == 9:
                        return [best_A, best_B]

    # Stage 3: Fallback to random search if exhaustive didn't find 9
    if time.time() < deadline - 2:
        odds = [i for i in range(1, 129) if i % 2 != 0]
        evens = [i for i in range(1, 129) if i % 2 == 0]
        
        for _ in range(5000):
            if time.time() > deadline - 2: break
            A = random.sample(odds, 3)
            B = random.sample(evens, 3)
            card, valid = evaluate_pair(A, B)
            if valid and card > best_card:
                best_A, best_B = A, B
                best_card = card
                if card == 9:
                    return [A, B]
        
        # Final local search from best found
        if time.time() < deadline - 1:
            la, lb = local_search(best_A, best_B, deadline, best_card)
            card, valid = evaluate_pair(la, lb)
            if valid and card > best_card:
                best_A, best_B = la, lb
                best_card = card

    return [best_A, best_B]
