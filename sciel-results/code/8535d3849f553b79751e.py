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
    
    while time.time() < deadline:
        if curr_valid and curr_card == 9:
            return list(current_A), list(current_B)
        
        improved = False
        # Try swapping each element of A with a random alternative
        for _ in range(50):
            idx = random.randint(0, 2)
            old_val = current_A[idx]
            new_val = random.choice([x for x in all_nums if x != old_val])
            test_A = list(current_A)
            test_A[idx] = new_val
            if len(set(test_A)) != 3:
                continue
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
                # Occasional plateau move
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
            new_val = random.choice([x for x in all_nums if x != old_val])
            test_B = list(current_B)
            test_B[idx] = new_val
            if len(set(test_B)) != 3:
                continue
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
    
    odds = [i for i in range(1, 129) if i % 2 != 0]
    evens = [i for i in range(1, 129) if i % 2 == 0]
    
    # Initialize with seed
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_sums = set()
    for a in best_A:
        for b in best_B:
            best_sums.add(a + b)
    best_cardinality = len(best_sums)  # 5
    
    # Try a few structured initializations
    structured_starts = [
        ([1, 3, 5], [2, 4, 6]),
        ([1, 5, 7], [2, 4, 8]),
        ([1, 7, 11], [2, 6, 10]),
        ([3, 5, 9], [2, 4, 8]),
        ([1, 3, 7], [4, 8, 12]),
    ]
    
    for sA, sB in structured_starts:
        card, valid = evaluate_pair(sA, sB)
        if valid and card > best_cardinality:
            best_A, best_B = list(sA), list(sB)
            best_cardinality = card
            if card == 9:
                return [best_A, best_B]
    
    # Phase 1: Random search with occasional local refinement
    random_iterations = 0
    while time.time() < deadline:
        # Alternate or randomize between parity cases
        if random.random() < 0.5:
            A = random.sample(odds, 3)
            B = random.sample(evens, 3)
        else:
            A = random.sample(evens, 3)
            B = random.sample(odds, 3)
        
        card, valid = evaluate_pair(A, B)
        
        # If we found a perfect 9, return immediately
        if card == 9 and valid:
            return [A, B]
        
        # If we found a better cardinality than seed
        if valid and card > best_cardinality:
            best_A = list(A)
            best_B = list(B)
            best_cardinality = card
            if card == 9:
                return [best_A, best_B]
        
        random_iterations += 1
        
        # Every 200 random trials, do a local search from the current best
        if random_iterations % 200 == 0:
            la, lb = local_search(best_A, best_B, deadline, best_cardinality)
            card, valid = evaluate_pair(la, lb)
            if valid and card > best_cardinality:
                best_A, best_B = la, lb
                best_cardinality = card
                if card == 9:
                    return [best_A, best_B]
        
        # Also occasionally do local search from the random candidate if it's decent
        if card >= 6 and valid and random.random() < 0.05:
            la, lb = local_search(A, B, deadline, best_cardinality)
            card, valid = evaluate_pair(la, lb)
            if valid and card > best_cardinality:
                best_A, best_B = la, lb
                best_cardinality = card
                if card == 9:
                    return [best_A, best_B]
    
    return [best_A, best_B]
