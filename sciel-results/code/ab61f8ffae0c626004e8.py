import time
import math
import random
import sys
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s limit
    
    # Precompute weights: L = lcm(1, ..., 22)
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def lcm(a, b):
        return a * b // gcd(a, b)

    L = 1
    for i in range(1, 23):
        L = lcm(L, i)
        
    weights = [0] * 23
    for i in range(1, 23):
        weights[i] = L // i

    all_elements = list(range(1, 23))
    
    # Helper: check if a set of indices has all distinct subset sums of weights
    def get_subset_sums(indices):
        if not indices:
            return {0}
        sums = {0}
        for idx in indices:
            w = weights[idx]
            new_sums = set()
            for s in sums:
                ns = s + w
                if ns in sums:
                    return None
                new_sums.add(ns)
            sums.update(new_sums)
        return sums

    # Faster incremental check
    def check_addable(current_sums, element_idx):
        w = weights[element_idx]
        new_sums = set()
        for s in current_sums:
            ns = s + w
            if ns in current_sums:
                return None
            new_sums.add(ns)
        return new_sums

    best_set = []
    best_sums = None
    best_size = 0
    
    # Strategy 1: Multiple deterministic greedy orders
    orders = [
        all_elements[::-1],
        all_elements,
        [1, 22, 2, 21, 3, 20, 4, 19, 5, 18, 6, 17, 7, 16, 8, 15, 9, 14, 10, 13, 11, 12],
        [2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21],
        [22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        current = []
        current_sums = {0}
        
        for e in order:
            if time.time() > deadline:
                break
            new_sums = check_addable(current_sums, e)
            if new_sums is not None:
                current.append(e)
                current_sums.update(new_sums)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
            best_sums = set(current_sums)
            
    # Strategy 2: Backtracking with pruning (the key improvement)
    # We do a DFS over the decision space: for each element, either include or exclude.
    # Pruning: 
    #   1. If including causes a collision, skip include branch.
    #   2. If len(current) + (remaining elements count) <= best_size, prune.
    #   3. Time limit check.
    
    def backtrack(current_set, current_sums, remaining_elements, depth):
        nonlocal best_set, best_sums, best_size
        
        if time.time() > deadline:
            return
        
        # Pruning: even if we add all remaining, can't beat best
        if len(current_set) + len(remaining_elements) <= best_size:
            return
        
        # If we've processed all elements, update best if better
        if not remaining_elements:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
                best_sums = set(current_sums)
            return
        
        # Try including the first remaining element
        next_elem = remaining_elements[0]
        rest = remaining_elements[1:]
        
        # Include branch
        new_sums = check_addable(current_sums, next_elem)
        if new_sums is not None:
            current_set.append(next_elem)
            backtrack(current_set, new_sums, rest, depth + 1)
            current_set.pop()
        
        # Exclude branch
        backtrack(current_set, current_sums, rest, depth + 1)
    
    # Sort elements in a good order for backtracking.
    # We want to try elements that are "easy" to include first (small weights = large indices)
    # so we build up a large set quickly, which helps with the pruning bound.
    # But we also want to explore deeply.
    # Let's try a few different orderings for the backtracking.
    
    # Order 1: Largest indices first (smallest weights)
    order1 = all_elements[::-1]
    backtrack([], {0}, order1, 0)
    
    # Order 2: Smallest indices first (largest weights)
    if time.time() < deadline:
        order2 = all_elements[:]
        backtrack([], {0}, order2, 0)
    
    # Order 3: Alternating
    if time.time() < deadline:
        order3 = []
        for i in range(11):
            order3.append(1 + i)
            order3.append(22 - i)
        backtrack([], {0}, order3, 0)
    
    # Strategy 3: Local Search / Hill Climbing on the best found
    if best_set and time.time() < deadline:
        best_set.sort()
        
        max_iterations = 50
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            current_set = best_set[:]
            current_sums = best_sums
            current_size = len(current_set)
            remaining = [e for e in all_elements if e not in current_set]
            
            improved = False
            
            for i in range(len(current_set)):
                if time.time() > deadline:
                    break
                
                removed = current_set[i]
                test_set = current_set[:i] + current_set[i+1:]
                test_sums = get_subset_sums(test_set)
                if test_sums is None:
                    continue
                
                added_count = 0
                new_test_sums = test_sums
                new_test_set = test_set[:]
                
                for e in remaining:
                    if time.time() > deadline:
                        break
                    new_sums = check_addable(new_test_sums, e)
                    if new_sums is not None:
                        new_test_set.append(e)
                        new_test_sums.update(new_sums)
                        added_count += 1
                
                if added_count > 1:
                    new_size = len(new_test_set)
                    if new_size > current_size:
                        current_set = new_test_set
                        current_sums = new_test_sums
                        current_size = new_size
                        remaining = [e for e in all_elements if e not in current_set]
                        improved = True
                        break
            
            if improved:
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_set
                    best_sums = current_sums
                    continue
            else:
                if _ > 0:
                    break
    
    # Strategy 4: Randomized Greedy with Restart
    if time.time() < deadline:
        for _ in range(30):
            if time.time() > deadline:
                break
            
            shuffled = all_elements[:]
            random.shuffle(shuffled)
            current = []
            current_sums = {0}
            
            for e in shuffled:
                if time.time() > deadline:
                    break
                new_sums = check_addable(current_sums, e)
                if new_sums is not None:
                    current.append(e)
                    current_sums.update(new_sums)
            
            if len(current) > best_size:
                best_size = len(current)
                best_set = current[:]
                best_sums = set(current_sums)
                
                if time.time() < deadline:
                    remaining = [e for e in all_elements if e not in best_set]
                    improved = True
                    while improved:
                        improved = False
                        for e in remaining[:]:
                            if time.time() > deadline:
                                break
                            new_sums = check_addable(best_sums, e)
                            if new_sums is not None:
                                best_set.append(e)
                                best_sums.update(new_sums)
                                remaining.remove(e)
                                improved = True

    # Fallback
    if not best_set:
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    return best_set
