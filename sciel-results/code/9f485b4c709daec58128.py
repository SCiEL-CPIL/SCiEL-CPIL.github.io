import time
from fractions import Fraction
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # LCM of 2..48 for integer arithmetic
    # LCM(2..48) = 2^4 * 3^3 * 5^2 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47
    # = 16 * 27 * 25 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47
    # Let's compute it properly
    from math import gcd
    def lcm(a, b):
        return a // gcd(a, b) * b
    
    L = 1
    for d in range(2, 49):
        L = lcm(L, d)
    
    # Integer reciprocal values: 1/d = L/d in scaled units
    recips = [L // d for d in denominators]
    one_scaled = L  # represents 1.0
    
    # Greedy initialization
    selected = set()
    total = 0
    for i in range(n):
        if total + recips[i] < one_scaled:
            selected.add(i)
            total += recips[i]
    
    best_indices = list(selected)
    best_sum = total
    best_deficit = one_scaled - best_sum
    
    # Local search with multiple move types
    def try_local_search(init_set, init_sum):
        nonlocal best_indices, best_sum, best_deficit
        nonlocal start_time, deadline
        
        current_set = set(init_set)
        current_sum = init_sum
        current_deficit = one_scaled - current_sum
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_deficit = one_scaled - current_sum
            
            # Try all 1-add moves
            for j in range(n):
                if j not in current_set:
                    new_sum = current_sum + recips[j]
                    if new_sum < one_scaled:
                        new_deficit = one_scaled - new_sum
                        if new_deficit < current_deficit:
                            current_set.add(j)
                            current_sum = new_sum
                            current_deficit = new_deficit
                            improved = True
            
            # Try all 1-remove moves
            for i in list(current_set):
                new_sum = current_sum - recips[i]
                new_deficit = one_scaled - new_sum
                if new_deficit < current_deficit:
                    current_set.remove(i)
                    current_sum = new_sum
                    current_deficit = new_deficit
                    improved = True
            
            # Try all 1-swap moves
            for i in list(current_set):
                for j in range(n):
                    if j not in current_set:
                        new_sum = current_sum - recips[i] + recips[j]
                        if new_sum < one_scaled:
                            new_deficit = one_scaled - new_sum
                            if new_deficit < current_deficit:
                                current_set.remove(i)
                                current_set.add(j)
                                current_sum = new_sum
                                current_deficit = new_deficit
                                improved = True
            
            # Try 2-for-1 swap moves (remove 1, add 2)
            if time.time() < deadline:
                for i in list(current_set):
                    base_sum = current_sum - recips[i]
                    for j in range(n):
                        if j not in current_set and j != i:
                            for k in range(n):
                                if k not in current_set and k != i and k != j:
                                    new_sum = base_sum + recips[j] + recips[k]
                                    if new_sum < one_scaled:
                                        new_deficit = one_scaled - new_sum
                                        if new_deficit < current_deficit:
                                            current_set.remove(i)
                                            current_set.add(j)
                                            current_set.add(k)
                                            current_sum = new_sum
                                            current_deficit = new_deficit
                                            improved = True
                                if time.time() > deadline:
                                    break
                        if time.time() > deadline:
                            break
                    if time.time() > deadline:
                        break
            
            # Try 1-for-2 swap moves (remove 2, add 1)
            if time.time() < deadline and len(current_set) >= 2:
                elems = list(current_set)
                for idx1 in range(len(elems)):
                    for idx2 in range(idx1 + 1, len(elems)):
                        i, j = elems[idx1], elems[idx2]
                        base_sum = current_sum - recips[i] - recips[j]
                        for k in range(n):
                            if k not in current_set:
                                new_sum = base_sum + recips[k]
                                if new_sum < one_scaled:
                                    new_deficit = one_scaled - new_sum
                                    if new_deficit < current_deficit:
                                        current_set.discard(i)
                                        current_set.discard(j)
                                        current_set.add(k)
                                        current_sum = new_sum
                                        current_deficit = new_deficit
                                        improved = True
                            if time.time() > deadline:
                                break
                        if time.time() > deadline:
                            break
                    if time.time() > deadline:
                        break
            
            if current_deficit < best_deficit:
                best_deficit = current_deficit
                best_sum = current_sum
                best_indices = list(current_set)
        
        return current_set, current_sum
    
    # Run local search from greedy start
    try_local_search(selected, total)
    
    # Random restarts if time permits
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        # Generate a random initial set
        rand_set = set()
        rand_sum = 0
        # Randomly add elements while sum < 1
        shuffled = list(range(n))
        random.shuffle(shuffled)
        for i in shuffled:
            if rand_sum + recips[i] < one_scaled:
                if random.random() < 0.5:  # Sometimes skip to get diverse starts
                    rand_set.add(i)
                    rand_sum += recips[i]
        
        if not rand_set:
            continue
        
        # Run local search from this start
        final_set, final_sum = try_local_search(rand_set, rand_sum)
        
        if time.time() > deadline:
            break
    
    return best_indices
