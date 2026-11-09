import time
import random
from math import lcm

def construct():
    denominators = list(range(2, 49))
    n = len(denominators)
    
    L = 1
    for d in denominators:
        L = lcm(L, d)
    
    weights = [L // d for d in denominators]
    
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Greedy initialization
    sorted_indices = sorted(range(n), key=lambda i: -weights[i])
    sorted_weights = [weights[i] for i in sorted_indices]
    
    greedy_sum = 0
    greedy_set = set()
    for i in range(n):
        w = sorted_weights[i]
        if greedy_sum + w < L:
            greedy_sum += w
            greedy_set.add(sorted_indices[i])
    
    best_sum = greedy_sum
    best_set = set(greedy_set)
    
    # Local search: 1-add, 1-swap, 1-for-2-swap
    def local_search(current_set, current_sum):
        nonlocal best_sum, best_set
        improved = True
        while improved:
            improved = False
            
            # 1-add: try adding each element not in set
            for i in range(n):
                if i not in current_set:
                    new_sum = current_sum + weights[i]
                    if new_sum < L and new_sum > current_sum:
                        current_set.add(i)
                        current_sum = new_sum
                        improved = True
                        if current_sum > best_sum:
                            best_sum = current_sum
                            best_set = set(current_set)
                        break
            
            # 1-swap: try swapping each element in set with each not in set
            if not improved:
                in_set = list(current_set)
                for i in in_set:
                    for j in range(n):
                        if j not in current_set:
                            new_sum = current_sum - weights[i] + weights[j]
                            if new_sum < L and new_sum > current_sum:
                                current_set.remove(i)
                                current_set.add(j)
                                current_sum = new_sum
                                improved = True
                                if current_sum > best_sum:
                                    best_sum = current_sum
                                    best_set = set(current_set)
                                break
                    if improved:
                        break
            
            # 1-for-2-swap: remove one, add two
            if not improved:
                in_set = list(current_set)
                for i in in_set:
                    for j in range(n):
                        if j not in current_set:
                            for k in range(n):
                                if k not in current_set and k != j:
                                    new_sum = current_sum - weights[i] + weights[j] + weights[k]
                                    if new_sum < L and new_sum > current_sum:
                                        current_set.remove(i)
                                        current_set.add(j)
                                        current_set.add(k)
                                        current_sum = new_sum
                                        improved = True
                                        if current_sum > best_sum:
                                            best_sum = current_sum
                                            best_set = set(current_set)
                                        break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        return current_set, current_sum
    
    # Run local search from greedy
    best_set, best_sum = local_search(set(best_set), best_sum)
    
    # Random restarts with local search
    rng = random.Random(42)
    while time.time() < deadline:
        # Create a random subset
        rand_set = set()
        rand_sum = 0
        for i in range(n):
            if rng.random() < 0.5:
                if rand_sum + weights[i] < L:
                    rand_set.add(i)
                    rand_sum += weights[i]
        
        # Local search from random start
        new_set, new_sum = local_search(set(rand_set), rand_sum)
        
        if new_sum > best_sum:
            best_sum = new_sum
            best_set = new_set
    
    # Convert to list of indices
    result = list(best_set)
    if not result:
        result = [0]
    
    return result
