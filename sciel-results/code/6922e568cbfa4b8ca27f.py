import math
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin for 90s limit
    
    N = 256
    n_nodes = 255
    numbers = list(range(2, 257))
    weights = np.array([1.0 / x for x in numbers])
    
    # Build conflict sets
    conflict_sets = [set() for _ in range(n_nodes)]
    for i in range(n_nodes):
        a = i + 2
        for j in range(i + 1, n_nodes):
            b = j + 2
            g = math.gcd(a, b)
            if a * b <= 256 * g:
                conflict_sets[i].add(j)
                conflict_sets[j].add(i)
                
    best_score = 0.0
    best_indices = set()
    
    def update_best(current_set, current_score):
        nonlocal best_score, best_indices
        if current_score > best_score:
            best_score = current_score
            best_indices = set(current_set)
            
    # Initial Greedy Solution
    # Sort by weight descending, then by degree ascending
    order = sorted(range(n_nodes), key=lambda i: (-weights[i], len(conflict_sets[i])))
    current_set = set()
    current_score = 0.0
    for i in order:
        if not conflict_sets[i].intersection(current_set):
            current_set.add(i)
            current_score += weights[i]
            
    update_best(current_set, current_score)
    
    # Simulated Annealing
    if time.time() < deadline - 5.0:
        T = 0.1
        T_end = 0.0001
        cooling_rate = 0.9995
        
        max_time = deadline - 1.0
        
        while time.time() < max_time and T > T_end:
            move_type = random.random()
            delta_score = 0.0
            new_set = None
            
            if move_type < 0.3:
                # Add
                i = random.randint(0, n_nodes - 1)
                if i not in current_set:
                    if not conflict_sets[i].intersection(current_set):
                        new_set = current_set | {i}
                        delta_score = weights[i]
                        
            elif move_type < 0.6:
                # Remove
                if current_set:
                    i = random.choice(list(current_set))
                    new_set = current_set - {i}
                    delta_score = -weights[i]
                    
            else:
                # Swap
                if current_set:
                    i = random.choice(list(current_set))
                    j = random.randint(0, n_nodes - 1)
                    if j != i and j not in current_set:
                        # Check compatibility of j with current_set \ {i}
                        compatible = True
                        for k in current_set:
                            if k != i and k in conflict_sets[j]:
                                compatible = False
                                break
                        if compatible:
                            new_set = (current_set - {i}) | {j}
                            delta_score = weights[j] - weights[i]
                            
            if new_set is None:
                continue
                
            if delta_score >= 0:
                current_set = new_set
                current_score += delta_score
                update_best(current_set, current_score)
            else:
                p = math.exp(delta_score / T)
                if random.random() < p:
                    current_set = new_set
                    current_score += delta_score
                    
            T *= cooling_rate
            
    if not best_indices:
        return [2]
        
    return [i + 2 for i in best_indices]
