import time
import sys
import random
import itertools
import math

def construct(n):
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    deadline = time.time() + 140
    
    base_perms = list(itertools.permutations(range(1, n)))
    K = len(base_perms)
    
    # Pre-calculate overlap matrix
    # overlap[i][j] = max o such that base_perms[i][n-1-o:] == base_perms[j][:o]
    overlap_matrix = [[0] * K for _ in range(K)]
    
    for i in range(K):
        u = base_perms[i]
        for j in range(K):
            v = base_perms[j]
            max_o = n - 1
            o = 0
            # Check from max possible down
            for k in range(max_o, -1, -1):
                if k == 0:
                    o = 0
                    break
                if u[n-1-k:] == v[:k]:
                    o = k
                    break
            overlap_matrix[i][j] = o
            
    # Initial path: sorted
    current_path = list(range(K))
    
    def calc_cost(path):
        cost = 0
        for i in range(len(path) - 1):
            cost += overlap_matrix[path[i]][path[i+1]]
        return cost
        
    current_cost = calc_cost(current_path)
    best_path = current_path[:]
    best_cost = current_cost
    
    # Simulated Annealing
    T = 100.0
    min_T = 1.0
    cooling = 0.999
    
    while time.time() < deadline - 5:
        # Random swap
        i = random.randint(0, K - 1)
        j = random.randint(0, K - 1)
        
        if i == j:
            continue
            
        # Perform swap
        current_path[i], current_path[j] = current_path[j], current_path[i]
        
        # Recalculate cost
        new_total_cost = calc_cost(current_path)
        
        delta = new_total_cost - current_cost
        
        if delta > 0: # We want to maximize cost (overlap)
            current_cost = new_total_cost
            if current_cost > best_cost:
                best_cost = current_cost
                best_path = current_path[:]
        else:
            # Accept with prob exp(delta / T)
            if random.random() < math.exp(delta / T):
                current_cost = new_total_cost
            else:
                # Revert
                current_path[i], current_path[j] = current_path[j], current_path[i]
        
        T *= cooling
        if T < min_T:
            T = min_T

    def build_word_from_path(path):
        if not path:
            return [1]
        word = list(base_perms[path[0]]) + [n] + list(base_perms[path[0]])
        for i in range(1, len(path)):
            p_idx = path[i]
            prev_p_idx = path[i-1]
            
            p = base_perms[p_idx]
            
            overlap = overlap_matrix[prev_p_idx][p_idx]
            
            block = list(p) + [n] + list(p)
            word.extend(block[overlap:])
            
        return word

    best_word = build_word_from_path(best_path)
    return "".join(map(str, best_word))
