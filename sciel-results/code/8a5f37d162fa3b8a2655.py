"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _precompute_bad_triples(max_n):
    """
    Precompute all bad triples {a,b,c} with a<b<c<=max_n
    where lcm(a,b)==lcm(a,c)==lcm(b,c).
    Also returns mapping from element to list of triple indices.
    """
    n = max_n
    # Precompute LCM table for speed
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            if i != j:
                lcm_table[j][i] = lcm_table[i][j]
    
    bad_triples = []
    element_to_triples = [[] for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    idx = len(bad_triples)
                    bad_triples.append((a, b, c))
                    element_to_triples[a].append(idx)
                    element_to_triples[b].append(idx)
                    element_to_triples[c].append(idx)
                    
    return bad_triples, element_to_triples

def _greedy_initial_solution(bad_triples, element_to_triples, max_n, seed=0):
    """
    Generate an initial valid solution using greedy addition.
    Returns a boolean list and active count (0).
    """
    random.seed(seed)
    selected = [False] * (max_n + 1)
    
    order = list(range(1, max_n + 1))
    random.shuffle(order)
    
    for x in order:
        creates_triple = False
        for t_idx in element_to_triples[x]:
            a, b, c = bad_triples[t_idx]
            others = [y for y in (a, b, c) if y != x]
            if selected[others[0]] and selected[others[1]]:
                creates_triple = True
                break
        
        if not creates_triple:
            selected[x] = True
            
    return selected, 0

def _simulated_annealing(bad_triples, element_to_triples, max_n, initial_selected, initial_active, deadline, seed=0):
    """
    Simulated annealing.
    """
    random.seed(seed)
    selected = list(initial_selected)
    active_count = initial_active
    
    best_selected = list(selected)
    best_size = sum(selected)
    best_active = active_count
    
    t0 = 10.0
    t_end = 0.01
    cooling_rate = 0.999
    
    temp = t0
    steps = 0
    
    while temp > t_end:
        if time.time() > deadline:
            break
            
        x = random.randint(1, max_n)
        
        if selected[x]:
            # Try to remove x
            removed_count = 0
            for t_idx in element_to_triples[x]:
                a, b, c = bad_triples[t_idx]
                others = [y for y in (a, b, c) if y != x]
                if selected[others[0]] and selected[others[1]]:
                    removed_count += 1
            
            new_active = active_count - removed_count
            new_size = sum(selected) - 1
            
            penalty = 5.0
            old_score = sum(selected) - penalty * active_count
            new_score = new_size - penalty * new_active
            delta = new_score - old_score
            
            if delta >= 0 or random.random() < math.exp(delta / temp):
                selected[x] = False
                active_count = new_active
                
                if active_count == 0 and new_size > best_size:
                    best_size = new_size
                    best_active = 0
                    best_selected = list(selected)
        else:
            # Try to add x
            added_count = 0
            for t_idx in element_to_triples[x]:
                a, b, c = bad_triples[t_idx]
                others = [y for y in (a, b, c) if y != x]
                if selected[others[0]] and selected[others[1]]:
                    added_count += 1
            
            new_active = active_count + added_count
            new_size = sum(selected) + 1
            
            penalty = 5.0
            old_score = sum(selected) - penalty * active_count
            new_score = new_size - penalty * new_active
            delta = new_score - old_score
            
            if delta >= 0 or random.random() < math.exp(delta / temp):
                selected[x] = True
                active_count = new_active
                
                if active_count == 0 and new_size > best_size:
                    best_size = new_size
                    best_active = 0
                    best_selected = list(selected)
        
        temp *= cooling_rate
        steps += 1
        
    return best_selected, best_active

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 90s limit, 5s margin
    
    max_n = 128
    bad_triples, element_to_triples = _precompute_bad_triples(max_n)
    
    if not bad_triples:
        return list(range(1, max_n + 1))
    
    best_solution = None
    best_size = -1
    
    # Run multiple SA instances
    num_seeds = 20
    for seed in range(num_seeds):
        if time.time() > deadline - 5:
            break
        
        # Generate initial solution (Greedy Add)
        initial_selected, initial_active = _greedy_initial_solution(bad_triples, element_to_triples, max_n, seed=seed)
        
        # initial is valid
        if initial_active == 0:
            init_size = sum(initial_selected)
            if init_size > best_size:
                best_size = init_size
                best_solution = initial_selected
        
        sa_selected, sa_active = _simulated_annealing(bad_triples, element_to_triples, max_n, initial_selected, initial_active, deadline - 2, seed=seed)
        
        if sa_active == 0:
            sa_size = sum(sa_selected)
            if sa_size > best_size:
                best_size = sa_size
                best_solution = sa_selected
    
    if best_solution is None:
        # Fallback: Return a single element (always valid)
        return [1]
    
    # Convert boolean list to list of elements
    result = [i for i, val in enumerate(best_solution) if val and i > 0]
    return result

# EVOLVE-BLOCK-END
