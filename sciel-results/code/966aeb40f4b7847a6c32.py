import time
import random
from fractions import Fraction

def _greedy_construct(order, deadline):
    """Greedy construction given a specific order of elements to consider."""
    achievable_sums = {Fraction(0, 1)}
    current_set = []
    
    for x in order:
        if time.time() > deadline:
            break
            
        target = Fraction(1, x)
        
        if target in achievable_sums:
            continue
        
        current_set.append(x)
        
        # Update sums
        new_sums = set()
        for s in achievable_sums:
            ns = s + target
            if ns <= 1:
                new_sums.add(ns)
        achievable_sums.update(new_sums)
        
    return current_set

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    best_set = []
    
    # Try multiple orderings
    # 1. Original: 24 down to 1
    order1 = list(range(24, 0, -1))
    result = _greedy_construct(order1, deadline)
    if len(result) > len(best_set):
        best_set = result
    
    # 2. Try 1 up to 24
    if time.time() < deadline:
        order2 = list(range(1, 25))
        result = _greedy_construct(order2, deadline)
        if len(result) > len(best_set):
            best_set = result
    
    # 3. Try several random permutations
    random.seed(42)
    attempts = 0
    while time.time() < deadline and attempts < 200:
        order = list(range(1, 25))
        random.shuffle(order)
        result = _greedy_construct(order, deadline)
        if len(result) > len(best_set):
            best_set = result
        attempts += 1
    
    if not best_set:
        return list(range(13, 25))
        
    return best_set
