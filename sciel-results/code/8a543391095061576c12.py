import time
import random

def is_valid(vals):
    """
    Check if the sorted list of integers has distinct products for all non-empty consecutive blocks.
    """
    if not vals:
        return True
    
    products = set()
    n = len(vals)
    
    # Generate all consecutive block products
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= vals[j]
            if prod in products:
                return False
            products.add(prod)
            
    return True

def construct():
    # 1. Generate the seed greedy solution
    values, products = [], set()
    for x in range(2, 129):
        tails, prod = [x], x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    
    best_set = set(values)
    best_len = len(values)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0  # 50s margin
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add any missing number
        for x in range(2, 129):
            if x not in best_set:
                new_set = best_set | {x}
                if is_valid(sorted(new_set)):
                    best_set = new_set
                    best_len += 1
                    improved = True
                    break
        
        if improved:
            continue
            
        # Phase 2: Try swapping one element to enable adding another
        # This is more expensive, so we do it less frequently or carefully
        if time.time() > deadline - 5.0:
            break
            
        found_swap = False
        # Iterate over elements to remove
        for x in list(best_set):
            if found_swap:
                break
            # Iterate over elements to add
            for y in range(2, 129):
                if y in best_set or y == x:
                    continue
                
                new_set = best_set - {x} | {y}
                if not is_valid(sorted(new_set)):
                    continue
                
                # If swap is valid, try to add a third element z
                for z in range(2, 129):
                    if z in new_set:
                        continue
                    final_set = new_set | {z}
                    if is_valid(sorted(final_set)):
                        best_set = final_set
                        best_len += 1
                        improved = True
                        found_swap = True
                        break
        
    return sorted(best_set)
