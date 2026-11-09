import time
import itertools
import random

_LCM = 232792560

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    universe = list(range(1, 23))
    values = {x: _LCM // x for x in universe}
    
    best_size = 0
    best_set = []
    
    # Define search orders
    # 1. Descending Value (Large elements first)
    order_desc = sorted(universe, key=lambda x: values[x], reverse=True)
    # 2. Ascending Value (Small elements first)
    order_asc = sorted(universe, key=lambda x: values[x])
    
    search_orders = [order_desc, order_asc]
    
    # Add random orders for diversity
    for _ in range(20):
        rand_order = universe.copy()
        random.shuffle(rand_order)
        search_orders.append(rand_order)
        
    for order in search_orders:
        if time.time() > deadline - 5:
            break
            
        current_sums = {0}
        current_elements = []
        
        def _backtrack(idx, current_sums, current_elements):
            nonlocal best_size, best_set
            
            # Time check
            if time.time() > deadline - 2:
                return
            
            # Pruning: Can we beat the current best?
            if len(current_elements) + (len(order) - idx) <= best_size:
                return
            
            if idx == len(order):
                if len(current_elements) > best_size:
                    best_size = len(current_elements)
                    best_set = current_elements.copy()
                return
            
            x = order[idx]
            v = values[x]
            
            # Try to include x
            # Check for collision: any (s + v) in current_sums?
            new_sums = set()
            conflict = False
            for s in current_sums:
                ns = s + v
                if ns in current_sums:
                    conflict = True
                    break
                new_sums.add(ns)
            
            if not conflict:
                # Include x
                # The new set of sums is current_sums U new_sums
                # Since we broke on conflict, if no conflict, new_sums is disjoint from current_sums
                next_sums = current_sums | new_sums
                current_elements.append(x)
                _backtrack(idx + 1, next_sums, current_elements)
                current_elements.pop()
            
            # Try to exclude x
            _backtrack(idx + 1, current_sums, current_elements)

        _backtrack(0, {0}, [])

    # Ensure non-empty
    if not best_set:
        # Fallback: Any single element is valid
        best_set = [1]
        
    return best_set
