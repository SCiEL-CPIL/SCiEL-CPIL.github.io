"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    best = []
    
    def compute_tails(values, x):
        """Compute all consecutive-block products ending at x for the sequence values+[x]."""
        tails = []
        prod = x
        tails.append(prod)
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        return tails
    
    def is_valid(values, products, x):
        """Check if x can be appended to values without product collision."""
        tails = compute_tails(values, x)
        for t in tails:
            if t in products:
                return False, tails
        return True, tails
    
    def greedy_from(values, products, start):
        """Continue greedily from current state. Returns (values, products)."""
        for x in range(start, 129):
            if time.time() > deadline:
                break
            valid, tails = is_valid(values, products, x)
            if valid:
                values.append(x)
                products.update(tails)
        return values, products
    
    # Greedy baseline
    values, products = [], set()
    values, products = greedy_from(values, products, 2)
    best = list(values)
    
    # Try limited backtracking: remove last k elements and try alternatives
    for backtrack_depth in range(1, 8):
        if time.time() > deadline:
            break
        if len(values) <= backtrack_depth:
            continue
        
        # Save state
        saved_values = list(values)
        saved_products = set(products)
        
        # Remove last backtrack_depth elements
        removed = values[-backtrack_depth:]
        for _ in range(backtrack_depth):
            val = values.pop()
            # Remove products that involved val
            # Recompute products from scratch for the remaining sequence
        # Recompute products for remaining values
        products.clear()
        for i in range(len(values)):
            prod = 1
            for j in range(i, len(values)):
                prod *= values[j]
                products.add(prod)
        
        # Try alternative: skip the first removed element, try next available
        last_val = values[-1] if values else 1
        start_search = last_val + 1
        
        # Try each possible replacement for the removed elements
        for skip_offset in range(1, min(len(removed) + 1, 6)):
            if time.time() > deadline:
                break
            
            # Try to build from current state, skipping some of the removed values
            test_values = list(values)
            test_products = set(products)
            
            # Find what to try: start from where we left off, but skip some numbers
            # The removed elements were: removed[0], removed[1], ...
            # We want to try alternatives starting from the position of removed[skip_offset-1]
            # Actually, let's just try: from the current last value, try the next numbers
            # but skip the ones that were in removed
            
            removed_set = set(removed)
            found_any = False
            for x in range(start_search, 129):
                if time.time() > deadline:
                    break
                if x in removed_set and x == removed[0]:
                    # Skip the first removed element, try to find an alternative
                    continue
                valid, tails = is_valid(test_values, test_products, x)
                if valid:
                    test_values.append(x)
                    test_products.update(tails)
                    found_any = True
            
            # Continue greedily from here
            test_values, test_products = greedy_from(test_values, test_products, 
                                                      test_values[-1] + 1 if test_values else 2)
            
            if len(test_values) > len(best):
                best = list(test_values)
        
        # Restore state
        values = saved_values
        products = saved_products
    
    return best
# EVOLVE-BLOCK-END
