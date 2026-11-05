"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def _all_products(values):
    """Return set of all nonempty consecutive block products."""
    n = len(values)
    prods = set()
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= values[j]
            prods.add(prod)
    return prods

def _can_add(values, products, x):
    """Check if x can be appended to values (strictly increasing, no product collision)."""
    if values and x <= values[-1]:
        return False
    tails = [x]
    prod = x
    for a in reversed(values):
        prod *= a
        tails.append(prod)
    if products.intersection(tails):
        return False
    return True

def _greedy_pass(values, products, start_from=2):
    """Continue greedy from current state."""
    values = list(values)
    products = set(products)
    for x in range(start_from, 129):
        if _can_add(values, products, x):
            tails = [x]
            prod = x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            values.append(x)
            products.update(tails)
    return values, products

def _local_search(values, products, deadline):
    """Apply local search: remove-and-reinsert, insert-new operations."""
    values = list(values)
    products = set(products)
    best = len(values)
    best_values = list(values)
    n = len(values)
    
    for _ in range(3):  # Multiple passes
        improved = False
        for i in range(len(values)):
            if time.time() > deadline:
                break
            # Try removing values[i] and re-greedy
            removed = values[i]
            vals_without = values[:i] + values[i+1:]
            prods_without = _all_products(vals_without)
            
            # Re-greedy from the gap
            new_vals, new_prods = _greedy_pass(vals_without, prods_without, start_from=2)
            if len(new_vals) > best:
                best = len(new_vals)
                best_values = list(new_vals)
                values = list(new_vals)
                products = set(new_prods)
                improved = True
        
        # Try inserting new values at various positions
        used_set = set(values)
        for i in range(len(values) + 1):
            if time.time() > deadline:
                break
            lo = (values[i-1] + 1) if i > 0 else 2
            hi = (values[i] - 1) if i < len(values) else 128
            for v in range(lo, hi + 1):
                if v in used_set:
                    continue
                # Try inserting v at position i
                test_vals = values[:i] + [v] + values[i:]
                # Check strictly increasing
                if any(test_vals[j+1] <= test_vals[j] for j in range(len(test_vals)-1)):
                    continue
                test_prods = _all_products(test_vals)
                if len(test_prods) == len(test_vals) * (len(test_vals) + 1) // 2:
                    # Valid, continue greedy
                    new_vals, new_prods = _greedy_pass(test_vals, test_prods, start_from=128)
                    if len(new_vals) > best:
                        best = len(new_vals)
                        best_values = list(new_vals)
                        values = list(new_vals)
                        products = set(new_prods)
                        used_set = set(values)
                        improved = True
        if not improved:
            break
    
    return best_values

def construct():
    deadline = time.time() + 55.0  # Safety margin before 60s kill
    
    # Strategy 1: Standard greedy
    values1, products1 = [], set()
    for x in range(2, 129):
        if _can_add(values1, products1, x):
            tails = [x]
            prod = x
            for a in reversed(values1):
                prod *= a
                tails.append(prod)
            values1.append(x)
            products1.update(tails)
    best_values = list(values1)
    best_len = len(values1)
    
    # Strategy 2: Greedy with different starting points (skip some small values)
    for skip_start in range(2, 10):
        if time.time() > deadline:
            break
        values_try, products_try = [], set()
        for x in range(2, 129):
            if x < skip_start and x != 2:
                continue
            if _can_add(values_try, products_try, x):
                tails = [x]
                prod = x
                for a in reversed(values_try):
                    prod *= a
                    tails.append(prod)
                values_try.append(x)
                products_try.update(tails)
        if len(values_try) > best_len:
            best_len = len(values_try)
            best_values = list(values_try)
    
    # Strategy 3: Randomized multi-start greedy
    random.seed(42)
    for trial in range(20):
        if time.time() > deadline:
            break
        values_try, products_try = [], set()
        # Random subset of initial values
        candidates = list(range(2, 129))
        random.shuffle(candidates)
        # Try a random ordering but maintain increasing constraint
        for x in sorted(candidates):
            if _can_add(values_try, products_try, x):
                tails = [x]
                prod = x
                for a in reversed(values_try):
                    prod *= a
                    tails.append(prod)
                values_try.append(x)
                products_try.update(tails)
        if len(values_try) > best_len:
            best_len = len(values_try)
            best_values = list(values_try)
    
    # Apply local search to the best found
    if time.time() < deadline:
        products_best = _all_products(best_values)
        best_values = _local_search(best_values, products_best, deadline)
    
    return best_values
# EVOLVE-BLOCK-END
