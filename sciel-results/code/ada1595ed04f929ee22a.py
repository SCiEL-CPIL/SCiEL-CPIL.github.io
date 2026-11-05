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

def _is_valid(values):
    """Check if all nonempty consecutive block products are distinct."""
    n = len(values)
    prods = set()
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= values[j]
            if prod in prods:
                return False
            prods.add(prod)
    return True

def _local_search_sa(values, deadline):
    """Simulated Annealing / Hill Climbing local search."""
    values = list(values)
    best_values = list(values)
    best_len = len(values)
    current_len = len(values)
    
    used = set(values)
    all_nums = list(range(2, 129))
    
    # Simple Hill Climbing with random restarts for local moves
    for _ in range(5000): # Max iterations, but bounded by time
        if time.time() > deadline:
            break
        
        improved = False
        
        # 1. Try to insert a new number
        unused = [x for x in all_nums if x not in used]
        if not unused:
            break
            
        # Shuffle to avoid bias
        random.shuffle(unused)
        
        found_insert = False
        for x in unused:
            # Try all positions
            for i in range(len(values) + 1):
                lo = (values[i-1] + 1) if i > 0 else 2
                hi = (values[i] - 1) if i < len(values) else 128
                if x < lo or x > hi:
                    continue
                
                test_vals = values[:i] + [x] + values[i:]
                if _is_valid(test_vals):
                    values = test_vals
                    used.add(x)
                    current_len += 1
                    if current_len > best_len:
                        best_len = current_len
                        best_values = list(values)
                    improved = True
                    found_insert = True
                    break
            if found_insert:
                break
        
        if improved:
            continue
            
        # 2. Try swaps
        n = len(values)
        if n < 2:
            continue
            
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(pairs)
        
        for i, j in pairs[:20]: # Limit swaps per iteration
            if values[i] == values[j]:
                continue
            test_vals = values[:]
            test_vals[i], test_vals[j] = test_vals[j], test_vals[i]
            
            # Check strictly increasing
            valid_inc = True
            for k in range(len(test_vals)-1):
                if test_vals[k+1] <= test_vals[k]:
                    valid_inc = False
                    break
            if not valid_inc:
                continue
                
            if _is_valid(test_vals):
                values = test_vals
                improved = True
                break
                
        if improved:
            continue

        # 3. If stuck, try a restart with greedy
        if random.random() < 0.05: # 5% chance to restart
            new_vals, _ = _greedy_pass([], set(), start_from=2)
            values = new_vals
            used = set(values)
            current_len = len(values)
            if current_len > best_len:
                best_len = current_len
                best_values = list(values)

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
        best_values = _local_search_sa(best_values, deadline)
    
    return best_values
# EVOLVE-BLOCK-END
