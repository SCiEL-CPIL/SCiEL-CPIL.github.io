import time
import random

def _get_suffix_products(seq, start_idx):
    """Get all products of consecutive blocks ending at or after start_idx,
    that include at least one element from start_idx onward.
    Actually, we need all products of all consecutive blocks in the full sequence.
    But for insertion check, we only need suffix products ending at the new element."""
    pass

def _all_block_products(seq):
    """Compute all consecutive block products for a sequence."""
    n = len(seq)
    products = set()
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            products.add(prod)
    return products

def _can_add(seq, x):
    """Check if x can be appended to seq without creating duplicate block products.
    We only need to check new products (those ending at x) against existing products."""
    # Compute all existing products
    n = len(seq)
    existing = set()
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            existing.add(prod)
    
    # New products are suffix products: x, seq[-1]*x, seq[-2]*seq[-1]*x, etc.
    new_products = set()
    prod = x
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            prod = x
        else:
            prod *= seq[i]
        new_products.add(prod)
    
    return not (new_products & existing)

def _greedy_construct(deadline=None):
    """Standard greedy: try each number 2..128 in order, add if valid."""
    values = []
    products = set()
    for x in range(2, 129):
        if deadline and time.time() > deadline:
            break
        # Compute new suffix products
        new_prods = []
        prod = x
        new_prods.append(prod)
        for a in reversed(values):
            prod *= a
            new_prods.append(prod)
        if not set(new_prods) & products:
            values.append(x)
            products.update(new_prods)
    return values

def _try_rebuild_from(seq, start_idx, deadline=None):
    """Try to rebuild the sequence from index start_idx onward greedily.
    Keep seq[0:start_idx] fixed, then greedily add remaining numbers."""
    fixed = seq[:start_idx]
    # Compute all products of the fixed prefix
    n = len(fixed)
    prefix_products = set()
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= fixed[j]
            prefix_products.add(prod)
    
    # Also need suffix products that will be extended
    # When we add new elements, the new products include:
    # - products entirely in the new part
    # - products that start in the fixed part and end in the new part
    
    values = list(fixed)
    products = set(prefix_products)
    
    for x in range(fixed[-1] + 1, 129) if fixed else range(2, 129):
        if deadline and time.time() > deadline:
            break
        # Check if x can be added
        # New products: all consecutive blocks ending at x
        # These are: x, values[-1]*x, values[-2]*values[-1]*x, ...
        new_prods = set()
        prod = x
        new_prods.add(prod)
        for a in reversed(values):
            prod *= a
            new_prods.add(prod)
        
        if not (new_prods & products):
            values.append(x)
            products.update(new_prods)
    
    return values

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50 second safety margin
    
    best = _greedy_construct(deadline)
    best_len = len(best)
    
    # Local search: try removing each element and rebuilding
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Randomly choose a position to rebuild from
        pos = random.randint(0, len(best) - 1)
        candidate = _try_rebuild_from(best, pos, deadline)
        
        if len(candidate) > best_len:
            best = candidate
            best_len = len(best)
    
    # Also try randomized greedy with random skips
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        values = []
        products = set()
        for x in range(2, 129):
            if time.time() > deadline:
                break
            # Randomly skip with some probability
            if random.random() < 0.1:
                continue
            new_prods = []
            prod = x
            new_prods.append(prod)
            for a in reversed(values):
                prod *= a
                new_prods.append(prod)
            if not set(new_prods) & products:
                values.append(x)
                products.update(new_prods)
        
        if len(values) > best_len:
            best = values
            best_len = len(best)
    
    # Final local search on best
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        pos = random.randint(0, len(best) - 1)
        candidate = _try_rebuild_from(best, pos, deadline)
        
        if len(candidate) > best_len:
            best = candidate
            best_len = len(best)
    
    return best
