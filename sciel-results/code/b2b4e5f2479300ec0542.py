"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _compute_tails(values, x):
    """Compute all consecutive block products ending at x (i.e., x * suffix products)."""
    tails = [x]
    prod = x
    for a in reversed(values):
        prod *= a
        tails.append(prod)
    return tails

def _can_insert(values, products, x):
    """Check if x can be inserted into values (which is sorted) at its correct position."""
    # Find position where x would go
    lo, hi = 0, len(values)
    while lo < hi:
        mid = (lo + hi) // 2
        if values[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    pos = lo
    
    # x must not already be in values
    if pos < len(values) and values[pos] == x:
        return False
    
    # Compute the new tail products: x, x*values[pos-1], x*values[pos-1]*values[pos-2], ...
    # Wait, we need all consecutive blocks in the NEW sequence.
    # The new sequence is values[:pos] + [x] + values[pos:]
    # New consecutive blocks involving x:
    #   [x]
    #   [values[pos-1], x]
    #   [values[pos-2], values[pos-1], x]
    #   ...
    #   [values[0], ..., values[pos-1], x]
    #   [x, values[pos]]
    #   [x, values[pos], values[pos+1]]
    #   ...
    #   [x, values[pos], ..., values[-1]]
    
    # Products of blocks ending at x (suffixes of values[:pos] prepended to x):
    tails_left = [x]
    prod = x
    for i in range(pos-1, -1, -1):
        prod *= values[i]
        tails_left.append(prod)
    
    # Products of blocks starting at x (prefixes of values[pos:] appended to x):
    tails_right = []
    prod = x
    for i in range(pos, len(values)):
        prod *= values[i]
        tails_right.append(prod)
    
    # All new products are tails_left + tails_right[1:] (x is already in tails_left)
    # Actually, blocks that include x:
    # - Blocks entirely in values[:pos] or values[pos:] are unchanged
    # - New blocks are those that include x
    # These are:
    #   [x]
    #   [values[i], ..., values[pos-1], x] for i in 0..pos-1
    #   [x, values[pos], ..., values[j]] for j in pos..len(values)-1
    #   [values[i], ..., values[pos-1], x, values[pos], ..., values[j]] for i in 0..pos-1, j in pos..len(values)-1
    
    # The last category is products of blocks that span across x.
    # Let me compute all products of blocks that include x:
    
    new_products = set()
    # Block [x]
    new_products.add(x)
    # Blocks ending at x: [values[i]..values[pos-1], x]
    p = x
    for i in range(pos-1, -1, -1):
        p *= values[i]
        new_products.add(p)
    # Blocks starting at x: [x, values[pos]..values[j]]
    p = x
    for j in range(pos, len(values)):
        p *= values[j]
        new_products.add(p)
    # Blocks spanning x: [values[i]..values[pos-1], x, values[pos]..values[j]]
    # = (product of values[i..pos-1]) * x * (product of values[pos..j])
    # Let left_prods[i] = product of values[i..pos-1] for i in 0..pos-1, and left_prods[pos] = 1
    # Let right_prods[j] = product of values[pos..j] for j in pos..len-1, and right_prods[pos-1] = 1
    if pos > 0 and pos < len(values):
        left = [1] * pos
        p = 1
        for i in range(pos-1, -1, -1):
            p *= values[i]
            left[i] = p
        right = [1] * (len(values) - pos)
        p = 1
        for j in range(pos, len(values)):
            p *= values[j]
            right[j - pos] = p
        for i in range(pos):
            for j in range(len(values) - pos):
                new_products.add(left[i] * x * right[j])
    
    # Check for conflicts with existing products
    if products.intersection(new_products):
        return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s out of 90s limit
    
    # Phase 1: Greedy construction
    values, products = [], set()
    for x in range(2, 129):
        tails = _compute_tails(values, x)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    
    base_len = len(values)
    
    # Phase 2: Try to insert excluded numbers
    excluded = [x for x in range(2, 129) if x not in set(values)]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in excluded[:]:
            if time.time() >= deadline:
                break
            if _can_insert(values, products, x):
                # Insert x into values
                lo, hi = 0, len(values)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if values[mid] < x:
                        lo = mid + 1
                    else:
                        hi = mid
                pos = lo
                values.insert(pos, x)
                excluded.remove(x)
                
                # Recompute products
                products = set()
                for i in range(len(values)):
                    p = 1
                    for j in range(i, len(values)):
                        p *= values[j]
                        products.add(p)
                improved = True
    
    return values
# EVOLVE-BLOCK-END
