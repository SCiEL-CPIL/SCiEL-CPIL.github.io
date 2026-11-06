"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def _greedy_construct():
    values, products = [], set()
    for x in range(2, 129):
        tails, prod = [x], x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    return values

def _try_insert(values, products, x):
    """Try to insert x into the sequence, return new (values, products) if valid."""
    new_values = values + [x]
    new_products = set(products)
    # Compute all new block products involving x
    prod = x
    for a in reversed(values):
        new_products.add(prod)
        prod *= a
    if new_products != products:  # check for duplicates
        return None
    return new_values, new_products

def _is_valid_sequence(seq):
    """Check if a sequence has all distinct block products."""
    products = set()
    n = len(seq)
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            if prod in products:
                return False
            products.add(prod)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # Get initial greedy solution
    greedy_seq = _greedy_construct()
    best_seq = greedy_seq[:]
    best_len = len(greedy_seq)
    
    rng = random.Random(42)
    
    # Compute products for best_seq for fast checking
    def compute_products(seq):
        products = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                products.add(prod)
        return products
    
    best_products = compute_products(best_seq)
    
    while time.time() < deadline:
        # Start from best_seq, perturb by removing 1-3 elements
        seq = best_seq[:]
        if len(seq) > 3:
            num_removals = rng.randint(1, 3)
            indices_to_remove = set()
            while len(indices_to_remove) < num_removals and len(seq) > 1:
                idx = rng.randint(0, len(seq) - 1)
                indices_to_remove.add(idx)
            for idx in sorted(indices_to_remove, reverse=True):
                seq.pop(idx)
        
        # Greedy re-insertion: try to add back removed elements and new ones
        removed = [x for x in best_seq if x not in seq]
        candidates = removed + [x for x in range(2, 129) if x not in seq]
        rng.shuffle(candidates)
        
        current_products = compute_products(seq)
        
        for x in candidates:
            if time.time() > deadline:
                break
            # Try inserting x at end (maintain sorted order by only considering valid positions)
            # Since we need strictly increasing, x must be > last element or we need to check all positions
            # Simpler: just try appending if x > seq[-1] (if seq non-empty)
            if seq and x <= seq[-1]:
                continue
            if not seq and x < 2:
                continue
            
            # Check if adding x creates duplicate products
            new_products = set(current_products)
            prod = x
            for a in reversed(seq):
                if prod in new_products:
                    break
                new_products.add(prod)
                prod *= a
            else:
                # No duplicate found
                seq.append(x)
                current_products = new_products
        
        # Check if we improved
        if len(seq) > best_len:
            best_seq = seq[:]
            best_len = len(seq)
            best_products = compute_products(best_seq)
    
    return best_seq
# EVOLVE-BLOCK-END
