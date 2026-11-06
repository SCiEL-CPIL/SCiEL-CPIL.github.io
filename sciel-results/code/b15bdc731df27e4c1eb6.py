import time
import random
from math import exp

# EVOLVE-BLOCK-START

def _get_products(seq):
    """
    Returns a set of all non-empty consecutive block products.
    Also returns the count of products.
    """
    prods = set()
    n = len(seq)
    for i in range(n):
        p = 1
        for j in range(i, n):
            p *= seq[j]
            prods.add(p)
    return prods

def _is_valid(seq):
    """
    Check if seq is strictly increasing and all block products are distinct.
    """
    if not seq:
        return True
    # Check strictly increasing
    for i in range(len(seq) - 1):
        if seq[i] >= seq[i + 1]:
            return False
    
    # Check distinct products
    n = len(seq)
    expected_count = n * (n + 1) // 2
    prods = _get_products(seq)
    return len(prods) == expected_count

def _greedy_init(order):
    """
    Build a sequence greedily based on the given order of candidates.
    """
    values = []
    products = set()
    for x in order:
        if x < 2 or x > 128:
            continue
        # Find valid insertion points to maintain strict increase
        # We can just try to append if x > values[-1]
        # But greedy usually appends to the end in these contexts.
        # However, the sequence must be strictly increasing.
        # If we iterate 2..128, we just append if valid.
        if values and x <= values[-1]:
            continue
        
        # Check if adding x to the end creates duplicate products
        # New products are: x, x*values[-1], x*values[-1]*values[-2], ...
        new_prods = []
        p = x
        new_prods.append(p)
        for a in reversed(values):
            p *= a
            new_prods.append(p)
            
        # If any new product is in existing products, invalid
        if any(prod in products for prod in new_prods):
            continue
            
        values.append(x)
        products.update(new_prods)
        
    return values

def _local_polish(seq, deadline):
    """
    Systematic local polish: try to insert each missing element at every valid position.
    """
    current = list(seq)
    current_len = len(current)
    all_candidates = list(range(2, 129))
    
    # Repeat until no improvement or deadline
    improved = True
    while improved and time.time() < deadline:
        improved = False
        missing = [x for x in all_candidates if x not in current]
        
        for x in missing:
            # Find all valid insertion indices
            valid_indices = []
            for i in range(len(current) + 1):
                if i == 0:
                    if i == len(current) or x < current[0]:
                        valid_indices.append(i)
                elif i == len(current):
                    if x > current[-1]:
                        valid_indices.append(i)
                else:
                    if current[i-1] < x < current[i]:
                        valid_indices.append(i)
            
            for idx in valid_indices:
                neighbor = current[:idx] + [x] + current[idx:]
                if _is_valid(neighbor):
                    current = neighbor
                    current_len = len(current)
                    improved = True
                    # After insertion, recompute missing
                    missing = [x for x in all_candidates if x not in current]
                    break
            if time.time() >= deadline:
                break
    
    return current

def _sa_search(initial_seq, deadline, seed=None):
    """
    Simulated Annealing to maximize length of valid sequence.
    """
    if seed is not None:
        random.seed(seed)
        
    current = list(initial_seq)
    best = list(current)
    best_len = len(best)
    current_len = len(current)
    
    T = 50.0
    cool = 0.999
    min_T = 0.1
    
    # Precompute all possible candidates
    all_candidates = list(range(2, 129))
    
    while time.time() < deadline and T > min_T:
        # Generate a neighbor
        neighbor = None
        move_type = random.random()
        
        if move_type < 0.5:
            # Add a number
            missing = [x for x in all_candidates if x not in current]
            if missing:
                x = random.choice(missing)
                # Find valid insertion positions
                valid_indices = []
                for i in range(len(current) + 1):
                    if i == 0:
                        if i == len(current) or x < current[0]:
                            valid_indices.append(i)
                    elif i == len(current):
                        if x > current[-1]:
                            valid_indices.append(i)
                    else:
                        if current[i-1] < x < current[i]:
                            valid_indices.append(i)
                
                if valid_indices:
                    idx = random.choice(valid_indices)
                    neighbor = current[:idx] + [x] + current[idx:]
                    
        elif move_type < 0.7:
            # Remove a number
            if current:
                idx = random.randrange(len(current))
                neighbor = current[:idx] + current[idx+1:]
                
        else:
            # Swap: remove one, add one
            if current:
                missing = [x for x in all_candidates if x not in current]
                if missing:
                    x_remove_idx = random.randrange(len(current))
                    x_remove = current[x_remove_idx]
                    x_add = random.choice(missing)
                    
                    # Remove x_remove
                    temp = current[:x_remove_idx] + current[x_remove_idx+1:]
                    
                    # Find valid insertion positions for x_add in temp
                    valid_indices = []
                    for i in range(len(temp) + 1):
                        if i == 0:
                            if i == len(temp) or x_add < temp[0]:
                                valid_indices.append(i)
                        elif i == len(temp):
                            if x_add > temp[-1]:
                                valid_indices.append(i)
                        else:
                            if temp[i-1] < x_add < temp[i]:
                                valid_indices.append(i)
                    
                    if valid_indices:
                        idx = random.choice(valid_indices)
                        neighbor = temp[:idx] + [x_add] + temp[idx:]
        
        if neighbor is None:
            T *= cool
            continue
            
        # Check validity
        if not _is_valid(neighbor):
            T *= cool
            continue
            
        new_len = len(neighbor)
        delta = new_len - current_len
        
        if delta > 0:
            # Always accept better
            current = neighbor
            current_len = new_len
            if new_len > best_len:
                best = neighbor
                best_len = new_len
        else:
            # Accept worse with probability exp(delta / T)
            # delta is <= 0
            prob = exp(delta / T)
            if random.random() < prob:
                current = neighbor
                current_len = new_len
                
        T *= cool
        
    return best

def construct():
    start = time.time()
    deadline = start + 85.0  # Safety margin
    
    # Phase 1: Greedy Initialization with multiple strategies
    best = None
    best_len = 0
    
    # Strategy 1: Natural order
    seq = _greedy_init(list(range(2, 129)))
    if len(seq) > best_len:
        best = seq
        best_len = len(seq)
    
    # Strategy 2: Reverse natural order
    seq = _greedy_init(list(range(128, 1, -1)))
    if len(seq) > best_len:
        best = seq
        best_len = len(seq)
    
    # Strategy 3: Primes first, then composites
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    composites = [x for x in range(2, 129) if x not in primes]
    seq = _greedy_init(primes + composites)
    if len(seq) > best_len:
        best = seq
        best_len = len(seq)
    
    # Strategy 4: Random shuffles
    random.seed(42)
    for i in range(10):
        order = list(range(2, 129))
        random.shuffle(order)
        seq = _greedy_init(order)
        if len(seq) > best_len:
            best = seq
            best_len = len(seq)
    
    # Phase 2: Local polish on best
    if time.time() < deadline - 30.0:
        best = _local_polish(best, deadline - 30.0)
        best_len = len(best)
    
    # Phase 3: Simulated Annealing
    remaining = deadline - time.time()
    if remaining > 5.0:
        sa_deadline = deadline - 5.0
        best = _sa_search(best, sa_deadline, seed=123)
    
    # Phase 4: Final local polish
    if time.time() < deadline - 2.0:
        best = _local_polish(best, deadline - 2.0)
    
    # Final validation
    if not _is_valid(best):
        # Fallback
        best = _greedy_init(list(range(2, 129)))
    
    return best

# EVOLVE-BLOCK-END
