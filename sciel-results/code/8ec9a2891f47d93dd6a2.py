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
                # Since we want strictly increasing, we can find where x fits
                # But we must check product validity for the resulting sequence.
                # To keep it simple and fast, we just pick a random valid position
                # or try to append if possible, or insert at a random spot.
                # Actually, inserting at a random spot changes all products.
                # Let's just try to insert x at a random valid index (sorted).
                # We generate the sorted list with x inserted, then check validity.
                
                # Find all indices where x can be inserted to maintain sort
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
    
    # Phase 1: Greedy Initialization
    # Try natural order
    seq1 = _greedy_init(list(range(2, 129)))
    best = seq1
    best_len = len(best)
    
    # Try a few random shuffles for better init
    random.seed(42)
    for i in range(5):
        order = list(range(2, 129))
        random.shuffle(order)
        seq = _greedy_init(order)
        if len(seq) > best_len:
            best = seq
            best_len = len(seq)
            
    # Phase 2: Simulated Annealing
    # We will run SA for the remaining time
    # Split time into a few restarts if possible, or one long run
    # One long run with restarts inside is hard to manage simply.
    # Let's do one long SA run starting from best.
    
    # If we have time for multiple runs, we can restart from best periodically.
    # But SA with cooling usually benefits from a single long run or periodic reheat.
    # Let's just run one SA run until deadline - 5s.
    
    best = _sa_search(best, deadline - 5.0, seed=123)
    
    # Check if we have time for another run from a different start
    if time.time() < deadline - 10.0:
        # Try a different initial state
        # Maybe a random subset?
        # Or just restart SA from current best with different seed
        # Reheating: Start T high again
        # The _sa_search function starts with T=50.0.
        # We can just call it again.
        best = _sa_search(best, deadline - 2.0, seed=456)
        
    # Final validation
    if not _is_valid(best):
        # Fallback
        best = _greedy_init(list(range(2, 129)))
        
    return best

# EVOLVE-BLOCK-END
