import time
import random
import sys

def _block_products(seq):
    """Return a set of all nonempty consecutive block products."""
    n = len(seq)
    products = set()
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            products.add(prod)
    return products

def _is_valid(seq):
    """Check if the sequence has unique block products."""
    n = len(seq)
    products = set()
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            if prod in products:
                return False
            products.add(prod)
    return True

def _greedy_construct():
    """Greedy baseline: add each x in 2..128 if valid."""
    values = []
    for x in range(2, 129):
        candidate = values + [x]
        if _is_valid(candidate):
            values = candidate
    return values

def _try_insert(seq, x):
    """Try to insert x into seq maintaining sorted order. Return new seq if valid, else None."""
    lo, hi = 0, len(seq)
    while lo < hi:
        mid = (lo + hi) // 2
        if seq[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    insert_idx = lo
    
    if insert_idx < len(seq) and seq[insert_idx] == x:
        return None

    new_seq = seq[:insert_idx] + [x] + seq[insert_idx:]
    if _is_valid(new_seq):
        return new_seq
    return None

def _try_swap(seq, i, x):
    """Try to replace seq[i] with x. Maintain sorted order."""
    if i > 0 and x <= seq[i-1]:
        return None
    if i < len(seq) - 1 and x >= seq[i+1]:
        return None
    if x == seq[i]:
        return None
    
    new_seq = seq[:i] + [x] + seq[i+1:]
    if _is_valid(new_seq):
        return new_seq
    return None

def _try_remove_insert(seq, remove_idx, insert_x):
    """Remove seq[remove_idx], then insert insert_x. Return new seq if valid and longer, else None."""
    if insert_x in seq:
        # If insert_x is already in seq, removing one element and adding another won't increase length
        # unless we're doing a swap, which is handled elsewhere
        return None
    
    # Remove
    new_seq = seq[:remove_idx] + seq[remove_idx+1:]
    
    # Insert
    result = _try_insert(new_seq, insert_x)
    if result is not None and len(result) == len(seq):
        # This is just a swap, not an improvement in length
        # But it might enable future insertions
        return result
    return None

def _try_remove_two_insert_one(seq, remove1, remove2, insert_x):
    """Remove two elements, insert one. This decreases length by 1, so not useful for maximizing length.
    But it might enable future insertions. Skip for now."""
    return None

def _local_search(seq, deadline, rng):
    """Local search: try insertions, swaps, and remove-insert moves."""
    best_seq = list(seq)
    best_len = len(best_seq)
    
    while time.time() < deadline:
        improved = False
        candidates = [x for x in range(2, 129) if x not in best_seq]
        
        if not candidates:
            break
        
        rng.shuffle(candidates)
        
        # Try all insertions
        for x in candidates:
            if time.time() > deadline:
                break
            new_seq = _try_insert(best_seq, x)
            if new_seq is not None:
                best_seq = new_seq
                best_len = len(best_seq)
                improved = True
                candidates = [x for x in range(2, 129) if x not in best_seq]
                rng.shuffle(candidates)
                break
        
        if time.time() > deadline:
            break
            
        if not improved:
            # No insertions found. Try swaps to change configuration.
            swap_found = False
            for i in range(len(best_seq)):
                if time.time() > deadline:
                    break
                sample_candidates = rng.sample(candidates, min(20, len(candidates))) if candidates else []
                for x in sample_candidates:
                    new_seq = _try_swap(best_seq, i, x)
                    if new_seq is not None:
                        best_seq = new_seq
                        swap_found = True
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                # Try remove-insert: remove one element, insert a different one
                # This maintains length but changes the set
                remove_insert_found = False
                for i in range(len(best_seq)):
                    if time.time() > deadline:
                        break
                    for x in candidates:
                        new_seq = _try_remove_insert(best_seq, i, x)
                        if new_seq is not None:
                            best_seq = new_seq
                            remove_insert_found = True
                            improved = True
                            break
                    if improved:
                        break
                
                if not improved:
                    # Stuck. Try a random restart or break.
                    break
    
    return best_seq

def _simulated_annealing(seq, deadline, rng):
    """Simulated annealing: try moves that may decrease quality temporarily."""
    best_seq = list(seq)
    current_seq = list(best_seq)
    current_len = len(current_seq)
    
    temperature = 10.0
    cooling = 0.95
    
    while time.time() < deadline and temperature > 0.1:
        candidates = [x for x in range(2, 129) if x not in current_seq]
        
        if not candidates:
            temperature *= cooling
            continue
        
        # Generate a random move
        move_type = rng.random()
        
        if move_type < 0.5:
            # Try insertion
            x = rng.choice(candidates)
            new_seq = _try_insert(current_seq, x)
            if new_seq is not None:
                # Insertion always improves length
                current_seq = new_seq
                if len(current_seq) > len(best_seq):
                    best_seq = current_seq
        elif move_type < 0.8:
            # Try swap
            if current_seq:
                i = rng.randrange(len(current_seq))
                x = rng.choice(candidates)
                new_seq = _try_swap(current_seq, i, x)
                if new_seq is not None:
                    # Swap maintains length; accept with some probability
                    if rng.random() < temperature:
                        current_seq = new_seq
        else:
            # Try remove-insert
            if current_seq:
                i = rng.randrange(len(current_seq))
                x = rng.choice(candidates)
                new_seq = _try_remove_insert(current_seq, i, x)
                if new_seq is not None:
                    if rng.random() < temperature:
                        current_seq = new_seq
        
        temperature *= cooling
    
    return best_seq

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    rng = random.Random(42)
    
    # 1. Greedy Initialization
    best_seq = _greedy_construct()
    best_len = len(best_seq)
    
    # 2. Local Search
    best_seq = _local_search(best_seq, deadline, rng)
    best_len = len(best_seq)
    
    # 3. Simulated Annealing if time remains
    if time.time() < deadline - 5.0:
        sa_seq = _simulated_annealing(best_seq, deadline, rng)
        if len(sa_seq) > best_len:
            best_seq = sa_seq
            best_len = len(best_seq)
    
    # 4. Final local search polish
    if time.time() < deadline - 2.0:
        best_seq = _local_search(best_seq, deadline, rng)
    
    return best_seq
