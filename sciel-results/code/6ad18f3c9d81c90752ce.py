import time
import random
import math

def _is_valid(seq):
    n = len(seq)
    if n == 0:
        return True
    products = set()
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            if prod in products:
                return False
            products.add(prod)
    return True

def _can_insert_sorted(seq, x):
    """Assumes seq is sorted. Checks if x can be inserted."""
    lo, hi = 0, len(seq)
    while lo < hi:
        mid = (lo + hi) // 2
        if seq[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    if lo < len(seq) and seq[lo] == x:
        return None # Duplicate
    new_seq = seq[:lo] + [x] + seq[lo:]
    if _is_valid(new_seq):
        return new_seq
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    all_nums = set(range(2, 129))
    best_seq = []
    best_len = 0
    
    # Initial greedy
    seq = []
    for x in range(2, 129):
        new_seq = _can_insert_sorted(seq, x)
        if new_seq:
            seq = new_seq
    
    if len(seq) > best_len:
        best_seq = seq
        best_len = len(seq)
        
    # Simulated Annealing / Random Search
    # We will run multiple restarts or a long SA loop
    
    # Let's do a single long SA loop on the best found so far
    current_set = set(best_seq)
    current_seq = best_seq
    
    T = 1.0
    cooling = 0.99
    
    # We need a way to evaluate "quality" for swaps that don't change length.
    # Maybe use the number of "blocked" insertions?
    # Or just accept all valid swaps.
    
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            break
            
        move_type = random.choice(['add', 'remove', 'swap'])
        
        if move_type == 'add':
            unused = list(all_nums - current_set)
            if not unused: continue
            x = random.choice(unused)
            new_seq = _can_insert_sorted(current_seq, x)
            if new_seq:
                # Always accept if valid and increases length
                current_seq = new_seq
                current_set.add(x)
                if len(current_seq) > best_len:
                    best_seq = current_seq
                    best_len = len(current_seq)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
                
        elif move_type == 'remove':
            if not current_set: continue
            x = random.choice(list(current_set))
            new_set = current_set - {x}
            new_seq = sorted(new_set)
            # Always valid if subset of valid set? 
            # Yes, removing elements from a valid sequence keeps it valid.
            # So we don't need to check _is_valid for removal.
            current_set = new_set
            current_seq = new_seq
            # Length decreased. Accept with prob?
            # For SA, we accept if T is high.
            # But we want to maximize length. 
            # Let's only remove if we are stuck (no improvements for a while).
            if no_improve_count < 100:
                # Revert
                current_set.add(x)
                current_seq = sorted(current_set)
            else:
                no_improve_count += 1
                
        elif move_type == 'swap':
            if not current_set or not (all_nums - current_set):
                continue
            x = random.choice(list(current_set))
            y = random.choice(list(all_nums - current_set))
            
            new_set = current_set - {x} | {y}
            new_seq = sorted(new_set)
            
            if _is_valid(new_seq):
                # Accept swap (length unchanged)
                # To avoid cycling, we can just accept.
                current_set = new_set
                current_seq = new_seq
                # Don't reset no_improve_count, as length didn't change
                # But it might help escape a local optimum.
            else:
                no_improve_count += 1
                
        # Cool down
        T *= cooling
        if T < 0.01:
            T = 1.0 # Restart temperature?
            no_improve_count = 0 # Reset counter to allow more removals?

    return best_seq
