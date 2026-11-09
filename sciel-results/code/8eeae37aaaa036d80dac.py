import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s to be safe within 60s limit
    
    # 1. Initial Greedy Solution
    # The greedy approach adds x if it doesn't create a duplicate product.
    # We iterate x from 2 to 128.
    values, products = [], set()
    for x in range(2, 129):
        # Calculate all new products ending at x
        tails = [x]
        prod = x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    
    best_values = values[:]
    best_len = len(values)
    
    # Helper to check validity of a set (converted to sorted list)
    def is_valid(seq):
        prods = set()
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                if p in prods:
                    return False
                prods.add(p)
        return True

    # Helper to get sorted list from set
    def get_sorted(s):
        return sorted(s)
    
    # Current state
    current_set = set(best_values)
    current_len = best_len
    
    # Iterated Local Search
    while time.time() < deadline:
        # Phase 1: Try to add any element that keeps validity
        added = False
        candidates = [x for x in range(2, 129) if x not in current_set]
        random.shuffle(candidates)
        
        for x in candidates:
            new_set = current_set | {x}
            if is_valid(get_sorted(new_set)):
                current_set = new_set
                current_len += 1
                added = True
                # Update best
                if current_len > best_len:
                    best_len = current_len
                    best_values = get_sorted(current_set)
                break # Restart candidate list after one addition? Or continue?
                # Usually, adding one at a time is safer to maintain validity logic.
        
        if added:
            continue # Loop back to try adding more
        
        # Phase 2: Stuck. Perform a kick (remove some elements) and repair.
        # Remove 1-3 random elements
        kick_size = random.randint(1, 3)
        if current_len > kick_size:
            to_remove = random.sample(list(current_set), kick_size)
            for x in to_remove:
                current_set.remove(x)
            current_len -= kick_size
            
            # Repair: Greedily add elements back
            # We want to recover length.
            # Try adding elements in random order or sorted order?
            # Sorted order might be more consistent. Random allows exploration.
            avail = [x for x in range(2, 129) if x not in current_set]
            random.shuffle(avail)
            
            for x in avail:
                if time.time() > deadline:
                    break
                new_set = current_set | {x}
                if is_valid(get_sorted(new_set)):
                    current_set = new_set
                    current_len += 1
            
            # If we improved or maintained, we are in a new local optimum.
            # If we got worse, we might want to revert? 
            # But ILS usually keeps the perturbed state if it leads to a new basin.
            # However, if we dropped length, we should probably revert to the previous best 
            # if we don't expect to recover it quickly. 
            # But the repair phase *does* try to recover. 
            # If current_len < best_len, we might just stay here and try again, 
            # or revert to best_values.
            
            # Strategy: If we didn't beat best_len, revert to best_values?
            # No, that might get stuck in the same local optimum.
            # Let's keep the current_set even if smaller, as the kick changed the landscape.
            # But to be safe, if we are significantly worse, revert.
            if current_len < best_len - 2:
                 current_set = set(best_values)
                 current_len = best_len

    return best_values
