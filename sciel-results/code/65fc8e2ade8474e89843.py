import random
import time

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def check_valid(vals):
    """
    vals: sorted list of unique integers
    returns True if all consecutive block products are distinct
    """
    prods = set()
    n = len(vals)
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= vals[j]
            if prod in prods:
                return False
            prods.add(prod)
    return True

def construct():
    numbers = list(range(2, 129))
    primes = [x for x in numbers if is_prime(x)]
    composites = [x for x in numbers if not is_prime(x)]
    
    best_set = set(primes)
    best_len = len(best_set)
    
    start_time = time.time()
    
    # Heuristic: Sort composites by value descending (large composites are safer)
    # or by number of prime factors?
    # Let's stick to random restarts but with a local search step.
    
    while time.time() - start_time < 55:
        current_set = set(primes)
        
        # 1. Greedy Add Phase
        shuffled_composites = composites[:]
        random.shuffle(shuffled_composites)
        
        for c in shuffled_composites:
            if c in current_set:
                continue
            new_set = current_set | {c}
            new_vals = sorted(new_set)
            if check_valid(new_vals):
                current_set = new_set
        
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = set(current_set)
            
        # 2. Local Search Phase (Swap/Add/Remove)
        # Try to improve current_set by swapping composites
        improved = True
        while improved and (time.time() - start_time < 55):
            improved = False
            # Try removing a composite from current_set and adding a different one
            composites_in_set = [c for c in current_set if c in composites]
            composites_not_in_set = [c for c in composites if c not in current_set]
            
            if not composites_in_set or not composites_not_in_set:
                break
                
            random.shuffle(composites_in_set)
            random.shuffle(composites_not_in_set)
            
            for c_out in composites_in_set:
                if time.time() - start_time > 55:
                    break
                # Try removing c_out
                temp_set = current_set - {c_out}
                # Try adding a random c_in
                # To be efficient, just try a few
                for c_in in random.sample(composites_not_in_set, min(5, len(composites_not_in_set))):
                    candidate_set = temp_set | {c_in}
                    candidate_vals = sorted(candidate_set)
                    if check_valid(candidate_vals):
                        # If we swapped, length is same. 
                        # But maybe this swap allows further additions?
                        # For now, accept if valid.
                        # To ensure progress, we might require length increase in greedy phase, 
                        # but here we are just exploring the space.
                        if len(candidate_set) > len(current_set):
                             current_set = candidate_set
                             improved = True
                             break
                        # If same length, we might get stuck. 
                        # Only accept swap if it leads to a "better" state? 
                        # Hard to define. Let's just try to ADD more after a swap.
                        # Actually, simple swap keeps length same. 
                        # The goal is to find a set of MAX length.
                        # So we should only accept a swap if it opens up more additions.
                        # This is complex. 
                        # Let's stick to Greedy + Restarts, which is proven to work well for this size.
                        # The current program is already doing Restarts.
                        # Maybe the improvement is just better sorting?
                        pass
                if improved:
                    break

    return sorted(best_set)
