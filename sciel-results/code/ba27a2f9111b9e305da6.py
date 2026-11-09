import random
import time
import itertools

def is_sidon_4(subset):
    """Check if a 4-element subset is Sidon."""
    # Sort for consistency? Not needed for sum uniqueness check.
    # Sums:
    # s0+s0, s0+s1, s0+s2, s0+s3
    # s1+s1, s1+s2, s1+s3
    # s2+s2, s2+s3
    # s3+s3
    s0, s1, s2, s3 = subset
    sums = [
        2*s0, s0+s1, s0+s2, s0+s3,
        2*s1, s1+s2, s1+s3,
        2*s2, s2+s3,
        2*s3
    ]
    return len(set(sums)) == 10

def count_sidon_4_subsets(s):
    """Count number of Sidon 4-subsets in set s."""
    count = 0
    # Optimization: Precompute sums? 
    # For n=12, 495 subsets.
    # Just iterate.
    for combo in itertools.combinations(s, 4):
        if is_sidon_4(combo):
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    # We want to minimize count_sidon_4_subsets.
    # Target is 0.
    
    best_set = None
    best_score = float('inf')
    
    # Helper to update best
    def update_best(s, score):
        nonlocal best_set, best_score
        if score < best_score:
            best_score = score
            best_set = list(s)
            if best_score == 0:
                return True # Success
        return False

    # 1. Try some structured initial guesses
    # AP
    ap = list(range(12))
    score_ap = count_sidon_4_subsets(ap)
    if update_best(ap, score_ap):
        return best_set
        
    # Random initializations
    random.seed(42)
    
    # Simulated Annealing Parameters
    T = 1.0
    cooling_rate = 0.99
    min_T = 0.01
    
    # Main Loop
    while time.time() < deadline:
        # Random Restart
        current_set = set(random.sample(range(129), 12))
        current_score = count_sidon_4_subsets(list(current_set))
        
        if update_best(current_set, current_score):
            return best_set
            
        # Local Search / Annealing
        T = 1.0
        while T > min_T and time.time() < deadline:
            # Generate Neighbor
            # Strategy: Swap one element or replace one
            move_type = random.choice([0, 1])
            new_set = list(current_set)
            
            if move_type == 0:
                # Replace one element
                idx = random.randint(0, 11)
                old_val = new_set[idx]
                new_val = random.randint(0, 128)
                if new_val in new_set and new_val != old_val:
                    continue
                new_set[idx] = new_val
            else:
                # Swap two elements (values from a pool? or just random new values for two slots)
                # Let's stick to single replacement for simplicity and speed, 
                # but add a "kick" of 2 replacements if stuck.
                idx1 = random.randint(0, 11)
                idx2 = random.randint(0, 11)
                if idx1 == idx2:
                    idx2 = (idx1 + 1) % 12
                old1 = new_set[idx1]
                old2 = new_set[idx2]
                new1 = random.randint(0, 128)
                new2 = random.randint(0, 128)
                # Ensure distinctness
                if new1 in new_set and new1 != old1: continue
                new_set[idx1] = new1
                if new2 in new_set and new2 != old2: continue
                new_set[idx2] = new2

            new_score = count_sidon_4_subsets(new_set)
            
            delta = new_score - current_score
            
            # Accept if better, or with probability exp(-delta/T) if worse
            if delta < 0 or (delta == 0 and random.random() < 0.5) or random.random() < pow(2.718, -delta / max(T, 1e-9)):
                current_set = set(new_set)
                current_score = new_score
                
                if update_best(current_set, current_score):
                    return best_set
            
            T *= cooling_rate
        
        # If stuck for a while, restart with different seed or strategy
        random.seed(random.randint(0, 1000000))
        
    # Return best found
    return best_set
