import time
import random
from itertools import combinations

def is_sidon_4(subset):
    """Check if a 4-element subset is Sidon."""
    # subset is a tuple of 4 ints
    sums = set()
    # 10 sums: (0,0), (0,1), (0,2), (0,3), (1,1), (1,2), (1,3), (2,2), (2,3), (3,3)
    # Indices: 0,1,2,3
    s = subset[0] + subset[0]
    if s in sums: return False
    sums.add(s)
    s = subset[0] + subset[1]
    if s in sums: return False
    sums.add(s)
    s = subset[0] + subset[2]
    if s in sums: return False
    sums.add(s)
    s = subset[0] + subset[3]
    if s in sums: return False
    sums.add(s)
    s = subset[1] + subset[1]
    if s in sums: return False
    sums.add(s)
    s = subset[1] + subset[2]
    if s in sums: return False
    sums.add(s)
    s = subset[1] + subset[3]
    if s in sums: return False
    sums.add(s)
    s = subset[2] + subset[2]
    if s in sums: return False
    sums.add(s)
    s = subset[2] + subset[3]
    if s in sums: return False
    sums.add(s)
    s = subset[3] + subset[3]
    if s in sums: return False
    sums.add(s)
    return True

def count_sidon_4(S):
    """Count the number of Sidon subsets of size 4 in S."""
    count = 0
    for combo in combinations(S, 4):
        if is_sidon_4(combo):
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    # Initial candidate: range(12) is dense, likely low Sidon size
    best_set = list(range(12))
    best_count = count_sidon_4(best_set)
    
    if best_count == 0:
        return best_set

    # Local Search / Simulated Annealing
    current_set = best_set[:]
    current_count = best_count
    
    # Temperature for annealing
    T = 1.0
    cooling_rate = 0.99
    
    while time.time() < deadline:
        T *= cooling_rate
        
        # Pick a random index to perturb
        i = random.randint(0, 11)
        orig_val = current_set[i]
        
        # Try a few random new values
        # Bias towards values close to original to maintain density?
        # Or random in [0, 128]?
        # Dense sets are better, so try small range first, then wider.
        found_better = False
        for _ in range(10):
            if time.time() > deadline:
                break
            
            # Generate new value
            # 50% chance small perturbation, 50% random
            if random.random() < 0.5:
                delta = random.randint(-5, 5)
                new_val = orig_val + delta
            else:
                new_val = random.randint(0, 128)
            
            if new_val < 0 or new_val > 128:
                continue
            if new_val in current_set and new_val != orig_val:
                continue
                
            new_set = current_set[:]
            new_set[i] = new_val
            new_set = sorted(new_set)
            
            # Check distinctness (already handled by set check above mostly, but sort ensures order)
            if len(set(new_set)) != 12:
                continue
                
            new_count = count_sidon_4(new_set)
            
            if new_count < current_count:
                current_set = new_set
                current_count = new_count
                found_better = True
                if current_count == 0:
                    return current_set
                break
            elif new_count == current_count:
                # Accept to explore
                if random.random() < 0.5:
                    current_set = new_set
                    break
            else:
                # Accept worse with probability
                # Delta is positive
                prob = 0.01 * T
                if random.random() < prob:
                    current_set = new_set
                    current_count = new_count
                    break
        
        # Update best
        if current_count < best_count:
            best_count = current_count
            best_set = current_set[:]
            if best_count == 0:
                return best_set

    return best_set
