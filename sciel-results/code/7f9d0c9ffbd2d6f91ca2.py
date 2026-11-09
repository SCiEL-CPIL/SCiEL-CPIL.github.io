import time
import random
from itertools import combinations

def is_sidon(subset):
    """Check if a sorted list of integers is a Sidon set."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(ambient):
    """Find the size of the largest Sidon subset of the ambient set."""
    # Check from largest size down
    for k in range(len(ambient), 0, -1):
        for combo in combinations(ambient, k):
            if is_sidon(combo):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    # Initial candidate: consecutive integers
    # This is known to have high additive structure
    current_set = list(range(12))
    current_score = max_sidon_size(current_set)
    
    best_set = current_set
    best_score = current_score
    
    # Local Search
    # We want to minimize the score.
    # Min possible score is 2 (any 2 distinct integers are Sidon).
    # If we find 2, we stop.
    
    # Neighbor generation: stay dense
    def get_neighbor(s):
        idx = random.randint(0, len(s)-1)
        old_val = s[idx]
        # Small perturbation to maintain density
        delta = random.randint(-5, 5)
        new_val = old_val + delta
        
        # Clamp to [0, 128]
        if new_val < 0:
            new_val = 0
        elif new_val > 128:
            new_val = 128
            
        if new_val == old_val:
            return None
            
        new_s = s.copy()
        new_s[idx] = new_val
        
        # Ensure distinct
        if len(set(new_s)) < len(new_s):
            # Try to resolve collision by moving another element or rejecting
            return None
            
        return new_s

    iterations = 0
    stuck_count = 0
    
    while time.time() < deadline:
        iterations += 1
        neighbor = get_neighbor(current_set)
        if neighbor is None:
            # If we can't move, try a bigger jump or restart
            if random.random() < 0.1:
                # Random restart with dense set
                base = random.randint(0, 117)
                current_set = list(range(base, base + 12))
                current_score = max_sidon_size(current_set)
                if current_score < best_score:
                    best_score = current_score
                    best_set = current_set
            continue
        
        neighbor_score = max_sidon_size(neighbor)
        
        if neighbor_score < current_score:
            current_set = neighbor
            current_score = neighbor_score
            stuck_count = 0
            if current_score < best_score:
                best_score = current_score
                best_set = current_set
                # If we hit 2, it's optimal
                if best_score == 2:
                    break
        else:
            stuck_count += 1
            # Accept worse moves occasionally to escape local minima
            if stuck_count > 50 and random.random() < 0.2:
                current_set = neighbor
                current_score = neighbor_score
                stuck_count = 0

    return sorted(best_set)
