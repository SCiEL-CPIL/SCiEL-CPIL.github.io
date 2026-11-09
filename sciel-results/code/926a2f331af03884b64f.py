import random
import time
from itertools import combinations

def is_sidon(subset):
    """Check if a tuple/list of integers is a Sidon set (unique pair sums)."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def has_sidon_subset(S, size):
    """Check if any subset of S of given size is Sidon."""
    for comb in combinations(S, size):
        if is_sidon(comb):
            return True
    return False

def get_max_sidon_size(S):
    """Find the size of the largest Sidon subset of S."""
    # Check down from 12
    for k in range(12, 2, -1):
        if has_sidon_subset(S, k):
            return k
    if has_sidon_subset(S, 3):
        return 3
    return 2

def construct():
    start_time = time.time()
    time_limit = 50.0
    
    # Greedy construction to maximize sum collisions
    # This forces the set to be "dense" in a way that limits Sidon subsets
    
    best_set = None
    best_score = float('inf')
    
    # We will run multiple greedy trials with different random tie-breakers
    # to explore the space of "collision-heavy" sets
    
    for trial in range(20): # 20 trials, each fast
        if time.time() - start_time > time_limit - 5:
            break
            
        current_set = []
        candidates = list(range(0, 129))
        sums = set() # Set of all a+b for a,b in current_set, a<=b
        
        for i in range(12):
            best_x = None
            max_collisions = -1
            best_candidates = []
            
            for x in candidates:
                if x in current_set:
                    continue
                
                # Calculate new sums introduced by x
                # New sums: 2x, x + c for c in current_set
                new_sums = []
                new_sums.append(2 * x)
                for c in current_set:
                    new_sums.append(x + c)
                
                # Count how many of these new sums already exist in 'sums'
                collisions = 0
                for s in new_sums:
                    if s in sums:
                        collisions += 1
                
                # Tie-breaking: if collisions are equal, prefer smaller x to keep range tight?
                # Or random?
                # Let's add a small random noise to break ties and allow diversity
                # But primarily maximize collisions.
                
                if collisions > max_collisions:
                    max_collisions = collisions
                    best_candidates = [x]
                elif collisions == max_collisions:
                    best_candidates.append(x)
            
            if not best_candidates:
                # Fallback if something went wrong
                best_x = random.choice(candidates)
            else:
                # Pick random from best candidates to avoid deterministic loops
                best_x = random.choice(best_candidates)
            
            # Add best_x to current_set
            current_set.append(best_x)
            candidates.remove(best_x)
            
            # Update sums
            # Add 2*best_x
            sums.add(2 * best_x)
            # Add best_x + c for all c in current_set (excluding best_x itself to avoid double counting? No, sums set handles duplicates)
            # Actually, we need to add all pairs involving best_x
            for c in current_set:
                if c != best_x:
                    sums.add(best_x + c)
        
        # Evaluate the constructed set
        # We want to minimize the max Sidon size
        # We can estimate it or compute it exactly
        # Since 12 elements, exact computation is feasible if we are careful
        # But we only need to compare.
        # Let's compute max Sidon size for this set.
        
        # Optimization: If we find a Sidon subset of size 4, we know it's >= 4.
        # If we don't, it's <= 3.
        # We want to minimize this.
        
        # Check if size 4 exists
        if has_sidon_subset(current_set, 4):
            # It's at least 4.
            # We can try to see if it's 5, but 4 is already "bad" if we can get 3.
            # Let's just record it as 4 for now, or compute exactly if we have time.
            # Given the greedy nature, it might be 4 or 5.
            # Let's compute exactly to be sure, as 20 trials * exact check is fast.
            score = get_max_sidon_size(current_set)
        else:
            # Max size is <= 3
            score = get_max_sidon_size(current_set)
        
        if score < best_score:
            best_score = score
            best_set = current_set.copy()
            
            # If we found a set with max size 3, it's likely optimal (or close).
            # Can we get 2? Unlikely for 12 distinct ints.
            # If best_score == 3, we can break early or continue to see if 2 is possible (very hard).
            if best_score == 3:
                # We found a good one. Let's keep looking for a few more to ensure robustness,
                # or break if we have time.
                # Actually, 3 is the target. Let's break to save time for safety.
                # But maybe there's a 2? No, 12 distinct ints must have a 3-AP or Sidon 3.
                # If no 3-AP, then 3 is Sidon.
                # So min is 3.
                break

    if best_set is None:
        # Fallback
        best_set = random.sample(range(0, 20), 12)
        
    return best_set
