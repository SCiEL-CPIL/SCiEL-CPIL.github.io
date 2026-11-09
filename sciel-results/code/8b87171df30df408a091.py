import random
import time
from itertools import combinations

def is_sidon(subset):
    """Check if a tuple/list of integers is a Sidon set (unique pair sums including diagonals)."""
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
    for k in range(12, 2, -1):
        if has_sidon_subset(S, k):
            return k
    if has_sidon_subset(S, 3):
        return 3
    return 2

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    best_set = None
    best_score = float('inf')
    
    # We want to minimize the maximum Sidon subset size.
    # The minimum possible for 12 distinct integers is 3 (since any 3 distinct integers 
    # a<b<c have sums a+a, a+b, a+c, b+b, b+c, c+c. 
    # a+a < a+b < a+c < b+b < b+c < c+c is not always true, but a+a, a+b, b+b are distinct 
    # unless a=b which is not allowed. So size 3 is always Sidon? 
    # Wait. 1, 2, 3. Sums: 2, 3, 4, 5, 6. All unique. 
    # 1, 1, 2 is not allowed (distinct).
    # So any 3 distinct integers form a Sidon set? 
    # Let a < b < c.
    # Sums: 2a, a+b, 2b, a+c, b+c, 2c.
    # 2a < a+b < 2b < a+c < b+c < 2c ?
    # a+b < 2b iff a < b (True).
    # 2b < a+c ? Not necessarily. If a=1, b=10, c=11. 2b=20, a+c=12. 20 > 12.
    # So 2b and a+c can collide? 
    # 2b = a+c => c = 2b - a. 
    # If c = 2b - a, then a, b, c form an arithmetic progression.
    # In that case, 2b = a+c. So the sums are not unique.
    # So a 3-element set is NOT Sidon if it is an arithmetic progression.
    # Therefore, the minimum max Sidon size is 3 if we can avoid 3-APs? 
    # No, we want to MINIMIZE the max Sidon size.
    # If we have a 3-AP, that 3-subset is not Sidon.
    # But there might be other 3-subsets that ARE Sidon.
    # In fact, if the whole set has no 3-AP, then every 3-subset is Sidon.
    # So max Sidon size would be at least 3.
    # Can we make max Sidon size 2?
    # That would mean NO 3-subset is Sidon.
    # That means EVERY 3-subset is an arithmetic progression.
    # This is impossible for 12 distinct integers.
    # So the minimum possible max Sidon size is 3.
    
    # Strategy:
    # 1. Try to construct a set where every 3-subset is an AP? Impossible.
    # 2. Try to construct a set where the largest Sidon subset is exactly 3.
    #    This means no 4-subset is Sidon.
    #    To do this, we want to maximize collisions in pair sums.
    
    # Greedy approach with collision maximization:
    # At each step, pick the next number that creates the most sum collisions
    # with existing numbers.
    
    # We'll run multiple trials with different random seeds/strategies.
    
    for trial in range(50):
        if time.time() - start_time > time_limit - 5:
            break
            
        current_set = []
        candidates = list(range(0, 129))
        sums = set()
        
        # Optional: Start with a small seed set that is "dense"
        # For example, pick 3 numbers that form an AP to kill one 3-Sidon.
        # But we want to kill ALL 4-Sidons.
        
        for i in range(12):
            best_x = None
            max_collisions = -1
            best_candidates = []
            
            for x in candidates:
                if x in current_set:
                    continue
                
                # Calculate new sums introduced by x
                new_sums = [2 * x]
                for c in current_set:
                    new_sums.append(x + c)
                
                # Count how many of these new sums already exist in 'sums'
                collisions = 0
                for s in new_sums:
                    if s in sums:
                        collisions += 1
                
                # Also, we want to avoid creating new unique sums that could
                # allow future Sidon sets? Hard to measure.
                # Let's just maximize collisions.
                
                if collisions > max_collisions:
                    max_collisions = collisions
                    best_candidates = [x]
                elif collisions == max_collisions:
                    best_candidates.append(x)
            
            if not best_candidates:
                best_x = random.choice(candidates)
            else:
                # Add some randomness to explore different collision patterns
                # Sometimes picking a slightly less colliding number might lead
                # to a better global configuration.
                if random.random() < 0.3 and len(best_candidates) > 1:
                    best_x = random.choice(best_candidates)
                else:
                    best_x = best_candidates[0] # or random.choice
            
            current_set.append(best_x)
            candidates.remove(best_x)
            
            # Update sums
            sums.add(2 * best_x)
            for c in current_set:
                if c != best_x:
                    sums.add(best_x + c)
        
        # Evaluate the constructed set
        score = get_max_sidon_size(current_set)
        
        if score < best_score:
            best_score = score
            best_set = current_set.copy()
            
            if best_score == 3:
                # We found a set with max Sidon size 3. This is likely optimal.
                # Let's verify with a few more checks or just break.
                # Given the difficulty, 3 is a great result.
                break
    
    if best_set is None:
        # Fallback: A random set
        best_set = random.sample(range(0, 129), 12)
    
    return best_set
