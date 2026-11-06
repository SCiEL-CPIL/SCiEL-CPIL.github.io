import time
import random
from itertools import combinations

def _is_sidon_4_fast(a, b, c, d):
    """Check if a 4-element subset is Sidon (including diagonals)."""
    # Sums: 2a, a+b, a+c, a+d, 2b, b+c, b+d, 2c, c+d, 2d
    # We need all 10 to be distinct.
    # a,b,c,d are distinct.
    # Collisions possible:
    # 1. 2a == 2b (implies a==b, false)
    # 2. 2a == a+b (implies a==b, false)
    # 3. 2a == a+c (implies a==c, false)
    # 4. 2a == a+d (implies a==d, false)
    # 5. 2a == 2b (false)
    # 6. 2a == b+c (Possible: a is average of b,c)
    # 7. 2a == b+d (Possible)
    # 8. 2a == 2c (false)
    # 9. 2a == c+d (Possible)
    # 10. 2a == 2d (false)
    # ... and so on.
    # It's faster to just put them in a set.
    sums = [
        2*a,
        a+b, a+c, a+d,
        2*b, b+c, b+d,
        2*c, c+d,
        2*d
    ]
    return len(set(sums)) == 10

def _count_sidon_4(A):
    """Count the number of 4-element Sidon subsets in A."""
    count = 0
    n = len(A)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if _is_sidon_4_fast(A[i], A[j], A[k], A[l]):
                        count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin: evaluator kills at 60s (or 90s? Prompt says 90s wall clock, but 60s in contract. Let's use 58s to be safe).
    
    best_set = None
    best_count = float('inf')
    
    # We want to find a set with 0 Sidon-4 subsets.
    # If we can't, we return the one with the fewest (which will still have max Sidon >= 4).
    # But hopefully 0 is achievable.
    
    rng = random.Random(123)
    
    # Try multiple random starts with local search
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        # Initialize with a random set
        current_set = sorted(rng.sample(range(0, 129), 12))
        current_count = _count_sidon_4(current_set)
        
        if current_count == 0:
            return current_set
            
        if current_count < best_count:
            best_count = current_count
            best_set = current_set[:]
            
        # Local Search: Simulated Annealing / Hill Climbing
        T = 1.0
        for step in range(500):
            if time.time() > deadline:
                break
                
            # Pick a random index to change
            idx = rng.randint(0, 11)
            old_val = current_set[idx]
            
            # Try a few random new values
            improved = False
            for _ in range(10):
                new_val = rng.randint(0, 128)
                if new_val == old_val:
                    continue
                if new_val in current_set:
                    continue
                
                # Create new set
                new_set = current_set[:]
                new_set[idx] = new_val
                new_set.sort()
                
                # Calculate new count
                # Optimization: Only check 4-subsets involving idx?
                # Subsets involving idx: C(11, 3) = 165.
                # Subsets NOT involving idx: C(11, 4) = 330. These are unchanged.
                # So we only need to check the 165 subsets involving idx.
                # But we need the delta.
                # Let's just recompute for the 165 subsets.
                
                # Actually, let's just recompute the total count for simplicity first,
                # as 165 checks is very fast.
                
                # Count Sidon-4 in new_set involving idx
                # idx is at position pos in sorted new_set
                pos = new_set.index(new_val)
                
                # Find all 4-subsets containing pos
                # This is a bit complex to generate dynamically.
                # Given the small size, just calling _count_sidon_4 on new_set might be okay?
                # _count_sidon_4 does 495 checks.
                # 495 checks * 10 trials * 500 steps * 50 attempts = 1.2M checks.
                # Each check is ~10 ops. 12M ops. Should be fine in Python?
                # Maybe a bit slow. Let's optimize the check.
                
                # Let's try a faster specific check for the 165 subsets.
                # But implementing that correctly is error-prone.
                # Let's stick to full recompute but optimize _is_sidon_4_fast.
                
                new_count = _count_sidon_4(new_set)
                
                delta = new_count - current_count
                
                if delta < 0:
                    # Accept improvement
                    current_set = new_set
                    current_count = new_count
                    improved = True
                    if current_count == 0:
                        return current_set
                    break
                else:
                    # Accept worse with probability exp(-delta/T)
                    # delta is integer >= 0
                    if delta == 0:
                        # Neutral move, accept with 50% chance to explore
                        if rng.random() < 0.5:
                            current_set = new_set
                            # current_count unchanged
                            improved = True
                            break
                    elif rng.random() < 0.01: # Small probability to accept worse to escape local minima
                        current_set = new_set
                        current_count = new_count
                        improved = True
                        break
            
            if not improved:
                # If no move accepted, maybe perturb more aggressively
                # Just pick a random new value for a random element
                idx2 = rng.randint(0, 11)
                new_val2 = rng.randint(0, 128)
                if new_val2 not in current_set:
                    current_set[idx2] = new_val2
                    current_set.sort()
                    current_count = _count_sidon_4(current_set)
                    
            if current_count < best_count:
                best_count = current_count
                best_set = current_set[:]
                if best_count == 0:
                    return best_set
            
            # Cool down
            T *= 0.99

    return best_set if best_set else list(range(12))
