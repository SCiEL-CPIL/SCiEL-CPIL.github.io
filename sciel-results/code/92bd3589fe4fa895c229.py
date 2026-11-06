"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_sidon(subset):
    """Check if a set of integers is Sidon (all pair sums i<=j distinct)."""
    # Sort to ensure consistent ordering, though for sums it doesn't matter for uniqueness
    # But distinctness of sums is independent of order.
    n = len(subset)
    sums = set()
    for i in range(n):
        ai = subset[i]
        for j in range(i, n):
            sm = ai + subset[j]
            if sm in sums:
                return False
            sums.add(sm)
    return True

def _has_sidon_subset_of_size_k(arr, k):
    """Check if arr contains a Sidon subset of size k. Returns True if found."""
    n = len(arr)
    if k > n:
        return False
    # If k is small, check all combinations
    # For k=4, C(12,4) = 495. This is very fast.
    for combo in combinations(arr, k):
        if _is_sidon(combo):
            return True
    return False

def _max_sidon_size_fast(arr):
    """Find the largest Sidon subset size of arr. Optimized for small n."""
    n = len(arr)
    # We know min is 3 if 3-AP-free, but we want the exact max.
    # Check from n down to 3.
    # If n is 12, checking C(12,12)=1, C(12,11)=12, C(12,10)=66...
    # It's faster to check if there is a subset of size k.
    for k in range(n, 2, -1):
        if _has_sidon_subset_of_size_k(arr, k):
            return k
    # Check size 3
    if _has_sidon_subset_of_size_k(arr, 3):
        return 3
    return 2

def construct():
    deadline = time.time() + 50.0  # Safety margin for 60s limit
    
    # Initial candidate: known to have max Sidon size 4
    best_set = list(range(12))
    best_score = _max_sidon_size_fast(best_set)
    
    # Target: Find a set with max Sidon size 3.
    # This requires that NO 4-element subset is Sidon.
    
    # Strategy: Random Local Search
    # Start with random sets or structured sets.
    # Use _has_sidon_subset_of_size_k(arr, 4) as the primary filter for improvement.
    # If a set has no 4-Sidon subset, its max Sidon size is 3 (assuming it has a 3-Sidon subset, which is likely).
    
    random.seed(42)
    
    # Pre-compute some structured candidates
    candidates = []
    
    # 1. Random sets
    for _ in range(100):
        candidates.append(random.sample(range(129), 12))
        
    # 2. Structured sets (e.g., based on modular arithmetic)
    # Try sets that are "dense" in a way that creates many collisions.
    # E.g., {0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14}
    candidates.append([0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14])
    candidates.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    candidates.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15])
    
    # Evaluate initial candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        if len(set(cand)) != 12:
            continue
        if any(x < 0 or x > 128 for x in cand):
            continue
        
        # Quick check: does it have a 4-Sidon subset?
        # If not, max sidon size is likely 3.
        if not _has_sidon_subset_of_size_k(cand, 4):
            # Verify it has a 3-Sidon subset (should be true for random/structured)
            if _has_sidon_subset_of_size_k(cand, 3):
                best_set = cand[:]
                best_score = 3
                # We found a score of 3. Can we go lower?
                # Min is 3. So this is optimal?
                # Wait, is it possible to have max Sidon size 2?
                # As analyzed in inspiration 2, no. Any 3-AP-free set has all 3-subsets as Sidon.
                # To have max Sidon size 2, every 3-subset must NOT be Sidon.
                # This means every 3-subset has a collision.
                # Collision in 3 elements {a,b,c}: 2a=b+c, etc.
                # This implies the set is NOT 3-AP-free? No, it implies it HAS 3-APs?
                # If {a,b,c} has a collision, it's a 3-AP.
                # So if EVERY 3-subset is a 3-AP, then max Sidon size is 2.
                # Can we have a 12-element set where every 3-subset is a 3-AP?
                # No. If {a,b,c} is AP and {a,b,d} is AP, then c=d or ...
                # It is impossible for every 3-subset to be an AP for n>3.
                # So max Sidon size 2 is impossible.
                # So 3 is the minimum.
                # We can return immediately?
                # The task says "MINIMIZE that maximum subset size".
                # If 3 is the theoretical minimum, we are done.
                return best_set

    # If we didn't find one quickly, do local search
    current = best_set[:]
    current_score = best_score
    
    while time.time() < deadline:
        improved = False
        for _ in range(100):
            if time.time() > deadline:
                break
            i = random.randint(0, 11)
            new_val = random.randint(0, 128)
            if new_val in current:
                continue
            old_val = current[i]
            current[i] = new_val
            
            # Check if this move improves the score
            # We want to REDUCE the max Sidon size.
            # Current best is likely 4. We want 3.
            # So we check if the new set has NO 4-Sidon subset.
            if not _has_sidon_subset_of_size_k(current, 4):
                # It has no 4-Sidon subset.
                # Check if it has a 3-Sidon subset.
                if _has_sidon_subset_of_size_k(current, 3):
                    current_score = 3
                    improved = True
                    # Found optimal!
                    return current
                else:
                    # Max Sidon size 2? Impossible.
                    current[i] = old_val
            else:
                # It still has a 4-Sidon subset.
                # Maybe it has a smaller max Sidon size than current?
                # Unlikely if current is 4.
                # We can skip detailed check for speed, or do it.
                # For now, revert.
                current[i] = old_val
                
        if not improved:
            # Try a random restart
            current = random.sample(range(129), 12)
            if not _has_sidon_subset_of_size_k(current, 4):
                if _has_sidon_subset_of_size_k(current, 3):
                    return current

    return best_set
# EVOLVE-BLOCK-END
