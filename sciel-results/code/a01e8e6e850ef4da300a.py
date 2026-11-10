import time
import random
import itertools

def _check_valid(A):
    """
    Checks if set A is valid:
    1. All non-empty subset sums are distinct.
    2. No two distinct subset sums divide one another.
    """
    n = len(A)
    sums = []
    # Generate all non-empty subset sums
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += A[i]
        sums.append(s)
        
    # Check for duplicate sums
    if len(sums) != len(set(sums)):
        return False
        
    sums_set = set(sums)
    max_sum = max(sums_set)
    
    # Check divisibility
    # For each sum s, check if any multiple 2s, 3s... is in sums_set
    for s in sums_set:
        # Optimization: only check multiples up to max_sum
        for multiple in range(2 * s, max_sum + 1, s):
            if multiple in sums_set:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin for 90s limit
    
    best_set = [31, 32, 34]  # Known valid size 3
    best_size = 3
    
    # We will try to find sets of size 4 and 5
    # Strategy: Random restarts + Local search
    
    # Pre-generate some random candidates? No, do it on the fly.
    
    # Helper to get a random valid set of size k
    def find_random_valid(k, max_val=256, trials=1000):
        for _ in range(trials):
            A = random.sample(range(1, max_val + 1), k)
            if _check_valid(A):
                return A
        return None

    # 1. Try to find size 4
    # We can bias towards numbers that are "far apart" or prime-like to avoid divisibility
    # But random sampling works.
    
    # Let's do a more structured search for size 4
    # Iterate over first 2 elements, then try to find 2 more
    # This might be too slow if done naively.
    
    # Let's stick to random restarts with a time check.
    # Since size 3 is small, we can try to extend it.
    
    current_best = list(best_set)
    
    # Phase 1: Random restarts for size 4 and 5
    # To increase hit rate, we can filter candidates:
    # - Elements should be > 10 (small numbers divide many things)
    # - Elements should be pairwise non-dividing
    
    def generate_candidate(k):
        # Sample k distinct numbers from 20..256
        # This reduces the chance of small divisors
        pool = list(range(20, 257))
        A = random.sample(pool, k)
        # Sort for determinism in check (not strictly needed but good)
        A.sort()
        return A

    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Try size 5 first, then 4
        # If we find size 5, we are happy.
        target_sizes = [5, 4] if best_size < 5 else [5]
        
        found_upgrade = False
        for k in target_sizes:
            if k <= best_size:
                continue
                
            # Try a few random candidates
            # Number of trials per loop iteration depends on speed
            # _check_valid for k=5 is very fast (31 sums)
            
            # Adaptive trials: try more if we have time
            trials = 50 if k == 5 else 100
            
            for _ in range(trials):
                if time.time() > deadline:
                    break
                A = generate_candidate(k)
                # Quick filter: pairwise non-dividing
                valid_pairwise = True
                for i in range(k):
                    for j in range(i+1, k):
                        if A[j] % A[i] == 0:
                            valid_pairwise = False
                            break
                    if not valid_pairwise:
                        break
                if not valid_pairwise:
                    continue
                    
                if _check_valid(A):
                    if k > best_size:
                        best_set = A
                        best_size = k
                        found_upgrade = True
                    break # Found one of this size, move to next size or restart loop
        
        # If we found a new best, we can try to extend it locally
        # But random restarts are often faster for small k
        # Let's also try local search on the current best
        if time.time() < deadline:
            # Try to add an element to best_set if best_size < 5
            if best_size < 5:
                # Try random additions
                for _ in range(20):
                    if time.time() > deadline:
                        break
                    new_elem = random.randint(1, 256)
                    if new_elem in best_set:
                        continue
                    new_set = best_set + [new_elem]
                    new_set.sort()
                    # Quick pairwise check
                    ok = True
                    for i in range(len(new_set)):
                        for j in range(i+1, len(new_set)):
                            if new_set[j] % new_set[i] == 0:
                                ok = False
                                break
                        if not ok: break
                    if ok and _check_valid(new_set):
                        best_set = new_set
                        best_size = len(best_set)
                        break
                        
    # Ensure return type is list of ints
    return [int(x) for x in best_set]

# Run a quick self-test if executed directly
if __name__ == "__main__":
    res = construct()
    print(f"Found set: {res}, size: {len(res)}")
    print(f"Valid: {_check_valid(res)}")
