import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_set = []
    best_size = 0
    
    def run_greedy(order):
        chosen = []
        counts = {}
        for x in order:
            valid = True
            new_sums = []
            for y in chosen:
                s = x + y
                new_sums.append(s)
            new_sums.append(2 * x)
            
            for s in new_sums:
                if counts.get(s, 0) >= 2:
                    valid = False
                    break
            
            if valid:
                chosen.append(x)
                for s in new_sums:
                    counts[s] = counts.get(s, 0) + 1
        return chosen

    # 1. Ascending
    order = list(range(1, 1001))
    s = run_greedy(order)
    if len(s) > best_size:
        best_size = len(s)
        best_set = s
        
    # 2. Descending
    order = list(range(1000, 0, -1))
    s = run_greedy(order)
    if len(s) > best_size:
        best_size = len(s)
        best_set = s
        
    # 3. Random restarts
    rng = random.Random(42)
    
    while time.time() < deadline:
        order = list(range(1, 1001))
        rng.shuffle(order)
        s = run_greedy(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            
    # 4. Local search / swap improvement
    # Try to improve the best set by swapping elements
    # For each element in the set, try replacing it with an element not in the set
    # Check if the new set is still valid (all sums have at most 2 representations)
    
    def is_valid_set(s):
        """Check if set s is a valid B2[2] set."""
        counts = {}
        for i in range(len(s)):
            for j in range(i, len(s)):
                sum_val = s[i] + s[j]
                counts[sum_val] = counts.get(sum_val, 0) + 1
                if counts[sum_val] > 2:
                    return False
        return True

    def try_improve(s, rng, max_attempts=1000):
        """Try to improve set s by swapping elements."""
        s = sorted(s)
        s_set = set(s)
        all_nums = set(range(1, 1001))
        candidates = list(all_nums - s_set)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for _ in range(max_attempts):
                if time.time() > deadline:
                    break
                # Pick a random element to remove
                idx = rng.randint(0, len(s) - 1)
                old_val = s[idx]
                # Pick a random candidate to add
                new_val = rng.choice(candidates)
                
                # Try the swap
                new_s = s.copy()
                new_s[idx] = new_val
                new_s.sort()
                
                # Check validity (this is expensive, so we only do it sometimes)
                if is_valid_set(new_s):
                    # Update s and candidates
                    s = new_s
                    s_set = set(s)
                    candidates = list(all_nums - s_set)
                    improved = True
                    break  # Restart the inner loop with the new set
        
        return s

    # Apply local search to the best set found
    if best_set and time.time() < deadline:
        improved_set = try_improve(best_set, rng, max_attempts=500)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set = improved_set
        elif len(improved_set) == best_size:
            # Same size, but might be different structure - keep the one with smaller elements?
            # Actually, for the score, only size matters. So no improvement.
            pass
            
    return best_set
