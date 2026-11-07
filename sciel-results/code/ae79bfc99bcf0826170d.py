import time
import random

def construct():
    deadline = time.time() + 55.0
    best_set = []
    best_size = 0
    
    def check_divisibility(old_sums, new_sums):
        """
        old_sums: set of existing subset sums
        new_sums: set of new subset sums (all > max(old_sums))
        Returns True if no new_sum is divisible by any old_sum.
        """
        for ns in new_sums:
            for os in old_sums:
                if ns % os == 0:
                    return False
        return True

    def build_set(start_val, randomize_choice=False):
        """
        Builds a set starting with start_val using the superincreasing heuristic.
        """
        current_set = [start_val]
        current_sums = {start_val}
        current_sum = start_val
        
        while True:
            candidates = []
            # Candidates must be > current_sum to ensure distinct subset sums
            # and <= 256
            for x in range(current_sum + 1, 257):
                if x in current_set:
                    continue
                
                # Generate new sums
                # New sums are x and x+s for s in current_sums
                new_sums_list = [x] + [x + s for s in current_sums]
                new_sums_set = set(new_sums_list)
                
                # Check distinctness within new sums (guaranteed by construction? 
                # x > sum(current_sums) >= s. 
                # x+s1 = x+s2 => s1=s2. 
                # x+s = x => s=0 (not in set). 
                # x+s = x+s' => s=s'. 
                # So yes, distinct.)
                
                # Check divisibility
                if check_divisibility(current_sums, new_sums_set):
                    candidates.append((x, new_sums_set))
            
            if not candidates:
                break
            
            if randomize_choice and len(candidates) > 1:
                # Pick a random candidate to explore different branches
                # Bias towards smaller x to leave room for more elements?
                # Or random? Let's try random among the first few valid ones.
                # Actually, smaller x allows more elements. 
                # Let's try smallest first, but if stuck, we restart.
                # To improve diversity, we can shuffle candidates slightly.
                random.shuffle(candidates)
                x, new_sums_set = candidates[0]
            else:
                # Greedy: pick smallest x to maximize potential size
                x, new_sums_set = min(candidates, key=lambda item: item[0])
            
            current_set.append(x)
            current_sums = current_sums.union(new_sums_set)
            current_sum += x
            
            if time.time() > deadline:
                break
        
        return current_set

    # Try different starting values and strategies
    # Starting with small values allows more elements, but might hit divisibility walls sooner.
    # Starting with larger values might allow fewer elements but avoid some divisibility traps.
    
    # Strategy 1: Greedy smallest-first from various starts
    for start in range(1, 50):
        if time.time() > deadline:
            break
        s = build_set(start, randomize_choice=False)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s[:]

    # Strategy 2: Randomized choices from various starts
    for _ in range(200):
        if time.time() > deadline:
            break
        start = random.randint(1, 50)
        s = build_set(start, randomize_choice=True)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s[:]
            
    # Strategy 3: Random full sets (fallback or to find non-superincreasing solutions)
    # This is slower, so we do it only if we have time left.
    for _ in range(50):
        if time.time() > deadline:
            break
        # Try to find a random set of size 4 or 5
        target_size = random.randint(4, 6)
        for _ in range(10):
            if time.time() > deadline:
                break
            candidates = random.sample(range(1, 257), target_size)
            # Quick check: sort and check superincreasing-like property?
            # No, just check validity properly if it's small.
            # But checking validity for random sets is expensive (2^n sums).
            # For n=6, 64 sums. Very fast.
            # For n=8, 256 sums. Still fast.
            # Let's just run a proper validity check for random sets.
            
            def is_valid_general(elements):
                n = len(elements)
                sums = [0]
                for e in elements:
                    sums += [s + e for s in sums]
                sums = [s for s in sums if s != 0]
                if len(sums) != len(set(sums)):
                    return False
                sums_set = set(sums)
                sums_list = sorted(sums_set)
                max_sum = sums_list[-1]
                for a in sums_list:
                    if a * 2 > max_sum:
                        continue
                    multiple = a * 2
                    while multiple <= max_sum:
                        if multiple in sums_set:
                            return False
                        multiple += a
                return True

            if is_valid_general(candidates):
                if len(candidates) > best_size:
                    best_size = len(candidates)
                    best_set = candidates[:]

    # Ensure best_set is a list of ints
    if best_set:
        return list(best_set)
    else:
        return [1]
