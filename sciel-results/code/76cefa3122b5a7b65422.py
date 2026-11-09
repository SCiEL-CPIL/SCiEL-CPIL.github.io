import time
import random

def construct():
    start_time = time.time()
    
    # Heuristic: Try to build a set using a greedy approach with "independence" bias
    # We prefer numbers that are large enough to not be easily divided by smaller sums.
    
    def is_valid_set(sums_set):
        # Check if any sum in sums_set divides another
        # sums_set is a set of integers
        # Optimization: iterate through sums, check if any multiple is in the set
        for s in sums_set:
            if s == 0:
                continue
            # Check multiples
            # Max sum is bounded by 256 * len(sums_set)
            # But we can just iterate k
            # To be safe, check up to max_sum
            max_sum = max(sums_set) if sums_set else 0
            for k in range(2, max_sum // s + 1):
                if s * k in sums_set:
                    return False
        return True

    def get_sums(current_set):
        sums = {0}
        for x in current_set:
            new_sums = {s + x for s in sums}
            sums.update(new_sums)
        sums.discard(0)
        return sums

    best_set = []
    best_size = 0
    
    # Strategy 1: Greedy with superincreasing bias
    # Start with small primes, but ensure superincreasing property where possible
    # This ensures distinct sums.
    
    candidates = list(range(1, 257))
    
    # Try different starting points and biases
    for start_val in [2, 3, 5, 7, 11]:
        current_set = []
        current_sums = set()
        
        for x in candidates:
            if time.time() - start_time > 50:
                break
            
            # Skip if x is too small to be "independent"
            # Simple heuristic: x should be > sum(current_set) / 2 ?
            # Or just try adding
            
            # Check if adding x is valid
            new_sums = set(current_sums)
            for s in current_sums:
                new_sums.add(s + x)
            new_sums.add(x)
            
            # Check for duplicate sums (distinct subsets must have distinct sums)
            # Since we are adding x to existing sums, duplicates can only occur if
            # x is already in current_sums (impossible as x > max(current_set) usually)
            # or if s1 + x == s2 + x (implies s1==s2)
            # or if s + x == x (implies s==0, but 0 is not in current_sums)
            # So sums remain distinct if current_sums are distinct.
            
            # Check divisibility
            if is_valid_set(new_sums):
                current_set.append(x)
                current_sums = new_sums
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
        
        if time.time() - start_time > 50:
            break

    # Strategy 2: Randomized local search to improve best_set
    # Try to swap elements or add new ones
    
    for _ in range(100):
        if time.time() - start_time > 55:
            break
        
        current_set = best_set[:]
        current_sums = get_sums(current_set)
        
        # Try to add a random number
        added = False
        for _ in range(10):
            x = random.randint(1, 256)
            if x in current_set:
                continue
            
            new_sums = set(current_sums)
            for s in current_sums:
                new_sums.add(s + x)
            new_sums.add(x)
            
            if is_valid_set(new_sums):
                current_set.append(x)
                current_sums = new_sums
                added = True
                break
        
        if not added:
            # Try to swap an element
            if current_set:
                idx = random.randint(0, len(current_set) - 1)
                old_x = current_set[idx]
                for _ in range(10):
                    new_x = random.randint(1, 256)
                    if new_x in current_set:
                        continue
                    
                    # Remove old_x, add new_x
                    temp_set = current_set[:]
                    temp_set.remove(old_x)
                    temp_set.append(new_x)
                    temp_sums = get_sums(temp_set)
                    
                    if is_valid_set(temp_sums):
                        current_set = temp_set
                        current_sums = temp_sums
                        break
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]

    return best_set
