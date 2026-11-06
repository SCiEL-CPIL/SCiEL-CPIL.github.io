import time
import itertools

def _is_smooth(n, primes):
    for p in primes:
        while n % p == 0:
            n //= p
    return n == 1

def construct():
    start_time = time.time()
    # Safety margin: 90s limit, use 50s for search
    deadline = start_time + 50.0
    
    # Candidate prime sets
    # We prioritize sets with smaller primes and smaller size
    base_primes = [2, 3, 5, 7, 11, 13, 17, 19]
    
    candidates = []
    # Try size 2, 3, 4, 5
    for size in range(2, 6):
        for combo in itertools.combinations(base_primes, size):
            candidates.append(combo)
            
    # Sort by product of primes to prefer "denser" smooth numbers
    # Actually, lexicographical sort of the tuple is good enough for small primes
    # But product is a better metric for density
    candidates.sort(key=lambda x: 1) # Keep order? 
    # Let's sort by the sum of the primes, or just keep the generated order which is lexicographic
    # Lexicographic order of combinations from [2,3,5,7...] is:
    # (2,3), (2,5), (2,7)... (3,5)...
    # This prioritizes 2,3 which is good.
    
    best_result = list(range(1, 13)) # Fallback
    
    for primes in candidates:
        if time.time() > deadline:
            break
            
        # Set a sub-deadline for this specific prime set
        sub_deadline = time.time() + 5.0
        
        max_sum = 20000
        # Precompute smoothness
        is_smooth_arr = [False] * (max_sum + 1)
        for i in range(2, max_sum + 1):
            if _is_smooth(i, primes):
                is_smooth_arr[i] = True
        
        # Backtracking search
        found_set = None
        
        def backtrack(current_set, start_idx):
            nonlocal found_set, sub_deadline
            if found_set:
                return
            if time.time() > sub_deadline:
                return
                
            if len(current_set) == 12:
                found_set = current_set[:]
                return
            
            # Optimization: 
            # If we have selected k elements, we need 12-k more.
            # We can prune if the remaining candidates are insufficient? Hard to know.
            
            for x in range(start_idx, 10001):
                # Check if x is compatible with all in current_set
                valid = True
                for y in current_set:
                    if not is_smooth_arr[x + y]:
                        valid = False
                        break
                if valid:
                    backtrack(current_set + [x], x + 1)
                    if found_set:
                        return
            
        backtrack([], 1)
        
        if found_set:
            best_result = found_set
            # Since we iterate from smallest prime sets, the first found is optimal
            # regarding the number of distinct primes.
            # Wait, is it? 
            # The objective is to minimize the number of distinct primes in the UNION.
            # If we force the sums to be P-smooth, the union is a subset of P.
            # So the number of distinct primes is <= |P|.
            # If we find a set for P, the actual union might be smaller than P?
            # Yes, but we are minimizing the union.
            # If we find a set for P of size 3, the union has <= 3 primes.
            # If we then try P of size 4, we might find a set with union size 2?
            # No, if the union size was 2, it would have been found when we tried the specific set of 2 primes.
            # So iterating by |P| and finding a valid set for P implies the min union size is <= |P|.
            # The first successful P gives the minimum possible union size?
            # Not necessarily. It gives the minimum |P| such that a set exists.
            # But the union of the found set might be smaller than |P|?
            # Example: P={2,3,5}. We find a set where all sums are only divisible by 2 and 3.
            # Then the union is {2,3}, size 2.
            # But we would have tried P={2,3} first. If a set existed for {2,3}, we would have found it then.
            # So if we didn't find a set for {2,3}, it means NO set exists where all sums are 2,3-smooth.
            # So any set we find later must have sums that are NOT all 2,3-smooth.
            # Thus the union must contain a prime > 3.
            # So the union size will be > 2.
            # Therefore, the first successful prime set P gives a solution with union size exactly |P|?
            # Or at least, it gives the minimum possible union size.
            # Yes, because if a solution with union size k existed, it would be smooth for the set of those k primes, which we would have tried before larger sets.
            
            return best_result
        
    return best_result
