import time

def construct():
    start_time = time.time()
    
    # First 15 primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    def is_smooth(x, prime_set):
        for p in prime_set:
            while x % p == 0:
                x //= p
        return x == 1

    # Fallback: 1..12
    best_result = list(range(1, 13))
    
    # Try increasing number of primes
    for k in range(3, 12):
        if time.time() - start_time > 85:
            break
            
        S = primes[:k]
        
        # Search in expanding ranges
        max_val = 100
        
        while max_val <= 10000:
            if time.time() - start_time > 85:
                break
            
            # Precompute smoothness for sums up to 2*max_val
            limit_sum = 2 * max_val
            smooth = [False] * (limit_sum + 1)
            for s in range(2, limit_sum + 1):
                if is_smooth(s, S):
                    smooth[s] = True
            
            # Backtracking search for 12 distinct integers
            # a[0] < a[1] < ... < a[11]
            
            found = None
            
            def backtrack(start_idx, current_set):
                if len(current_set) == 12:
                    return current_set[:]
                
                for next_val in range(start_idx, max_val + 1):
                    valid = True
                    for prev_val in current_set:
                        if not smooth[prev_val + next_val]:
                            valid = False
                            break
                    if valid:
                        current_set.append(next_val)
                        res = backtrack(next_val + 1, current_set)
                        if res:
                            return res
                        current_set.pop()
                return None

            # Try starting with small a0
            for a0 in range(1, min(50, max_val) + 1):
                if time.time() - start_time > 85:
                    return best_result
                res = backtrack(a0 + 1, [a0])
                if res:
                    found = res
                    break
            
            if found:
                best_result = found
                return found
            
            # Expand range
            max_val = min(max_val * 2, 10000)

    return best_result
