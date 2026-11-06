import time
import math

def construct():
    target_num = 5
    target_den = 31
    # Known solution from seed
    best_solution = [8, 28, 1736]
    best_max_denom = 1736
    
    start_time = time.time()
    time_limit = 55.0 # Safety margin
    
    gamma = 0.5772156649015329
    
    def harmonic_approx(n):
        if n <= 0:
            return 0
        return math.log(n) + gamma + 1.0/(2.0*n)

    def can_complete(rem_num, rem_den, min_denom, max_denom):
        """
        Check if it's possible to sum to rem_num/rem_den using distinct denominators 
        in (min_denom, max_denom].
        """
        if rem_num == 0:
            return True
        if rem_num < 0:
            return False
            
        if max_denom <= min_denom:
            return False
            
        # Max possible sum is sum(1/d for d in range(min_denom+1, max_denom+1))
        # Approx H_max - H_min
        max_sum = harmonic_approx(max_denom) - harmonic_approx(min_denom)
        
        # If even the max possible sum is less than remainder, impossible
        if max_sum < rem_num / rem_den:
            return False
            
        return True

    def find_solution_with_max(k):
        """
        Attempts to find a solution with all denominators <= k.
        Returns the list of denominators if found, else None.
        """
        
        def dfs_local(rem_num, rem_den, min_denom, current_set):
            if time.time() - start_time > time_limit:
                return None
                
            if rem_num == 0:
                return current_set
                
            if rem_num < 0:
                return None
                
            # Pruning
            if not can_complete(rem_num, rem_den, min_denom, k):
                return None
                
            lower_bound = (rem_den + rem_num - 1) // rem_num
            if lower_bound < min_denom + 1:
                lower_bound = min_denom + 1
                
            if lower_bound > k:
                return None
                
            for d in range(lower_bound, k + 1):
                if time.time() - start_time > time_limit:
                    return None
                    
                next_num = rem_num * d - rem_den
                next_den = rem_den * d
                
                if next_num < 0:
                    continue
                    
                if next_num == 0:
                    return current_set + [d]
                    
                # Pruning: Check if remaining sum is possible
                if k > d:
                    max_remaining_sum = harmonic_approx(k) - harmonic_approx(d)
                    if next_num / next_den > max_remaining_sum:
                        break
                else:
                    # No more denominators, but next_num != 0
                    # This branch is a dead end
                    continue

                res = dfs_local(next_num, next_den, d, current_set + [d])
                if res is not None:
                    return res
                    
            return None
            
        return dfs_local(target_num, target_den, 0, [])

    # Binary Search for minimum K
    low = 1
    high = best_max_denom
    
    while low <= high:
        if time.time() - start_time > time_limit - 2.0:
            break
            
        mid = (low + high) // 2
        
        sol = find_solution_with_max(mid)
        
        if sol is not None:
            max_in_sol = max(sol)
            if max_in_sol < best_max_denom:
                best_max_denom = max_in_sol
                best_solution = sol
            high = mid - 1
        else:
            if time.time() - start_time > time_limit - 5.0:
                break
            low = mid + 1
            
    return best_solution
