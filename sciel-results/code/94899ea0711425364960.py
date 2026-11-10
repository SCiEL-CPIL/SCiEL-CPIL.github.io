import math
from functools import lru_cache

def construct():
    N = math.factorial(12)
    
    # Generate all divisors of N
    divisors = []
    for i in range(1, int(math.isqrt(N)) + 1):
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
    
    # Filter divisors > 12
    divisors = [d for d in divisors if d > 12]
    divisors.sort()
    
    # Backtracking search function
    def find_subset(candidates, target):
        n = len(candidates)
        
        def dfs(start_idx, remaining_target, chosen):
            if remaining_target == 1:
                if 1 <= len(chosen) <= 12:
                    return chosen
                else:
                    return None
            
            if start_idx >= n:
                return None
            
            # Pruning: If remaining_target is 1, we handled it.
            # If remaining_target is prime, it must be in the remaining candidates.
            # Simple check: if remaining_target > candidates[-1], it might be impossible 
            # unless we can form it with multiple factors? No, if it's prime, it must be a factor.
            # But we don't know if it's prime without checking.
            
            for i in range(start_idx, n):
                d = candidates[i]
                
                # If d > remaining_target, break (since sorted ascending)
                if d > remaining_target:
                    break
                
                # Divisibility check
                if remaining_target % d != 0:
                    continue
                
                # Additional pruning: 
                # If we pick d, the new remaining target is remaining_target // d.
                # We must be able to form this with elements at indices > i.
                # If remaining_target // d > 1, we need more factors.
                # If remaining_target // d is larger than the product of ALL remaining candidates, it's impossible.
                # But calculating product of all remaining is O(n).
                # We can precompute suffix products? 
                # suffix_prod[i] = product of candidates[i:]
                # However, we are skipping elements. The max product we can form with a subset of candidates[i:] is the product of ALL of them.
                # So if remaining_target // d > suffix_prod[i+1], it's impossible.
                
                # Let's implement suffix products for pruning
                # We need to pass suffix_prod or calculate it.
                # Since n is small (<=720), and depth is small, we can calculate on the fly? No.
                # Let's precompute suffix products for the current candidates list outside dfs?
                # But candidates changes with M.
                # We can compute suffix_prod once per M.
                
                res = dfs(i + 1, remaining_target // d, chosen + [d])
                if res is not None:
                    return res
            
            return None
        
        # Precompute suffix products for pruning
        # suffix_prod[i] = product of candidates[i:]
        # If product overflows or becomes too large, cap it at target
        suffix_prod = [1] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_prod[i] = suffix_prod[i + 1] * candidates[i]
            if suffix_prod[i] > target:
                suffix_prod[i] = target + 1 # Cap it
        
        def dfs_optimized(start_idx, remaining_target, chosen):
            if remaining_target == 1:
                if 1 <= len(chosen) <= 12:
                    return chosen
                else:
                    return None
            
            if start_idx >= n:
                return None
            
            # Pruning: If remaining_target > suffix_prod[start_idx], impossible
            if remaining_target > suffix_prod[start_idx]:
                return None
            
            for i in range(start_idx, n):
                d = candidates[i]
                
                if d > remaining_target:
                    break
                
                if remaining_target % d != 0:
                    continue
                
                new_target = remaining_target // d
                
                # Pruning: If new_target > suffix_prod[i+1], impossible
                if new_target > suffix_prod[i + 1]:
                    continue
                
                res = dfs_optimized(i + 1, new_target, chosen + [d])
                if res is not None:
                    return res
            
            return None
        
        return dfs_optimized(0, target, [])

    # Iterate through possible maximum values
    for M in divisors:
        candidates = [d for d in divisors if d <= M]
        
        # Quick check: is it possible?
        prod_all = 1
        for d in candidates:
            prod_all *= d
            if prod_all > N:
                break
        if prod_all < N:
            continue
            
        sol = find_subset(candidates, N)
        if sol:
            return sol
            
    return [N]
