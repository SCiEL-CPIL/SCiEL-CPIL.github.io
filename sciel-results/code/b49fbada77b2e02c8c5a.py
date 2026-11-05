import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    # Try N=8 first
    # If not found, try N=9, etc.
    # But N=8 is the theoretical minimum.
    
    best_solution = None
    
    # DFS parameters
    # deficit: Fraction, last_d: int, terms_left: int, path: list
    def dfs(deficit, last_d, terms_left, path):
        nonlocal best_solution, deadline
        
        if time.time() > deadline:
            return False
        
        if terms_left == 0:
            if deficit == 0:
                best_solution = path[:]
                return True
            return False
        
        if terms_left == 1:
            # deficit must be 1/d
            if deficit.denominator == 0: return False
            # deficit = p/q. We need 1/d = p/q => d = q/p
            p, q = deficit.numerator, deficit.denominator
            if p == 0: return False
            if q % p == 0:
                d = q // p
                if d > last_d:
                    best_solution = path + [d]
                    return True
            return False
        
        # Pruning:
        # Max possible sum with 'terms_left' terms starting from last_d+1
        # is sum(1/(last_d+1+i) for i in range(terms_left))
        # If this max sum < deficit, prune.
        max_sum = sum(Fraction(1, last_d + 1 + i) for i in range(terms_left))
        if max_sum < deficit:
            return False
        
        # Min possible sum with 'terms_left' terms?
        # Min sum is when denominators are as large as possible.
        # But denominators are bounded by 10^7.
        # Actually, we just need to ensure we don't pick d such that 1/d >= deficit (if terms_left > 1, 1/d must be < deficit? No, 1/d can be close to deficit, but remaining must be positive).
        # 1/d < deficit is required if terms_left > 1?
        # If 1/d == deficit, remaining is 0, but we have terms_left-1 terms left. So 1/d must be < deficit.
        # So d > 1/deficit.
        
        min_d = last_d + 1
        max_d = int(terms_left / deficit) # d <= terms_left / deficit
        # Also d <= 10^7
        
        if max_d > 10000000:
            max_d = 10000000
            
        # Iterate d from min_d to max_d
        # Optimization: Start from smallest d to maximize remaining sum?
        # Or largest d?
        # Smallest d gives largest 1/d, leaving smallest deficit for remaining terms.
        # This might be easier to satisfy?
        
        for d in range(min_d, max_d + 1):
            new_deficit = deficit - Fraction(1, d)
            if new_deficit < 0:
                break # Since d increases, 1/d decreases, deficit increases. 
                # Wait. If new_deficit < 0, it means 1/d > deficit.
                # We need 1/d < deficit.
                # So if 1/d >= deficit, we should break?
                # If 1/d == deficit, new_deficit = 0. 
                # If terms_left > 1, we cannot have 0 deficit with terms left.
                # So we require new_deficit > 0.
                # Thus 1/d < deficit.
                # So if 1/d >= deficit, break.
            
            if new_deficit > 0:
                if dfs(new_deficit, d, terms_left - 1, path + [d]):
                    return True
        
        return False
    
    # Try N=8
    if dfs(Fraction(1, 1), 4, 8, []):
        return best_solution
        
    # If N=8 fails or times out, try N=9
    # Reset deadline? No, just continue.
    # But best_solution is None if N=8 failed.
    
    if time.time() < deadline:
        if dfs(Fraction(1, 1), 4, 9, []):
            return best_solution
            
    if time.time() < deadline:
        if dfs(Fraction(1, 1), 4, 10, []):
            return best_solution
            
    # Fallback to baseline if all fail
    # Re-implement baseline logic quickly or just raise error?
    # Better to return a valid solution even if suboptimal.
    # The baseline solution is valid.
    # Let's try to reconstruct a simple valid solution quickly if DFS fails.
    # Greedy algorithm is fast.
    
    def greedy():
        rem = Fraction(1, 1)
        d = 4
        res = []
        while rem > 0:
            d += 1
            # Find smallest d > last_d such that 1/d <= rem
            # Actually, standard greedy picks smallest d > last_d such that 1/d <= rem?
            # No, smallest d such that 1/d <= rem is floor(1/rem) + 1?
            # If 1/d <= rem, d >= 1/rem.
            # We want distinct denominators.
            # Standard Fibonacci-Sylvester: d = floor(1/rem) + 1.
            # But we need d >= 5.
            # And d > last_d.
            
            d_candidate = int(1 / rem) + 1
            if d_candidate < 5:
                d_candidate = 5
            if d_candidate <= d:
                d_candidate = d + 1
            
            # Check if 1/d_candidate <= rem
            # If not, increment d_candidate
            while Fraction(1, d_candidate) > rem:
                d_candidate += 1
                
            res.append(d_candidate)
            rem -= Fraction(1, d_candidate)
            d = d_candidate
            
            if len(res) > 256:
                break
                
        return res
    
    return greedy()
