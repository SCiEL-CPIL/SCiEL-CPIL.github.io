import time
import math

def construct():
    """
    Finds a set of distinct integer denominators in [5, 10000000] 
    whose reciprocal sum is exactly 1, minimizing the count.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    # Target sum is 1
    target_num, target_den = 1, 1
    
    # We search for k = 5, 6, ...
    # Max denominator allowed
    MAX_D = 10000000
    
    def solve(k, min_d=5):
        """
        Recursively finds k denominators starting from min_d 
        that sum to target_num/target_den.
        Returns list of denominators or None.
        """
        if k == 0:
            if target_num == 0:
                return []
            return None
        
        if k == 1:
            if target_den % target_num == 0:
                d = target_den // target_num
                if d >= min_d and d <= MAX_D:
                    return [d]
            return None
        
        # Bound for max denominator in this step
        # Sum of remaining k terms must be target
        # 1/d_i <= target / k  => d_i >= k / target
        # Also 1/d_i < target => d_i > 1/target
        # And d_i >= min_d
        
        # Upper bound for d:
        # 1/d + (k-1)/d_next_min = target
        # If we pick d too large, we can't reach target with remaining terms.
        # Max possible sum with k terms starting from min_d is k/min_d.
        # Min possible sum with k terms starting from d is (k-1)/(d+1) + 1/d? No.
        
        # Simple bound:
        # 1/d <= target / k  => d >= k / target
        # Also 1/d < target => d > 1/target
        # Lower bound for d is min_d.
        
        # Upper bound:
        # We need 1/d + sum(1/d_i for i=1..k-1) = target
        # Max sum of remaining k-1 terms is (k-1) / (d+1) (if we could use d+1 repeatedly, but they must be distinct and increasing).
        # Actually, max sum of k-1 distinct integers >= d+1 is 1/(d+1) + 1/(d+2) + ...
        # This is hard to bound tightly quickly.
        # A loose upper bound:
        # 1/d + (k-1)/d = k/d <= target => d >= k/target
        # Also 1/d < target => d > 1/target.
        
        # Let's iterate d from min_d to some limit.
        # Limit: if 1/d + (k-1)/(d+1) < target, then d is too small? No, sum is too small.
        # If 1/d > target, d is too small.
        # If 1/d + (k-1)/d < target, then even with all terms equal to d, sum is too small.
        # So we need 1/d + (k-1)/d >= target => k/d >= target => d <= k/target.
        
        max_d = int(k / (target_num / target_den))
        # Also d must be such that we can form the sum.
        # Iterate d
        for d in range(min_d, max_d + 1):
            if time.time() > deadline:
                return None
            
            # Check if 1/d > target
            if d * target_num < target_den:
                continue
            
            # Check if remaining sum is possible
            # Rem = target - 1/d
            rem_num = target_num * d - target_den
            rem_den = target_den * d
            
            if rem_num <= 0:
                continue
                
            # Reduce fraction?
            # g = math.gcd(rem_num, rem_den)
            # rem_num //= g
            # rem_den //= g
            
            # We need k-1 terms summing to rem_num/rem_den, all > d.
            # Min possible sum for k-1 terms starting from d+1 is (k-1)/(d+1) ? No, 1/(d+1) + ...
            # Max possible sum is 1/(d+1) + 1/(d+2) + ...
            
            # Pruning:
            # If rem_num/rem_den > (k-1) / (d+1), it's impossible?
            # No, 1/(d+1) is the largest term.
            # Max sum of k-1 distinct terms > d is 1/(d+1) + 1/(d+2) + ... + 1/(d+k-1).
            # If rem > that sum, impossible.
            # If rem < (k-1) / (d+k-1)? No, min sum is 1/(d+k-1) + ...
            
            # Simple check:
            # If rem_num / rem_den > (k-1) / (d+1), then we need terms larger than 1/(d+1), which is impossible.
            if rem_num * (d + 1) > rem_den * (k - 1):
                continue
            
            # If rem_num / rem_den < (k-1) / (d + k - 1), then even the smallest possible terms sum to less than rem?
            # No, smallest terms are largest denominators.
            # Min sum of k-1 distinct terms > d is 1/(d+k-1) + ... + 1/(d+2k-2)?
            # This is getting complex. Let's just recurse and let the base case handle it.
            # But we should bound d to avoid infinite loops.
            
            res = solve(k - 1, d + 1)
            if res is not None:
                return [d] + res
                
        return None

    # Try k=5
    for k in range(5, 10):
        if time.time() > deadline:
            break
        res = solve(k)
        if res is not None:
            return res
            
    # Fallback: Greedy algorithm to find any valid solution (might be large k)
    # This ensures we return something valid even if search fails.
    def greedy():
        rem_num, rem_den = 1, 1
        res = []
        last_d = 4
        while rem_num > 0:
            # Next denominator d such that 1/d <= rem
            # d >= rem_den / rem_num
            d = rem_den // rem_num
            if d <= last_d:
                d = last_d + 1
            
            # Ensure d is within bounds
            if d > MAX_D:
                return None
            
            res.append(d)
            last_d = d
            
            # rem = rem - 1/d
            rem_num = rem_num * d - rem_den
            rem_den = rem_den * d
            
            # Reduce
            g = math.gcd(rem_num, rem_den)
            if g > 1:
                rem_num //= g
                rem_den //= g
                
            if len(res) > 256:
                return None
                
        return res

    res = greedy()
    if res is not None:
        return res
        
    raise RuntimeError("No solution found")

