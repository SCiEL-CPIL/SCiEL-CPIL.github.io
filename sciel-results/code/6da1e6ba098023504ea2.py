from fractions import Fraction

def construct():
    # We need to find 1..256 distinct integer denominators in [5, 10000000]
    # whose reciprocal sum is exactly 1, minimizing their count.
    
    # It is known that the minimum number of terms for an Egyptian fraction
    # of 1 with all denominators >= 5 is 8.
    # We will perform a depth-first search to find a solution with k=8 terms.
    # If that fails (or takes too long, though it shouldn't for k=8), we can try k=9, etc.
    # But k=8 is the target for the best score.
    
    # Search parameters
    target = Fraction(1, 1)
    min_denom = 5
    
    def search(rem, min_d, depth, path):
        if depth == 0:
            return path if rem == 0 else None
        
        if rem == 0:
            return path if depth == 0 else None # Must use exactly 'depth' terms? No, "minimize count". 
            # The problem says "Return 1..256 distinct integer denominators ... minimize their count".
            # So if we hit rem=0 early, that's a valid solution with fewer terms.
            # However, the minimum is 8. So we expect to find it at depth 8.
            # Let's allow early termination if rem becomes 0.
            return path
        
        # Bounds for the next denominator d
        # 1. 1/d <= rem  =>  d >= 1/rem
        # 2. d >= min_d
        # 3. We have 'depth' terms left. The maximum sum we can get with 'depth' terms
        #    starting from d is bounded. 
        #    Specifically, the largest possible sum for 'depth' terms with distinct denominators >= d
        #    is achieved by taking the smallest possible denominators: d, d+1, ..., d+depth-1.
        #    Let S_max = sum(1/(d+j) for j in 0..depth-1).
        #    If rem > S_max, then no solution exists for this d.
        #    Also, 1/d must be <= rem.
        
        d_min = max(min_d, (rem.denominator + rem.numerator - 1) // rem.numerator)
        
        # Upper bound for d:
        # If 1/d > rem, it's invalid. So d >= ceil(1/rem).
        # Also, we need rem - 1/d to be representable by depth-1 terms > d.
        # A loose upper bound: 
        # The smallest sum of depth-1 terms > d is approximately (depth-1)/(d+depth).
        # So rem - 1/d <= sum_{i=1}^{depth-1} 1/(d+i) < (depth-1)/(d+1).
        # This gives a constraint on how large d can be? No, this is a lower bound on the remainder.
        # Actually, if d is too large, 1/d is small, rem - 1/d is close to rem.
        # The remaining terms are even smaller.
        # The maximum sum of the remaining depth-1 terms is sum_{i=1}^{depth-1} 1/(d+i).
        # If rem - 1/d > sum_{i=1}^{depth-1} 1/(d+i), then it's impossible.
        # So we need: rem - 1/d <= sum_{i=1}^{depth-1} 1/(d+i)
        # This inequality helps prune large d?
        # Let's just iterate d from d_min upwards.
        
        # How large can d be?
        # If depth == 1:
        #   rem must be 1/d. So d = 1/rem.
        #   d must be integer and >= min_d.
        if depth == 1:
            if rem.denominator == 1:
                d = rem.numerator
                if d >= min_d:
                    return path + [d]
            elif rem.numerator == 1:
                d = rem.denominator
                if d >= min_d:
                    return path + [d]
            return None
        
        # Iterate d
        # Upper bound for d:
        # We need 1/d <= rem => d >= 1/rem
        # Also, the sum of the remaining depth-1 terms must be >= rem - 1/d.
        # Max sum of depth-1 terms with denominators > d is < (depth-1)/d.
        # So rem - 1/d < (depth-1)/d  =>  rem < depth/d  =>  d < depth/rem.
        # So d_max = floor(depth / rem) is a good upper bound.
        
        d_max = int(depth / rem)
        
        for d in range(d_min, d_max + 1):
            if Fraction(1, d) > rem:
                break # d is too small? No, if 1/d > rem, we need larger d.
                # Wait. d_min is calculated such that 1/d_min <= rem.
                # So for d >= d_min, 1/d <= rem.
                # The condition 1/d > rem would mean d < 1/rem.
                # Since we start at d_min >= ceil(1/rem), this shouldn't happen.
                
            new_rem = rem - Fraction(1, d)
            if new_rem < 0:
                continue
            if new_rem == 0:
                # We have used 'depth' terms? No, we used 1 term here, so depth-1 left.
                # If new_rem is 0, we have a solution with len(path)+1 terms.
                # But we are in a search for exactly 'depth' terms?
                # The problem asks to minimize count. 
                # If we find a solution with fewer terms, it's better.
                # But the recursion structure assumes we fill 'depth' slots.
                # If we hit 0 early, we should return the path so far.
                # However, the caller expects a list of length 'depth' added to path?
                # Let's adjust: if new_rem == 0, we can only return if depth == 1? 
                # No, if depth > 1 and we hit 0, we have a valid representation with fewer terms.
                # But we are searching for a specific depth k.
                # If we find a solution with k' < k, it's better.
                # But we are iterating k from 8 upwards.
                # So if we are in search for k=8, and we hit 0 at depth 5, that's a solution of size 5.
                # But we know min is 8. So this shouldn't happen.
                # If it does, it's a valid solution, so we return it.
                # But the interface expects a list.
                # Let's just return path + [d] if new_rem == 0 and we are done?
                # No, if new_rem == 0, we don't need more terms.
                # So we return path + [d].
                return path + [d]
            
            # Check feasibility for remaining depth-1 terms
            # Min denominator for next term is d+1
            # Max sum of depth-1 terms starting from d+1 is sum(1/(d+1+j) for j in 0..depth-2)
            # If new_rem > this max sum, we can break?
            # As d increases, 1/d decreases, new_rem increases.
            # The max sum of remaining terms decreases as d increases.
            # So if new_rem > max_sum at some d, it will be true for all larger d.
            # So we can break.
            
            # Calculate max_sum approx
            # max_sum < (depth-1) / (d+1)
            if new_rem > Fraction(depth - 1, d + 1):
                break
                
            res = search(new_rem, d + 1, depth - 1, path + [d])
            if res is not None:
                return res
                
        return None

    # Try k=8 first
    for k in range(8, 15):
        res = search(target, min_denom, k, [])
        if res is not None:
            # Verify solution
            if len(res) == len(set(res)):
                if all(5 <= d <= 10000000 for d in res):
                    if sum(Fraction(1, d) for d in res) == 1:
                        return res
    raise RuntimeError("No solution found")
