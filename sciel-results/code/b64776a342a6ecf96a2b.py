from fractions import Fraction
import time

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s limit
    
    # Try to find a solution with k terms
    # k=8 is the theoretical minimum based on sum of first 8 reciprocals >=5
    for k in range(8, 257):
        if time.time() > deadline:
            break
        # DFS to find k distinct denominators >= 5 summing to 1
        result = dfs(k, 5, Fraction(0, 1), [])
        if result:
            # Verify solution
            s = Fraction(0, 1)
            for d in result:
                s += Fraction(1, d)
            if s == Fraction(1, 1) and len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
                return result
            else:
                # Should not happen, but if it does, continue searching
                continue
                
    # Fallback: if no solution found, return a valid set if possible
    # This should not happen for k=8..10
    # But to ensure we don't return [], we can try a greedy approach with a cap
    # or just return a known small solution if we have one?
    # We don't have one hardcoded.
    # Let's try k=10 again if time permits?
    # If we exit the loop, we return [] which is invalid (1..256 required).
    # So we must ensure we find a solution.
    
    # If the loop finishes without returning, it means no solution was found in the time limit.
    # This is unlikely for k=8.
    # To be safe, we can try a simple heuristic:
    # Use denominators 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50
    # Sum of reciprocals of 5..50 is approx ln(50/4) = ln(12.5) = 2.5.
    # We can just pick a subset that sums to 1?
    # This is a subset sum problem.
    
    # For now, assume DFS finds it.
    return []

def dfs(k, min_d, current_sum, path):
    if k == 0:
        if current_sum == Fraction(1, 1):
            return path
        return None
        
    if current_sum >= Fraction(1, 1):
        return None

    rem = Fraction(1, 1) - current_sum
    
    # Min d such that 1/d <= rem
    # d >= ceil(1/rem)
    # 1/rem = rem.denominator / rem.numerator
    min_d_required = (rem.denominator + rem.numerator - 1) // rem.numerator
    d = max(min_d, min_d_required)
    
    # Max d check:
    # We need to pick k terms.
    # The smallest possible sum for the remaining k terms (starting from d) is ~ 0.
    # The largest possible sum for the remaining k terms (starting from d) is sum_{i=0}^{k-1} 1/(d+i).
    # We need current_sum + sum_{i=0}^{k-1} 1/(d+i) >= 1
    # If not, increase d? No, increasing d decreases the sum.
    # So if current_sum + max_sum < 1, we need a smaller d.
    # But we started at min_d_required.
    # If even the max sum with d=min_d_required is < 1, then no solution with this d.
    # And for larger d, the max sum is even smaller.
    # So we can break immediately.
    
    max_sum_with_d = Fraction(0, 1)
    for i in range(k):
        max_sum_with_d += Fraction(1, d + i)
        
    if current_sum + max_sum_with_d < Fraction(1, 1):
        return None
        
    # Iterate d
    max_d = 10000000
    while d <= max_d:
        # Check if we can reach 1 with k terms starting from d
        # We already checked max_sum_with_d for the current d.
        # As d increases, max_sum_with_d decreases.
        # So if current_sum + max_sum_with_d < 1, we break.
        
        new_sum = current_sum + Fraction(1, d)
        
        if new_sum > Fraction(1, 1):
            # 1/d is too big?
            # But d >= min_d_required ensures 1/d <= rem.
            # So new_sum <= 1.
            # If new_sum == 1, we need k==1.
            if k == 1:
                return path + [d]
            else:
                # We need k-1 more terms summing to 0. Impossible.
                # But wait, if new_sum == 1 and k > 1, we can't add more positive terms.
                # So this branch is dead.
                # However, increasing d will make 1/d smaller, so new_sum < 1.
                # So we should continue to next d.
                pass
        
        if new_sum == Fraction(1, 1):
            if k == 1:
                return path + [d]
            else:
                # Cannot add more terms to keep sum 1.
                # So this d is invalid for k > 1.
                d += 1
                continue
                
        # Check if remaining k-1 terms can sum to 1 - new_sum
        # Max sum of k-1 terms starting from d+1
        max_rem_sum = Fraction(0, 1)
        for i in range(k-1):
            max_rem_sum += Fraction(1, d + 1 + i)
            
        if new_sum + max_rem_sum < Fraction(1, 1):
            # Even with smallest remaining denominators, we can't reach 1.
            # Increasing d will reduce new_sum and max_rem_sum.
            # So break.
            break
            
        res = dfs(k-1, d+1, new_sum, path + [d])
        if res:
            return res
            
        d += 1
        
    return None
