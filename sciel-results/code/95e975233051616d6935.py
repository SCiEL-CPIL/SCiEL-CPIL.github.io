# EVOLVE-BLOCK-START
import math

def construct():
    # We established that the minimum number of terms is 8.
    # 1 = sum of 1/d_i, 5 <= d_1 < d_2 < ... < d_8 <= 10000000.
    
    def solve(n, d, min_denom, terms_left):
        """
        Find 'terms_left' distinct integers > min_denom such that
        sum(1/x_i) = n/d.
        Returns a list of denominators or None.
        """
        if terms_left == 0:
            if n == 0:
                return []
            return None
        
        if n == 0:
            return []
            
        # Lower bound for next denominator
        # 1/d_next <= n/d  => d_next >= d/n
        # Also d_next > min_denom
        # So d_next >= max(min_denom + 1, ceil(d/n))
        lower_bound = max(min_denom + 1, (d + n - 1) // n)
        
        # Upper bound logic:
        # We need sum of 'terms_left' reciprocals to be n/d.
        # The largest possible sum for 'terms_left' terms starting from d_next+1 is
        # S_max = sum(1/(d_next + j) for j in 1..terms_left)
        # If n/d - 1/d_next > S_max, then it's impossible.
        # As d_next increases, LHS increases and RHS decreases.
        # So if condition fails, we can break.
        
        # Also, n/d - 1/d_next must be > 0.
        # 1/d_next < n/d => d_next > d/n. This is covered by lower_bound if we use ceil.
        # Actually if d/n is integer, 1/(d/n) = n/d, so remaining is 0.
        # If terms_left > 1, remaining must be > 0.
        # If terms_left == 1, remaining must be 0.
        
        # Let's just iterate d_next from lower_bound upwards.
        # To avoid infinite loops or huge ranges, we can estimate an upper bound.
        # n/d - 1/d_next <= terms_left / (d_next + 1)
        # This gives a loose bound.
        
        # A better upper bound:
        # The smallest term is 1/d_last.
        # n/d = 1/d_next + ... + 1/d_last
        # n/d < terms_left / d_next
        # d_next < terms_left * d / n
        upper_bound = (terms_left * d) // n + 1
        
        # We should not exceed 10000000
        upper_bound = min(upper_bound, 10000000)
        
        # If lower_bound > upper_bound, no solution
        if lower_bound > upper_bound:
            return None
            
        for d_next in range(lower_bound, upper_bound + 1):
            # Check if remaining fraction n/d - 1/d_next can be formed by terms_left-1 terms
            # remaining = n/d - 1/d_next = (n*d_next - d) / (d*d_next)
            rem_num = n * d_next - d
            rem_den = d * d_next
            
            if rem_num < 0:
                # 1/d_next > n/d, so we've subtracted too much.
                # Since d_next is increasing, 1/d_next is decreasing, so rem_num will increase.
                # Wait, if rem_num < 0, then 1/d_next > n/d.
                # We need sum of remaining terms to be negative? Impossible.
                # But as d_next increases, 1/d_next decreases, so rem_num increases.
                # So we should continue?
                # No, if 1/d_next > n/d, then n/d - 1/d_next < 0.
                # We cannot represent a negative number with positive unit fractions.
                # So this d_next is invalid.
                # However, for larger d_next, 1/d_next is smaller, so rem_num might become positive.
                # So we just skip this d_next.
                continue
                
            if rem_num == 0:
                if terms_left == 1:
                    return [d_next]
                else:
                    # We have used 1 term, but we have terms_left > 1 left to use?
                    # No, the function signature is "find 'terms_left' terms".
                    # If we pick d_next, we have used 1 term. We need terms_left-1 more.
                    # If rem_num == 0, we need terms_left-1 terms to sum to 0.
                    # This is only possible if terms_left-1 == 0.
                    # So if terms_left > 1, this is a dead end.
                    continue
            
            # Pruning: Check if rem_num/rem_den can be represented by terms_left-1 terms.
            # Max sum of terms_left-1 terms starting from d_next+1:
            # S_max = sum(1/(d_next + j) for j in 1..terms_left-1)
            # If rem_num/rem_den > S_max, then it's impossible.
            # Calculate S_max approximately or exactly for small terms_left.
            # terms_left is small (<= 8).
            
            # Exact check for S_max:
            # We need rem_num/rem_den <= S_max
            # rem_num * (denominator of S_max) <= rem_den * (numerator of S_max)
            
            # Let's compute S_max exactly using fractions? No, integers.
            # S_max = sum(1/(d_next+j)).
            # To avoid huge numbers, we can check iteratively or use a loose bound.
            # Loose bound: S_max < (terms_left-1) / (d_next+1)
            # If rem_num/rem_den > (terms_left-1)/(d_next+1), then impossible.
            # rem_num * (d_next+1) > rem_den * (terms_left-1)
            
            if rem_num * (d_next + 1) > rem_den * (terms_left - 1):
                # The required remaining sum is larger than the maximum possible sum
                # of the remaining terms.
                # As d_next increases, rem_num/rem_den increases (LHS of original eq)
                # and (terms_left-1)/(d_next+1) decreases.
                # So if it's too big now, it will be too big for all larger d_next.
                break
                
            # Recurse
            res = solve(rem_num, rem_den, d_next, terms_left - 1)
            if res is not None:
                return [d_next] + res
                
        return None

    # Start with target 1/1, min denominator 4 (so first is >= 5), 8 terms.
    # We know 7 terms is impossible (max sum < 1).
    # We know 8 terms is possible.
    
    # Try 8 terms first
    result = solve(1, 1, 4, 8)
    
    if result is not None:
        # Verify
        total = 0
        for d in result:
            total += 1.0 / d
        # Due to floating point, check with Fraction for exactness in verification
        from fractions import Fraction
        exact_sum = sum(Fraction(1, d) for d in result)
        if exact_sum == 1 and len(set(result)) == 8 and all(5 <= d <= 10000000 for d in result):
            return result
            
    # Fallback: Try 9 terms if 8 failed (shouldn't happen)
    result = solve(1, 1, 4, 9)
    if result is not None:
        from fractions import Fraction
        exact_sum = sum(Fraction(1, d) for d in result)
        if exact_sum == 1 and len(set(result)) == 9 and all(5 <= d <= 10000000 for d in result):
            return result
            
    raise RuntimeError("No solution found")
# EVOLVE-BLOCK-END
