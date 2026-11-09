from fractions import Fraction

def construct():
    target = Fraction(5, 31)
    
    # We search for the minimum M such that there exists a subset of 
    # {ceil(1/target), ..., M} summing to target.
    
    M = 7
    # Fallback solution if search takes too long or fails
    fallback = [8, 28, 1736]
    
    while M <= 2000: # Limit search space to ensure speed
        # Precompute suffix sums of reciprocals for 1..M
        # suf[i] = sum(1/j for j in i..M)
        suf = [Fraction(0, 1)] * (M + 2)
        for i in range(M, 0, -1):
            suf[i] = suf[i + 1] + Fraction(1, i)
            
        start_d = (target.denominator + target.numerator - 1) // target.numerator
        
        def dfs(rem, min_d, current):
            if rem == 0:
                return current[:]
            if len(current) >= 256:
                return None
            
            # Pruning: max possible sum from min_d to M
            if suf[min_d] < rem:
                return None
                
            # Determine range for next d
            # 1/d <= rem => d >= 1/rem
            lower = max(min_d, (rem.denominator + rem.numerator - 1) // rem.numerator)
            if lower > M:
                return None
            
            # Check if even taking all from lower to M is enough
            if suf[lower] < rem:
                return None
                
            for d in range(lower, M + 1):
                new_rem = rem - Fraction(1, d)
                if new_rem < 0:
                    continue
                res = dfs(new_rem, d + 1, current + [d])
                if res:
                    return res
            return None

        sol = dfs(target, start_d, [])
        if sol:
            return sol
        M += 1
        
    return fallback
