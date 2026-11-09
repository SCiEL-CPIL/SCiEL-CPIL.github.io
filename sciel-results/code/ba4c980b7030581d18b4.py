import math
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55 seconds to be safe within 60s limit
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def search(num, den, k, min_d, path):
        # Check time
        if time.time() > deadline:
            return None
        
        if k == 0:
            return path if num == 0 and den == 1 else None
            
        if k == 1:
            # We need num/den = 1/d
            # So d = den/num
            if den % num == 0:
                d = den // num
                if d >= min_d and d <= 10000000:
                    return path + [d]
            return None

        # Lower bound for d:
        # 1/d <= num/den => d >= den/num
        # Also d >= min_d
        d_min = max(min_d, math.ceil(den / num))
        
        # Upper bound for d:
        # num/den <= k/d => d <= k * den / num
        d_max = int((k * den) / num)
        
        # Also d must be <= 10000000
        if d_max > 10000000:
            d_max = 10000000
            
        if d_min > d_max:
            return None

        # Additional pruning: the remaining sum after choosing d must be achievable
        # with k-1 terms, each > d. The minimum sum with k-1 terms starting from d+1
        # is approximately (k-1)/(d+1) if all were d+1, but they must be distinct and increasing.
        # A tighter lower bound: sum of 1/(d+i) for i=1..k-1
        # But this is expensive to compute. Use a simpler bound.
        
        for d in range(d_min, d_max + 1):
            # Calculate remaining fraction: num/den - 1/d
            # = (num*d - den) / (den*d)
            n2 = num * d - den
            d2 = den * d
            
            # n2 must be positive
            if n2 <= 0:
                continue
            
            # Reduce fraction
            g = gcd(n2, d2)
            n2_reduced = n2 // g
            d2_reduced = d2 // g
            
            # Pruning: check if remaining fraction can be represented
            # with k-1 unit fractions all > d
            # The maximum sum with k-1 terms > d is (k-1)/(d+1)
            # The minimum sum is 1/(d+1) + 1/(d+2) + ... 
            # Simple check: n2_reduced/d2_reduced must be <= (k-1)/(d+1)
            if n2_reduced * (d + 1) > (k - 1) * d2_reduced:
                continue
            
            result = search(n2_reduced, d2_reduced, k - 1, d + 1, path + [d])
            if result is not None:
                return result
        return None

    best_solution = None
    # Try k from 4 upwards with time limits
    for k in range(4, 257): 
        # Check time
        if time.time() > deadline:
            break
            
        solution = search(1, 1, k, 5, [])
        if solution is not None:
            best_solution = solution
            break  # We found a solution with minimal k, so we can stop
            
    # If no solution found, fall back to a known valid solution
    if best_solution is None:
        # Known valid solution with all denominators >= 5
        # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/35 + 1/70 + 1/140
        # Let me verify: LCM of 5,6,7,10,14,35,70,140 = 420
        # 420/5 + 420/6 + 420/7 + 420/10 + 420/14 + 420/35 + 420/70 + 420/140
        # = 84 + 70 + 60 + 42 + 30 + 12 + 6 + 3 = 307. Not 420.
        
        # Let me find a proper one.
        # Use the identity: 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/35 + 1/70 + 1/140 + 1/280
        # This is getting complex. Let me use a simpler approach.
        
        # Actually, let's just return a hardcoded valid solution.
        # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/35 + 1/70 + 1/140
        # We computed 307/420. Need 113/420 more. Not a unit fraction.
        
        # Let me use: 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/35 + 1/70 + 1/140 + 1/280
        # LCM = 840
        # 840/5=168, 840/6=140, 840/7=120, 840/10=84, 840/14=60, 840/35=24, 840/70=12, 840/140=6, 840/280=3
        # Sum = 168+140+120+84+60+24+12+6+3 = 617. Not 840.
        
        # Let me just use a DP-based approach as fallback.
        target, reached = 2520, {0: []}
        for d in range(5, target + 1):
            if target % d:
                continue
            w = target // d
            for total, values in list(reached.items()):
                if total + w <= target and total + w not in reached:
                    reached[total + w] = values + [d]
            if target in reached:
                best_solution = reached[target]
                break
                
    return best_solution
# EVOLVE-BLOCK-END
