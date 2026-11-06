from fractions import Fraction
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s safety margin
    
    def get_fallback():
        target, reached = 2520, {0: []}
        for d in range(5, target + 1):
            if target % d:
                continue
            w = target // d
            for total, values in list(reached.items()):
                if total + w <= target and total + w not in reached:
                    reached[total + w] = values + [d]
            if target in reached:
                return reached[target]
        raise RuntimeError("no baseline representation")
    
    # First, get the 12-term fallback quickly
    fallback = get_fallback()
    
    # Try to find a solution with fewer terms (8, 9, 10, 11)
    # 7 terms is impossible as max sum of 1/5..1/11 is < 1
    
    def try_find_solution(k, min_den=5, max_den=10000000):
        """Try to find k distinct denominators in [min_den, max_den] whose reciprocal sum is 1."""
        # Use integer arithmetic for speed
        # State: (num, den, min_d, k_left)
        # We want sum 1/1 = 1
        
        best = [None]
        visited = set()
        
        def backtrack(num, den, min_d, k_left, current):
            if time.time() > deadline:
                return
            
            # Normalize fraction
            if num == 0:
                if k_left == 0:
                    best[0] = current[:]
                return
            
            if k_left == 0:
                return
                
            # Check if state visited
            state = (num, den, min_d, k_left)
            if state in visited:
                return
            visited.add(state)
            
            # Pruning
            # Max sum with k_left terms from min_d is k_left / min_d
            # num/den > k_left / min_d  =>  num * min_d > den * k_left
            if num * min_d > den * k_left:
                return
                
            # Min sum with k_left terms is k_left / max_den
            # num/den < k_left / max_den => num * max_den < den * k_left
            if num * max_den < den * k_left:
                return
            
            if k_left == 1:
                # num/den = 1/d => d = den/num
                if den % num == 0:
                    d = den // num
                    if d >= min_d and d <= max_den and d not in current:
                        best[0] = current + [d]
                return
            
            # Determine range for next denominator d
            # 1/d <= num/den => d >= den/num
            d_low = max(min_d, den // num + 1)
            
            # Upper bound for d:
            # After picking d, remaining R' = num/den - 1/d = (d*num - den) / (d*den)
            # We need k_left - 1 terms >= d+1 to sum to R'
            # Max sum of k_left-1 terms >= d+1 is (k_left-1)/(d+1)
            # So (d*num - den) / (d*den) <= (k_left-1)/(d+1)
            # (d*num - den) * (d+1) <= (k_left-1) * d * den
            # d^2*num + d*num - d*den - den <= (k_left-1)*d*den + (k_left-1)*den
            # d^2*num + d*num - d*den - den - (k_left-1)*d*den - (k_left-1)*den <= 0
            # d^2*num + d*(num - den - (k_left-1)*den) - den*(1 + k_left - 1) <= 0
            # d^2*num + d*(num - k_left*den) - k_left*den <= 0
            
            # This is a quadratic inequality in d.
            # Let A = num, B = num - k_left*den, C = -k_left*den
            # A*d^2 + B*d + C <= 0
            # Roots are d = (-B +/- sqrt(B^2 - 4AC)) / 2A
            # Since A > 0, the inequality holds between the roots.
            # We only care about d > 0.
            
            A = num
            B = num - k_left * den
            C = -k_left * den
            
            disc = B*B - 4*A*C
            if disc < 0:
                return
            
            sqrt_disc = int(disc ** 0.5)
            # Upper root
            d_high = (-B + sqrt_disc) // (2 * A)
            
            d_high = min(d_high, max_den)
            
            if d_low > d_high:
                return
            
            # Iterate d
            # To improve speed, we can limit the search range or use heuristics
            # But with memoization, we should be able to handle the space.
            
            # Try d from d_low to d_high
            # Heuristic: try values closer to the "average" denominator first?
            # Average d approx k_left / (num/den) = k_left * den / num
            avg_d = k_left * den // num
            
            # Create a sorted list of candidates, prioritizing those near avg_d
            # Or just iterate. For k=8, the branching factor might be high.
            # Let's iterate from d_low upwards, but maybe skip some?
            # No, we need to be exact.
            
            # Optimization: If d_low is very small, and d_high is large, 
            # the loop can be slow.
            # However, for k=8, the denominators are likely small (5-20).
            # Let's just iterate.
            
            for d in range(d_low, d_high + 1):
                if d in current:
                    continue
                
                # Calculate new remaining fraction
                # R' = num/den - 1/d = (d*num - den) / (d*den)
                new_num = d * num - den
                new_den = d * den
                
                if new_num <= 0:
                    continue
                
                # Reduce fraction?
                # Not strictly necessary for correctness, but helps with state deduplication
                # if we normalize.
                # Let's normalize to keep numbers small.
                from math import gcd
                g = gcd(new_num, new_den)
                if g > 1:
                    new_num //= g
                    new_den //= g
                
                current.append(d)
                backtrack(new_num, new_den, d + 1, k_left - 1, current)
                current.pop()
                if best[0] is not None:
                    return
        
        backtrack(1, 1, min_den, k, [])
        return best[0]
    
    # Try k=8, 9, 10, 11
    # 7 is impossible
    for k in [8, 9, 10, 11]:
        if time.time() > deadline - 5:
            break
        sol = try_find_solution(k)
        if sol is not None:
            # Verify
            s = sum(Fraction(1, d) for d in sol)
            if s == 1 and len(set(sol)) == len(sol) and all(5 <= d <= 10000000 for d in sol):
                return sol
    
    return fallback
