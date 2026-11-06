from fractions import Fraction
import time

def construct():
    # Try to find a minimal solution (k=8)
    start_time = time.time()
    
    def find_solution(K, max_time=10):
        target = Fraction(1, 1)
        
        def dfs(rem, prev_d, depth, path):
            if time.time() - start_time > max_time:
                return None
            if depth == K:
                if rem == 0:
                    return path[:]
                return None
            if rem < 0:
                return None
                
            m = K - depth + 1
            # d >= 1/rem
            d_low = max(prev_d + 1, rem.denominator // rem.numerator + (1 if rem.denominator % rem.numerator != 0 else 0))
            
            # d <= m/rem (approx)
            # To be safe, we don't set a hard d_high, but rely on break condition.
            # However, we need an upper bound to avoid infinite loop if break doesn't trigger?
            # Break triggers when max_add < new_rem.
            # max_add decreases with d.
            # So it will trigger eventually.
            
            # But we should cap d to 10^7
            d_high = 10000000
            
            for d in range(d_low, d_high + 1):
                new_rem = rem - Fraction(1, d)
                if new_rem < 0:
                    continue
                
                if m == 1:
                    if new_rem == 0:
                        return path + [d]
                else:
                    # Check if max possible sum of remaining m-1 terms is sufficient
                    # Max sum is sum of 1/(d+1) to 1/(d+m-1)
                    max_add = 0
                    for j in range(1, m):
                        max_add += Fraction(1, d+j)
                    
                    if max_add < new_rem:
                        # Cannot reach target with this d or any larger d
                        break
                    
                    res = dfs(new_rem, d, depth + 1, path + [d])
                    if res:
                        return res
            return None

        return dfs(target, 4, 1, [])

    res = find_solution(8)
    if res:
        return res

    # Fallback to seed logic if k=8 fails or times out
    # Seed logic: Subset DP on divisors of 2520
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
