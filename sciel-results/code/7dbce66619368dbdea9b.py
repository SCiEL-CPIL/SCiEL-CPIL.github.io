from fractions import Fraction
import time

def construct():
        target = Fraction(5, 31)
        start_time = time.time()
        deadline = start_time + 50  # 50s safety margin
        
        # Try to find a solution with max denominator M
        # Start with small M and increase
        # We know seed has M=1736.
        # We want to beat that.
        
        # Helper to solve for a given M
        def solve_with_max_m(m_max, max_depth=10):
            # DFS
            # state: rem, last_d, path
            def dfs(rem, last_d, path):
                if time.time() > deadline:
                    return None
                if rem == 0:
                    return path
                if len(path) >= max_depth:
                    return None
                
                # min_d
                # 1/d <= rem => d >= 1/rem
                min_d_val = (rem.denominator + rem.numerator - 1) // rem.numerator
                min_d = max(last_d + 1, min_d_val)
                
                if min_d > m_max:
                    return None
                
                # Optional pruning: 
                # Max sum possible with remaining depth?
                # Not easy to compute exactly.
                
                for d in range(min_d, m_max + 1):
                    if 1 / d > rem:
                        continue
                    
                    new_rem = rem - Fraction(1, d)
                    res = dfs(new_rem, d, path + [d])
                    if res is not None:
                        return res
                return None
            
            return dfs(target, 1, [])

        # Iterate M
        # Start from a reasonable lower bound.
        # 1/6 > 5/31, so d1 >= 7.
        # If we use 2 terms: 1/7 + 1/d2 = 5/31 => 1/d2 = 4/217 => d2 = 54.25. No integer.
        # If we use 3 terms: 1/7 + 1/55 + 1/d3. d3 ~ 4000.
        # So M must be large if we stick to 3 terms with small d1.
        # To get small M, we likely need more terms.
        
        # Let's try M from 10 to 1000
        best_solution = None
        for m in range(10, 1001):
            if time.time() > deadline:
                break
            sol = solve_with_max_m(m, max_depth=8)
            if sol:
                best_solution = sol
                break
        
        if best_solution:
            return best_solution
        
        # Fallback to seed-like greedy
        r, values = Fraction(5, 31) - Fraction(1, 8), [8]
        while r:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        return values
