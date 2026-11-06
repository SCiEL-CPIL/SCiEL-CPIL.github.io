def construct():
       from fractions import Fraction
       import time
       
       target = Fraction(5, 31)
       
       # Precompute for a reasonable max D, say 500
       # But we want the MINIMUM D.
       # So we iterate D from 1 upwards.
       
       # Optimization: We can just run the search for a fixed D,
       # but we need to find the smallest D.
       # The search for small D is very fast.
       
       def search(D):
           # Precompute suffix sums for 1..D
           # suffix[i] = sum_{j=i}^D 1/j
           suffix = [Fraction(0,1)] * (D + 2)
           for i in range(D, 0, -1):
               suffix[i] = suffix[i+1] + Fraction(1, i)
           
           # Backtracking
           # State: current_sum, last_d
           # We want to find a path to target.
           
           # Use a stack or recursion. Recursion is fine for small depth.
           # Depth is at most D, but practically much smaller.
           
           path = []
           
           def backtrack(current_sum, last_d):
               if current_sum == target:
                   return True
               if last_d >= D:
                   return False
               
               R = target - current_sum
               if R <= 0:
                   return False
               
               # Pruning: max possible addition
               if suffix[last_d + 1] < R:
                   return False
               
               # Start d from max(last_d+1, ceil(1/R))
               start_d = last_d + 1
               # 1/d <= R => d >= 1/R
               min_d = (R.denominator + R.numerator - 1) // R.numerator
               if min_d > start_d:
                   start_d = min_d
               
               for d in range(start_d, D + 1):
                   next_sum = current_sum + Fraction(1, d)
                   if next_sum > target:
                       continue # 1/d is too big, but d increases, so 1/d decreases.
                       # Wait. If 1/d > R, then next_sum > target.
                       # Since d is increasing, 1/d is decreasing.
                       # So for larger d, 1/d might be <= R.
                       # So we should NOT break, but continue?
                       # No, if start_d is chosen as ceil(1/R), then 1/start_d <= R.
                       # So 1/d <= R for all d >= start_d.
                       # So next_sum <= target.
                   
                   # Check if remainder can be formed
                   rem = target - next_sum
                   if rem == 0:
                       path.append(d)
                       return True
                   
                   if rem < 0:
                       continue
                   
                   # Prune: can we form rem with denominators > d?
                   if suffix[d + 1] < rem:
                       continue
                   
                   path.append(d)
                   if backtrack(next_sum, d):
                       return True
                   path.pop()
               
               return False
           
           if backtrack(Fraction(0,1), 0):
               return path[:]
           return None
       
       # Iterate D
       # To avoid slow start, maybe jump?
       # But D is likely small.
       
       # Safety: time limit
       start_time = time.time()
       
       # We know seed has D=1736.
       # Let's try up to 500 first.
       for D in range(10, 501):
           if time.time() - start_time > 50:
               break
           res = search(D)
           if res:
               return res
       
       # Fallback to seed if not found
       r, values = Fraction(5, 31) - Fraction(1, 8), [8]
       while r:
           d = (r.denominator + r.numerator - 1) // r.numerator
           values.append(d)
           r -= Fraction(1, d)
       return values
