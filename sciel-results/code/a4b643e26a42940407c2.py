from fractions import Fraction
import time

def construct():
       start_time = time.time()
       deadline = start_time + 55 # Safety margin

       # Known 10-term solution
       fallback = [5, 6, 7, 8, 9, 10, 11, 17, 216, 353430]

       # Try to find a solution with fewer terms
       # We assume WLOG d1 < d2 < ... < dk
       
       def dfs(remaining, min_d, k, path):
           if time.time() > deadline:
               return None
           
           if k == 0:
               if remaining == 0:
                   return path
               return None
           
           if remaining <= 0:
               return None
           
           # Max sum of k terms starting from min_d
           # Sum_{i=0}^{k-1} 1/(min_d + i)
           # If this is < remaining, impossible
           max_sum = sum(Fraction(1, min_d + i) for i in range(k))
           if max_sum < remaining:
               return None
           
           # Next denominator d
           # 1/d <= remaining => d >= ceil(1/remaining)
           d_min = max(min_d, (1 // remaining.numerator) * remaining.denominator + (1 if 1 % remaining.numerator else 0))
           # Actually simpler: d_min = max(min_d, -(-1 // remaining.numerator) * remaining.denominator) ?
           # ceil(1/r) where r = p/q is ceil(q/p)
           p, q = remaining.numerator, remaining.denominator
           d_min_val = q // p
           if q % p != 0:
               d_min_val += 1
           d_min = max(min_d, d_min_val)
           
           # Upper bound for d
           # We need remaining - 1/d to be formable by k-1 terms starting from d+1.
           # Max sum of k-1 terms starting from d+1 is M.
           # So remaining - 1/d <= M
           # 1/d >= remaining - M
           # If remaining - M > 0, then d <= 1 / (remaining - M)
           # If remaining - M <= 0, no upper bound from this constraint (d can be large)
           
           # Iterate d from d_min upwards
           # How far up?
           # If k==1, d must be exactly 1/remaining
           if k == 1:
               if remaining.denominator % remaining.numerator == 0:
                   d = remaining.denominator // remaining.numerator
                   if d >= min_d:
                       return path + [d]
               return None
           
           # For k > 1, we can iterate d.
           # Limit d to avoid infinite loop.
           # If d is too large, 1/d is very small, remaining - 1/d is close to remaining.
           # But max_sum for next step will be smaller.
           # Let's just iterate d up to a reasonable limit or until time runs out.
           # A good upper bound: 
           # If we pick d, the next min_d is d+1.
           # The max sum of k-1 terms from d+1 is S_next_max.
           # We need remaining - 1/d <= S_next_max.
           # If 1/d > remaining - S_next_max, it's valid.
           # If 1/d < remaining - S_next_max, it's invalid (sum too small).
           # So d > 1 / (remaining - S_next_max) is invalid?
           # No, if 1/d is too small, the sum of the rest must be larger, but the max sum of the rest is fixed.
           # So if 1/d < remaining - S_next_max, then remaining - 1/d > S_next_max, so impossible.
           # Thus we need 1/d >= remaining - S_next_max.
           # So d <= 1 / (remaining - S_next_max).
           
           # Calculate S_next_max for a given d?
           # S_next_max depends on d.
           # Let's just iterate d.
           
           # Optimization: if k is small, we can bound d better.
           # For now, iterate d from d_min to some limit.
           # Limit: if d > 10000000, break.
           # Also, if d is very large, 1/d is tiny.
           
           # Heuristic: if k is small, d won't be too huge.
           # Let's try up to 1000 or so initially? No, denominators can be large.
           # But for k=8, the denominators are likely not astronomically large unless the remainder is tiny.
           
           # Let's try iterating d.
           # To prevent timeout, we can limit the number of iterations or use a tighter bound.
           
           # Upper bound for d:
           # If remaining - 1/d > 0, and we need to form it with k-1 terms.
           # The smallest possible sum for k-1 terms starting from d+1 is not useful for upper bound of d.
           # The largest possible sum is S_max(d+1).
           # So we need remaining - 1/d <= S_max(d+1).
           # 1/d >= remaining - S_max(d+1).
           # If remaining <= S_max(d+1), this is always true for positive 1/d.
           # If remaining > S_max(d+1), then d <= 1 / (remaining - S_max(d+1)).
           
           # Since S_max(d+1) decreases as d increases, the bound on d changes.
           # Let's just loop d.
           
           # To make it faster, we can step d.
           # But we must check all d.
           
           # Let's limit d to 10000 for the first few levels to keep it fast?
           # No, that might miss solutions.
           
           # Alternative: Use the property that for small k, d is bounded.
           # For k=2: 1/a + 1/b = r. b = a / (ar - 1).
           # For k=3: 1/a + 1/b + 1/c = r.
           
           # Let's just run DFS with time check.
           
           for d in range(d_min, 10000001):
               if time.time() > deadline:
                   return None
               
               # Check upper bound constraint
               # S_max for k-1 terms starting from d+1
               # Approximation: (k-1) / (d+1)
               # If remaining - 1/d > (k-1)/(d+1) + epsilon, then impossible?
               # Actually, S_max = sum_{i=1}^{k-1} 1/(d+i)
               # If k-1 is small, we can compute it.
               if k - 1 <= 5:
                   s_max_next = sum(Fraction(1, d + i) for i in range(1, k))
                   if remaining - Fraction(1, d) > s_max_next:
                       continue
                   # Also if remaining - Fraction(1, d) < 0, break (since 1/d is decreasing, remainder increases? No, remainder = r - 1/d. As d increases, 1/d decreases, remainder increases. Wait.
                   # r is fixed. d increases -> 1/d decreases -> r - 1/d increases.
                   # So if r - 1/d > s_max_next, it will only get worse (larger) as d increases.
                   # So we can break!
                   if remaining - Fraction(1, d) > s_max_next:
                       break
               
               new_rem = remaining - Fraction(1, d)
               if new_rem < 0:
                   break # d is too small? No, d_min ensures 1/d <= remaining.
                   # Wait, d_min = ceil(1/r). So 1/d <= r. So new_rem >= 0.
                   
               res = dfs(new_rem, d + 1, k - 1, path + [d])
               if res:
                   return res
           
           return None

       # Try k=8
       for k in [8, 9]:
           if time.time() > deadline - 5:
               break
           res = dfs(Fraction(1, 1), 5, k, [])
           if res:
               return res
       
       return fallback
