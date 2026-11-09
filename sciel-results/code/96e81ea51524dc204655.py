import math

def construct():
       N = math.factorial(12)
       
       # Get all divisors
       divisors = []
       for i in range(1, int(math.isqrt(N)) + 1):
           if N % i == 0:
               divisors.append(i)
               if i != N // i:
                   divisors.append(N // i)
       
       # Filter divisors > 12
       divisors = [d for d in divisors if d > 12]
       divisors.sort()
       
       # We want to minimize the max factor.
       # Iterate M from 13 upwards.
       # For each M, consider divisors <= M.
       
       # To speed up, we can just iterate through the sorted divisors list and try to form the product.
       # But we need to ensure distinctness.
       
       # Let's use the DFS approach with a bound M.
       # We can binary search M? Or just linear scan.
       # Since the optimal M is likely small (around 30-50), linear scan is fine.
       
       def dfs(remaining, start_idx, max_val, current_factors):
           if remaining == 1:
               return current_factors
           
           # If remaining is less than the smallest available divisor, fail
           if start_idx >= len(divisors):
               return None
           if divisors[start_idx] > remaining:
               return None
           
           # Pruning: if the product of the largest possible factors is too small?
           # Or if remaining cannot be formed.
           
           # Iterate possible next factors
           for i in range(start_idx, len(divisors)):
               d = divisors[i]
               if d > max_val:
                   break
               if d > remaining:
                   break
               if remaining % d == 0:
                   # Recurse
                   result = dfs(remaining // d, i + 1, max_val, current_factors + [d])
                   if result is not None:
                       return result
               # Optimization: if remaining // d < next divisor, break?
               # The next divisor is divisors[i+1].
               # If remaining // d < divisors[i+1] (and i+1 < len), then no further factors can divide it?
               # Actually, if remaining // d is not 1, it must be composed of factors > d.
               # So if remaining // d < divisors[i+1], then it's impossible to form remaining//d with factors > d.
               if remaining // d != 1 and i + 1 < len(divisors):
                   if remaining // d < divisors[i+1]:
                       break
                       
           return None

       # Iterate M
       # Lower bound for M: The smallest factor must be > 12.
       # Let's start M from 13.
       # Upper bound: 12! itself.
       
       # To be safe with time, we can just run this.
       # The number of divisors is 792.
       # The depth is at most 12.
       # The branching factor is limited by the divisors.
       
       for M in range(13, N + 1):
           # Optimization: only consider M that are divisors of N?
           # No, the max factor doesn't have to be a divisor of N in the set, but the factors themselves are divisors.
           # However, the max factor *is* one of the factors, so it must be a divisor of N.
           # So we can iterate M over the divisors of N that are > 12.
           
           # Let's create a list of candidate M values.
           # Actually, the loop above iterates all integers.
           # It's better to iterate over the sorted divisors as potential M.
           
           pass

       # Let's restructure to iterate M over divisors.
       candidate_ms = [d for d in divisors]
       # divisors is already sorted.
       
       best_factors = None
       for M in candidate_ms:
           # We need to find a subset of divisors <= M that multiplies to N.
           # We can pass M to dfs.
           # But dfs uses the global `divisors` list.
           # We need to restrict the search to divisors <= M.
           
           # We can just check if d <= M in the loop.
           
           result = dfs(N, 0, M, [])
           if result is not None:
               return result
               
       # If no solution found (should not happen), return [N]
       return [N]
