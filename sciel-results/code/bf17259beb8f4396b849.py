import math

def construct():
       target = math.factorial(12)
       
       # Generate divisors
       divisors = []
       for i in range(1, int(math.isqrt(target)) + 1):
           if target % i == 0:
               divisors.append(i)
               if i != target // i:
                   divisors.append(target // i)
               
       # Filter > 12
       valid_divs = sorted([d for d in divisors if d > 12])
       
       # We want to minimize the max factor.
       # Try M from 13 upwards.
       # For each M, check if a subset of valid_divs <= M multiplies to target.
       
       def can_form(target, allowed_divs):
           # allowed_divs is sorted list of divisors <= M and > 12
           # Use DFS
           n = len(allowed_divs)
           
           # Pruning: if product of all allowed < target, return False
           # if product of all allowed > target, it's possible but we need subset
           
           # Sort descending might help find large factors first?
           # Or ascending to prune small products?
           # Let's use ascending.
           
           def dfs(idx, current_prod, chosen):
               if current_prod == target:
                   return chosen
               if idx >= n:
                   return None
               
               # Pruning: 
               # Max possible product with remaining elements
               # This is expensive to compute. 
               # Instead, simple bound:
               # If current_prod * (max_allowed ** remaining_slots) < target, fail?
               # We don't know remaining slots.
               
               # Better pruning:
               # If current_prod > target, return None.
               if current_prod > target:
                   return None
               
               # Option 1: Skip allowed_divs[idx]
               res = dfs(idx + 1, current_prod, chosen)
               if res: return res
               
               # Option 2: Take allowed_divs[idx]
               if current_prod * allowed_divs[idx] <= target:
                   res = dfs(idx + 1, current_prod * allowed_divs[idx], chosen + [allowed_divs[idx]])
                   if res: return res
                   
               return None
           
           return dfs(0, 1, [])

       # Iterate M
       # Lower bound: 13
       # Upper bound: target (trivial solution [target] is not valid if target <= 12? No target > 12.
       # But [target] is 1 factor. Distinct? Yes. >12? Yes.
       # So M=target is always a solution.
       
       # To be safe and fast, we can just iterate M.
       # But valid_divs changes with M.
       
       # Optimization: 
       # The number of divisors is 720.
       # We can sort valid_divs.
       # We iterate M.
       # For each M, the allowed set is valid_divs[:k] where valid_divs[k] > M.
       
       # However, DFS on 700 items is too slow if M is large.
       # But we expect M to be small (< 100).
       # For M < 100, the number of divisors is small.
       
       for M in range(13, 1000):
           allowed = [d for d in valid_divs if d <= M]
           if not allowed:
               continue
           
           # Quick check: product of all allowed must be >= target
           # If product of all allowed < target, skip
           prod_all = 1
           for d in allowed:
               prod_all *= d
               if prod_all > target:
                   break
           if prod_all < target:
               continue
           
           # Also, we need at least 1 factor.
           # Run DFS
           # To speed up DFS, sort allowed in descending order?
           # If we pick large factors first, we reduce the remaining product faster.
           allowed_desc = sorted(allowed, reverse=True)
           
           def dfs_desc(idx, current_prod, chosen):
               if current_prod == target:
                   return chosen
               if idx >= len(allowed_desc):
                   return None
               
               # Pruning: if current_prod > target, fail
               if current_prod > target:
                   return None
               
               # Pruning: even if we take all remaining, can we reach target?
               # Compute suffix product?
               # Precompute suffix products of allowed_desc
               
               # Let's just use the simple DFS first, it should be fast for small M.
               
               # Option 1: Take
               if current_prod * allowed_desc[idx] <= target:
                   res = dfs_desc(idx + 1, current_prod * allowed_desc[idx], chosen + [allowed_desc[idx]])
                   if res: return res
               
               # Option 2: Skip
               res = dfs_desc(idx + 1, current_prod, chosen)
               if res: return res
               
               return None
           
           sol = dfs_desc(0, 1, [])
           if sol:
               return sol
               
       # Fallback
       return [target]
