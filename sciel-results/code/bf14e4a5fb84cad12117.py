   def construct():
       import time
       start = time.time()
       
       # Phase 1: Start with large numbers
       S = list(range(51, 101))
       
       def find_subset_summing_to(elements, target):
           # elements is a list
           # returns a list of elements summing to target, or None
           # Use DP to track one path
           # dp[s] = (prev_sum, element) is not enough to reconstruct easily if we just want ANY subset
           # Let's use: dp[s] = list of elements? No, memory.
           # dp[s] = (prev_s, x)
           prev = [None] * (target + 1)
           reachable = [False] * (target + 1)
           reachable[0] = True
           
           for x in elements:
               if x > target: continue
               # Iterate backwards to avoid using x multiple times
               for s in range(target, x - 1, -1):
                   if not reachable[s] and reachable[s - x]:
                       reachable[s] = True
                       prev[s] = (s - x, x)
           
           if not reachable[target]:
               return None
           
           # Backtrack
           subset = []
           s = target
           while s > 0:
               prev_s, x = prev[s]
               subset.append(x)
               s = prev_s
           return subset
       
       # Clean S
       while True:
           if time.time() - start > 80: break
           viol = find_subset_summing_to(S, 200)
           if viol is None:
               break
           # Remove one element from viol
           # Heuristic: Remove the largest element to keep smaller ones?
           # Or remove the one that appears in most violations?
           # Simple: Remove the first one.
           S.remove(viol[0])
       
       # Phase 2: Try to add smaller numbers
       small_candidates = list(range(1, 51))
       # Sort by some heuristic? Maybe random or increasing.
       # Increasing is fine.
       for x in small_candidates:
           if time.time() - start > 80: break
           # Check if adding x creates a sum of 200
           # This is equivalent to checking if (200 - x) is a subset sum of S
           # Or simpler: check if S U {x} has a subset summing to 200.
           # We can just call find_subset_summing_to(S + [x], 200)
           # But that's O(N*T). N=100, T=200. 20k ops.
           # 50 candidates * 20k = 1M ops. Very fast.
           
           trial = S + [x]
           if find_subset_summing_to(trial, 200) is None:
               S.append(x)
       
       return S
   