import numpy as np
import time
import random

def construct():
       start_time = time.time()
       deadline = start_time + 55.0
       
       best_set = []
       best_size = 0
       
       # Use a fixed seed for reproducibility in testing? No, random is fine.
       # But maybe try specific strategies.
       
       for attempt in range(500):
           if time.time() > deadline:
               break
           
           candidates = list(range(1, 257))
           random.shuffle(candidates)
           
           current_set = []
           current_set_set = set()
           
           # Max sum estimate
           MaxSum = 16000
           dp = np.zeros((61, MaxSum+1), dtype=bool)
           dp[0, 0] = True
           current_max_sum = 0
           
           for x in candidates:
               if time.time() > deadline:
                   break
               
               if x in current_set_set:
                   continue
               
               n_curr = len(current_set)
               
               # We can optimize by only checking up to current_max_sum + x
               # But vectorized ops on the whole array are fast enough.
               
               new_dp = dp.copy()
               
               # Update DP
               for k in range(n_curr + 1, 0, -1):
                   new_dp[k, x:] |= dp[k-1, :-x]
                   
               # Check validity
               # Only need to check sums that could be affected?
               # No, previous sums are still valid.
               # But we need to ensure NO sum has >1 cardinality.
               # Since previous dp was valid, we only need to check sums that are NEW or CHANGED.
               # Actually, if a sum s was in S_k before, and now also in S_j, it's a collision.
               # So we just check the whole array.
               
               counts = new_dp.sum(axis=0)
               if not np.any(counts > 1):
                   current_set.append(x)
                   current_set_set.add(x)
                   dp = new_dp
                   current_max_sum += x
                   
           if len(current_set) > best_size:
               best_size = len(current_set)
               best_set = current_set
               
       return best_set
