import time
from math import lcm
import bisect

def construct():
       start_time = time.time()
       deadline = start_time + 80.0

       denominators = list(range(2, 49))
       n = len(denominators)
       
       L = 1
       for d in denominators:
           L = lcm(L, d)
       
       weights = [L // d for d in denominators]
       
       mid = n // 2
       w1 = weights[:mid]
       w2 = weights[mid:]
       
       def gen_sums(weights_list):
           sums = [(0, 0)]
           for i, w in enumerate(weights_list):
               if time.time() > deadline:
                   return sums
               bit = 1 << i
               current_len = len(sums)
               # Use list comprehension for speed?
               # new_sums = [(s+w, m|bit) for s, m in sums[:current_len] if s+w < L]
               # This is faster than manual loop in CPython
               new_sums = []
               L_local = L
               for j in range(current_len):
                   s, m = sums[j]
                   ns = s + w
                   if ns < L_local:
                       new_sums.append((ns, m | bit))
               sums.extend(new_sums)
           return sums

       sums1 = gen_sums(w1)
       sums2 = gen_sums(w2)
       
       if time.time() > deadline:
           return [0]
       
       sums2.sort(key=lambda x: x[0])
       sums2_vals = [s[0] for s in sums2]
       
       best_sum = -1
       best_m1 = 0
       best_m2 = 0
       
       bisect_right = bisect.bisect_right
       L_local = L
       
       for s1, m1 in sums1:
           if time.time() > deadline:
               break
           target = L_local - 1 - s1
           if target < 0:
               continue
           idx = bisect_right(sums2_vals, target)
           if idx > 0:
               s2, m2 = sums2[idx - 1]
               total = s1 + s2
               if total > best_sum:
                   best_sum = total
                   best_m1 = m1
                   best_m2 = m2
                   
       # Reconstruct
       best_indices = []
       for i in range(mid):
           if best_m1 & (1 << i):
               best_indices.append(i)
       for i in range(len(w2)):
           if best_m2 & (1 << i):
               best_indices.append(mid + i)
               
       if not best_indices:
           best_indices = [0]
           
       return best_indices
