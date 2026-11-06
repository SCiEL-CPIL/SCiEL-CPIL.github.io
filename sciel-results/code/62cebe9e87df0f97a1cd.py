import itertools
import random
import time

def construct():
       n = 16
       k = 4
       # Generate all k-subsets as bitmasks
       all_sets = []
       for combo in itertools.combinations(range(n), k):
           mask = 0
           for i in combo:
               mask |= (1 << i)
           all_sets.append(mask)
       
       # Precompute pair intersections? No, too many pairs (1.6M).
       # Just compute on the fly.
       
       best_family = []
       start_time = time.time()
       deadline = start_time + 85.0 # Safety margin
       
       # We can try to order sets by some heuristic initially?
       # Or just random shuffles.
       
       # To improve greedy, we can use a "conflict count" to pick the best candidate.
       # But that requires O(N^2 * M) per step, which might be slow.
       # Let's try random shuffles first.
       
       for _ in range(1000): # Max iterations, limited by time
           if time.time() > deadline:
               break
           
           random.shuffle(all_sets)
           current_family = []
           
           for c in all_sets:
               # Check if c can be added
               valid = True
               # Optimization: if len(current_family) is large, this is the bottleneck
               # We can break early if invalid
               for i in range(len(current_family)):
                   a = current_family[i]
                   # Intersect a and c
                   ac = a & c
                   for j in range(i + 1, len(current_family)):
                       b = current_family[j]
                       # Check sunflower condition
                       # A sunflower with core K requires:
                       # A \cap B = K, A \cap C = K, B \cap C = K
                       # Let K = A \cap B.
                       # We need A \cap C == K and B \cap C == K.
                       ab = a & b
                       if ab == ac and ab == (b & c):
                           valid = False
                           break
                   if not valid:
                       break
               
               if valid:
                   current_family.append(c)
                   
           if len(current_family) > len(best_family):
               best_family = current_family
               
       # Convert bitmasks back to sorted lists
       result = []
       for mask in best_family:
           s = []
           for i in range(n):
               if mask & (1 << i):
                   s.append(i + 1)
           result.append(s)
           
       return result
