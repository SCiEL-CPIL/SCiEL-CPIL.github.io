import time
import random
from math import gcd

def construct():
       # Compute LCM of 1..22
       def lcm(a, b):
           return a * b // gcd(a, b)
       
       L = 1
       for i in range(1, 23):
           L = lcm(L, i)
           
       # Values are L//i
       # We want to select a subset of indices 1..22
       # such that the corresponding values form a dissociated set.
       
       def try_ordering(order):
           sums = {0}
           selected_indices = []
           for idx in order:
               v = L // idx
               new_sums = {s + v for s in sums}
               # Check intersection
               # If intersection is not empty, skip
               if new_sums & sums:
                   continue
               sums |= new_sums
               selected_indices.append(idx)
           return selected_indices
       
       best_set = []
       
       # Try deterministic orderings
       # 1. Ascending index (1, 2, ..., 22) -> Descending values
       order1 = list(range(1, 23))
       res1 = try_ordering(order1)
       if len(res1) > len(best_set):
           best_set = res1
           
       # 2. Descending index (22, 21, ..., 1) -> Ascending values
       order2 = list(range(22, 0, -1))
       res2 = try_ordering(order2)
       if len(res2) > len(best_set):
           best_set = res2
           
       # 3. Random orderings
       # Set a time limit
       start_time = time.time()
       time_limit = 50.0 # Keep 10s margin
       
       random.seed(42) # For reproducibility if needed, or just use system random
       for _ in range(1000):
           if time.time() - start_time > time_limit:
               break
           order3 = list(range(1, 23))
           random.shuffle(order3)
           res3 = try_ordering(order3)
           if len(res3) > len(best_set):
               best_set = res3
               
       return best_set
