import random
import time
import numpy as np

N = 64

def construct():
       def dist_sq(p1, p2):
           dx = p1[0] - p2[0]
           dy = p1[1] - p2[1]
           return dx*dx + dy*dy

       def is_valid_addition(p, current_set):
           # current_set is a list of tuples
           # Check 1: Distances from p to all q in current_set must be unique
           dists_from_p = set()
           for q in current_set:
               d = dist_sq(p, q)
               if d in dists_from_p:
                   return False
               dists_from_p.add(d)
               
           # Check 2 & 3: For every pair q, r in current_set, check isosceles conditions involving p
           # Actually, we can iterate q in current_set, and for each q, check against all r in current_set
           # But we must be careful not to double count or miss.
           # The conditions are:
           # 1. d(p,q) != d(p,r) for q != r (Checked above)
           # 2. d(p,q) != d(q,r) for q != r (Apex q)
           # 3. d(p,r) != d(q,r) for q != r (Apex r)
           
           # Note: Condition 2 and 3 are symmetric in q and r if we swap roles?
           # If we check all pairs (q, r) with q != r:
           #   d(p,q) != d(q,r)
           #   d(p,r) != d(q,r)
           # This covers all cases.
           
           m = len(current_set)
           for i in range(m):
               q = current_set[i]
               d_pq = dist_sq(p, q)
               for j in range(m):
                   if i == j:
                       continue
                   r = current_set[j]
                   d_qr = dist_sq(q, r)
                   # Apex q: d(p,q) == d(q,r)?
                   if d_pq == d_qr:
                       return False
                   # Apex r: d(p,r) == d(q,r)?
                   d_pr = dist_sq(p, r)
                   if d_pr == d_qr:
                       return False
           return True

       # Generate all points
       points = [(x, y) for x in range(N) for y in range(N)]
       
       # Heuristic: Sort points to improve greedy performance.
       # Points closer to the center have more neighbors, so they are more constrained?
       # Or points on the boundary?
       # Let's try sorting by (x+y) or just random shuffle with a fixed seed for reproducibility.
       # A common trick is to start with a sparse grid and fill in.
       
       random.seed(42)
       random.shuffle(points)
       
       current_set = []
       
       # Greedy Phase
       for p in points:
           if is_valid_addition(p, current_set):
               current_set.append(p)
       
       best_set = current_set[:]
       best_size = len(current_set)
       
       # Local Search Phase
       start_time = time.time()
       timeout = 280
       
       # Create a set for O(1) lookup of current points
       current_set_set = set(current_set)
       # We need to handle swaps carefully.
       
       # Candidate points: all points not in current_set
       # We will randomly pick a point to add/swap.
       
       while time.time() - start_time < timeout:
           # Try to add a random point
           p = random.choice(points)
           if p in current_set_set:
               continue
               
           if is_valid_addition(p, current_set):
               current_set.append(p)
               current_set_set.add(p)
               if len(current_set) > best_size:
                   best_size = len(current_set)
                   best_set = current_set[:]
           else:
               # Try swap
               if current_set:
                   q_idx = random.randint(0, len(current_set)-1)
                   q = current_set[q_idx]
                   # Remove q
                   current_set.pop(q_idx)
                   current_set_set.remove(q)
                   
                   if is_valid_addition(p, current_set):
                       current_set.append(p)
                       current_set_set.add(p)
                       if len(current_set) > best_size:
                           best_size = len(current_set)
                           best_set = current_set[:]
                   else:
                       # Revert
                       current_set.append(q)
                       current_set_set.add(q)
                       
       return best_set
