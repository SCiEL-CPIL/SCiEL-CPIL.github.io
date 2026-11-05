import time
from math import lcm

def construct():
       start_time = time.time()
       deadline = 85.0
       
       # LCM of 1..22
       L = 1
       for i in range(1, 23):
           L = lcm(L, i)
           
       # Weights
       weights = [L // i for i in range(1, 23)]
       
       best_set = []
       best_size = 0
       
       # We will use DFS
       # idx: current index in 0..21
       # current_sums: set of sums
       # current_set: list of indices included
       
       def dfs(idx, current_sums, current_set):
           nonlocal best_set, best_size
           
           if time.time() - start_time > deadline:
               return
               
           # Update best
           if len(current_set) > best_size:
               best_size = len(current_set)
               best_set = list(current_set)
               
           if idx == 22:
               return
               
           # Pruning: even if we take all remaining, can't beat best
           if len(current_set) + (22 - idx) <= best_size:
               return
               
           w = weights[idx]
           
           # Option 1: Include idx
           # Check if we can add w without collision
           # Collision if (sums + w) intersects sums
           # We check: for s in sums, if (s + w) in sums -> collision
           # Note: s + w is a new sum. We are adding the set {s+w | s in sums} to sums.
           # Collision if there exists s1, s2 in sums such that s1 + w = s2.
           # i.e., s2 - s1 = w.
           # So we check if w is in {s2 - s1 | s1, s2 in sums, s1 != s2}.
           # Or simply: if any((s + w) in current_sums for s in current_sums)
           
           can_add = True
           # Optimization: if w is very large, it's unlikely to collide.
           # But let's just check.
           # To speed up, we can break early on finding a collision.
           for s in current_sums:
               if (s + w) in current_sums:
                   can_add = False
                   break
           
           if can_add:
               new_sums = current_sums | {s + w for s in current_sums}
               current_set.append(idx)
               dfs(idx + 1, new_sums, current_set)
               current_set.pop()
           
           # Option 2: Exclude idx
           dfs(idx + 1, current_sums, current_set)
       
       # Start DFS
       # To encourage larger sets, we might want to try including elements first.
       # The order 1..22 (large weights first) is good.
       dfs(0, {0}, [])
       
       return [i + 1 for i in best_set]
