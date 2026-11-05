import time
import random

def construct():
       start_time = time.time()
       best_set = []
       
       # Initial deterministic run (descending) as a strong baseline
       selected, reachable = [], {0}
       for x in range(100, 0, -1):
           new = {s + x for s in reachable if s + x <= 200}
           if 200 not in new:
               selected.append(x)
               reachable |= new
       best_set = selected
       
       # Randomized restarts
       # We can try different orders.
       # To speed up, we can pre-compute or just run the greedy.
       # The greedy is O(N * |Reachable|). |Reachable| can be up to 200.
       # So each run is ~ 100 * 200 = 20,000 ops.
       # 1 million runs in 10s? Maybe.
       
       # Let's try 10,000 restarts first.
       elements = list(range(1, 101))
       
       for _ in range(10000):
           if time.time() - start_time > 50:
               break
           
           random.shuffle(elements)
           selected, reachable = [], {0}
           for x in elements:
               # Check if x can be added
               # We need to check if there exists s in reachable such that s + x = 200
               # i.e., 200 - x in reachable
               if 200 - x in reachable:
                   continue
               
               # Also need to ensure that adding x doesn't create a sum > 200 that we care about?
               # No, we only care about sums == 200.
               # But we must update reachable with new sums s+x that are <= 200.
               # If s+x > 200, it doesn't help form 200 later (since all elements positive).
               # So we only track sums <= 200.
               
               selected.append(x)
               # Update reachable
               # new_sums = {s + x for s in reachable if s + x <= 200}
               # reachable |= new_sums
               
               # Optimization:
               # We can just iterate and add
               # But we must be careful not to modify reachable while iterating?
               # In Python, iterating over a set while modifying it is bad.
               # So we create a list of new sums or a new set.
               
               # Actually, the check `if 200 - x in reachable` is sufficient to prevent immediate 200.
               # But adding x creates new sums that might combine with FUTURE elements to make 200.
               # So we MUST update reachable.
               
               new_sums = []
               for s in reachable:
                   ns = s + x
                   if ns <= 200:
                       new_sums.append(ns)
               reachable.update(new_sums)
           
           if len(selected) > len(best_set):
               best_set = selected
               
       return best_set
