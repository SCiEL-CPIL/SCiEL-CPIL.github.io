import random
import time

def construct():
       start_time = time.time()
       deadline = start_time + 85  # 5s margin
       
       best_size = 0
       best_set = set()
       
       # Natural order greedy first
       S = {1, 2, 3}
       for x in range(4, 513):
           ok = True
           max_d = (x - 1) // 3
           for d in range(1, max_d + 1):
               a = x - 3 * d
               if a < 1:
                   break
               if a in S and (a + d) in S and (a + 2 * d) in S:
                   ok = False
                   break
           if ok:
               S.add(x)
       best_size = len(S)
       best_set = S.copy()
       
       # Randomized greedy
       random.seed(42) # Fixed seed for reproducibility? Or random?
       # Use a fixed seed to ensure deterministic output if needed, but random order is the key.
       # Let's try a few random permutations.
       
       for _ in range(100): # Try 100 random permutations
           if time.time() > deadline:
               break
           
           S = set()
           # Start with random small set? Or just empty?
           # If we start empty, we can add 1,2,3,4?
           # 1,2,3,4 is a 4-AP. So we can't add all 4.
           # The greedy algorithm will handle this.
           
           # We need to ensure we don't form a 4-AP.
           # We can just iterate through the permutation and add if valid.
           # But "valid" check needs to be fast.
           
           # Optimization: Precompute conflicts?
           # Or just check on the fly.
           
           perm = list(range(1, 513))
           random.shuffle(perm)
           
           for x in perm:
               if time.time() > deadline:
                   break
               ok = True
               # Check if adding x creates a 4-AP with elements in S
               # x can be any of the 4 terms.
               # We need to check all d such that the 4 terms are in [1, 512].
               # And at least 3 of them are in S (plus x).
               
               # This is slower than the "x is largest" check because x can be any term.
               # However, S is a set, so membership is O(1).
               
               # For each d:
               # Case 1: x is a (smallest). Terms: x, x+d, x+2d, x+3d.
               #          Need x+d, x+2d, x+3d in S.
               # Case 2: x is a+d. Terms: x-d, x, x+d, x+2d.
               #          Need x-d, x+d, x+2d in S.
               # Case 3: x is a+2d. Terms: x-2d, x-d, x, x+d.
               #          Need x-2d, x-d, x+d in S.
               # Case 4: x is a+3d. Terms: x-3d, x-2d, x-d, x.
               #          Need x-3d, x-2d, x-d in S.
               
               # We can iterate d from 1 to 512//3 = 170.
               # For each d, check the 3 conditions.
               
               for d in range(1, 171):
                   # Case 1
                   if (x+d <= 512 and x+d in S and x+2*d in S and x+3*d in S):
                       ok = False
                       break
                   # Case 2
                   if (x-d >= 1 and x-d in S and x+d in S and x+2*d in S):
                       ok = False
                       break
                   # Case 3
                   if (x-2*d >= 1 and x-2*d in S and x-d in S and x+d in S):
                       ok = False
                       break
                   # Case 4
                   if (x-3*d >= 1 and x-3*d in S and x-2*d in S and x-d in S):
                       ok = False
                       break
               
               if ok:
                   S.add(x)
           
           if len(S) > best_size:
               best_size = len(S)
               best_set = S.copy()
       
       return sorted(best_set)
