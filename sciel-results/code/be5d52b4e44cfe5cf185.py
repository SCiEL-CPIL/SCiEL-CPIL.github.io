   import random
   import time
   import numpy as np

   def construct():
       start_time = time.time()
       deadline = start_time + 80
       
       # Precompute APs
       n = 128
       aps = []
       for a in range(1, n+1):
           max_d = (n - a) // 3
           for d in range(1, max_d + 1):
               aps.append((a-1, a+d-1, a+2*d-1, a+3*d-1)) # 0-indexed
               
       def is_valid(colors):
           for a, b, c, d in aps:
               if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                   return False
           return True
           
       def count_violations(colors):
           count = 0
           for a, b, c, d in aps:
               if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                   count += 1
           return count
           
       # Try k=3, 4, ...
       for k in range(3, 128):
           if time.time() > deadline - 5:
               break
           
           # Local Search
           best_score = float('inf')
           best_colors = None
           
           # Multiple restarts
           for restart in range(50):
               if time.time() > deadline - 5:
                   break
               
               colors = np.random.randint(0, k, size=n)
               current_violations = count_violations(colors)
               
               if current_violations == 0:
                   return list(colors)
                   
               # WalkSAT
               no_improve = 0
               while time.time() < deadline - 5 and no_improve < 1000:
                   # Pick a random violation
                   # To be fast, we might just pick a random variable and check neighbors?
                   # Better: pick a random AP, if violated, pick a random variable in it.
                   # But we need to know which APs are violated.
                   # Efficient way: maintain a list of violated AP indices?
                   # For simplicity in this short window, we can just resample violations occasionally.
                   
                   # Simple heuristic:
                   # Pick random index i
                   i = np.random.randint(n)
                   old_c = colors[i]
                   new_c = np.random.randint(k)
                   if new_c == old_c:
                       continue
                       
                   colors[i] = new_c
                   new_violations = count_violations(colors)
                   
                   if new_violations < current_violations:
                       current_violations = new_violations
                       no_improve = 0
                       if current_violations == 0:
                           return list(colors)
                   else:
                       # Accept with some probability? Or revert?
                       # Standard WalkSAT: accept if improvement, else accept with prob p.
                       # Or just revert if no improvement to keep it simple (hill climbing).
                       # Hill climbing might get stuck. Let's add random restarts.
                       colors[i] = old_c
                       no_improve += 1
                       
               if current_violations < best_score:
                   best_score = current_violations
                   best_colors = colors.copy()
                   
           if best_colors is not None and best_score == 0:
               return list(best_colors)
               
       # Fallback to seed if no better found?
       # The seed is valid. We can just run the seed logic if we fail.
       # But we want to return a list.
       # If we didn't find a valid one with k<seed_k, we should return a valid one.
       # The seed logic is valid. Let's just execute it if we fail.
       
       colors=[]
       for x in range(1,129):
            for color in range(128):
                if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                    and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                    colors.append(color);break
       return colors
   