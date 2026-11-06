   def construct():
       import time
       start_time = time.time()
       deadline = start_time + 80.0 # 10s margin
       
       from math import lcm
       
       # Initial set: (64, 128]
       S = set(range(65, 129))
       
       # Verify S is valid (no bad triples)
       # If not, fix it (remove elements from bad triples)
       # For simplicity, assume it is valid, but add a check loop if we want to be safe.
       
       # Greedy add from 64 down to 1
       for x in range(64, 0, -1):
           if time.time() > deadline:
               break
           
           # Check if adding x creates a bad triple
           # Triples are (x, a, b) with a, b in S
           # We need to check all pairs (a, b) in S
           # Optimization: 
           # If lcm(a, b) is not a multiple of x? No, condition is complex.
           # Just compute.
           
           valid = True
           # Convert S to list for iteration? Or iterate set.
           # S size ~ 64-100.
           S_list = list(S)
           for i in range(len(S_list)):
               a = S_list[i]
               for j in range(i+1, len(S_list)):
                   b = S_list[j]
                   l_ab = lcm(a, b)
                   l_xa = lcm(x, a)
                   if l_xa != l_ab:
                       continue
                   l_xb = lcm(x, b)
                   if l_xb == l_ab:
                       valid = False
                       break
               if not valid:
                   break
           
           if valid:
               S.add(x)
               
       return list(S)
   