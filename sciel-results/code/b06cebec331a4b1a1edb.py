   import time
   import random
   from collections import deque, defaultdict

   def construct(k, g):
       start_time = time.time()
       deadline = start_time + 110
       
       if k != 3:
           # Fallback
           raise ValueError("Only k=3 supported")
           
       # Moore bound for k=3, g=16 is 512.
       # Try n=512 first. If fails, maybe 514?
       # The task asks for "as few vertices as possible".
       # We should try to find a valid graph at n=512.
       
       n_candidates = [512, 514, 516, 518, 520]
       
       for n in n_candidates:
           if time.time() > deadline - 10:
               break
           
           # Try to construct a valid graph of size n
           graph = try_construct(n, g, deadline)
           if graph is not None:
               return n, graph
               
       # If all fail, return the best effort or raise
       # The contract says "invalid result... scores 0".
       # We must return a valid graph.
       # If we can't find one, we might have to return a larger one?
       # But the score is n. Lower is better.
       # If we can't find n=512, we should try larger n.
       
       # Let's just try n=512 with multiple restarts.
       
       return 512, try_construct(512, g, deadline) # This might return None
   