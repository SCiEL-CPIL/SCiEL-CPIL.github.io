import time
import sys

def construct():
       divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
       start_time = time.time()
       deadline = start_time + 85.0 # 5s safety margin
       
       # Sort divisors descending to constrain the search space earlier
       # We need to keep track of original indices to return in correct order
       indexed_divs = sorted(enumerate(divisors), key=lambda x: x[1], reverse=True)
       
       best_span = float('inf')
       best_reps = [0] * 10
       
       # We will iterate over possible starting points L.
       # Since the span is likely small, L doesn't need to be huge.
       # However, the optimal L might not be a multiple of any divisor.
       # But the set of "tight" windows is often anchored by a multiple of one of the divisors.
       # Let's iterate L from 1 to 2000. If we don't find a good one, we might miss it, 
       # but given the density, a good solution exists early.
       
       # Actually, a smarter way: 
       # The optimal solution's min value $L^*$ must be a multiple of at least one divisor?
       # Not necessarily. But $L^*$ is close to a multiple.
       # Let's just iterate L.
       
       for L in range(1, 5000):
           if time.time() > deadline:
               break
           
           # If L is already greater than the max value in our best solution, we might prune?
           # No, a different L could yield a smaller span.
           
           # Backtracking search
           # Order: process divisors in descending order of value
           # chosen: list of chosen values
           # We need to ensure distinctness.
           
           def dfs(idx, chosen_vals, current_max):
               nonlocal best_span, best_reps
               
               if time.time() > deadline:
                   return
                   
               if idx == len(indexed_divs):
                   span = current_max - L
                   if span < best_span:
                       best_span = span
                       # Reconstruct best_reps in original order
                       # chosen_vals is in the order of indexed_divs
                       temp_reps = [0] * 10
                       for i, (orig_idx, val) in enumerate(zip([d[0] for d in indexed_divs], chosen_vals)):
                           temp_reps[orig_idx] = val
                       best_reps = temp_reps[:]
                   return
               
               orig_idx, d = indexed_divs[idx]
               
               # Start from the smallest multiple >= L
               k = (L + d - 1) // d
               val = k * d
               
               # Try multiples
               while True:
                   # Pruning: if this value makes the span exceed best_span, stop
                   # Note: future values will be >= val, so if val - L >= best_span, no solution here
                   if val - L >= best_span:
                       break
                   
                   if val not in chosen_vals:
                       chosen_vals.append(val)
                       new_max = max(current_max, val)
                       dfs(idx + 1, chosen_vals, new_max)
                       chosen_vals.pop()
                   
                   val += d
                   # Optional: if val is getting too large, break
                   # The condition val - L >= best_span handles this

           # Initialize chosen_vals
           dfs(0, [], L)
           
           # Heuristic: If we found a span of 1, we can't do better (min possible is 1? 
           # Wait, distinct integers. If we have 10 distinct integers, min span is 9.
           # E.g., 1,2,3,4,5,6,7,8,9,10 -> span 9.
           # Can we get span 9?
           # Divisors: 6,10,14,15,21,22,26,33,35,39.
           # Multiples in a window of size 9?
           # 39 is a divisor. In a window of size 9, there is at most 1 multiple of 39.
           # 35 is a divisor. At most 1 multiple of 35.
           # 33 is a divisor. At most 1 multiple of 33.
           # 26, 22, 21, 15, 14, 10, 6.
           # It is highly likely that span > 9.
           # Let's just let the search find the minimum.
           
           if best_span <= 9:
               break

       # Ensure types are correct
       return [int(x) for x in best_reps]
