import time
import random
from itertools import combinations_with_replacement

def construct():
       start_time = time.time()
       time_limit = 88.0 # Safety margin

       def elapsed():
           return time.time() - start_time

       def check_time():
           return elapsed() > time_limit

       # We want to find the largest set.
       # Let's try sizes from high to low.
       # Based on score 1.7778 (size 16), we aim for 17, 18, etc.
       
       # Pre-compute nothing, just use bitsets.
       # Max sum is 3 * 4096 = 12288.
       # Python int handles this easily.
       
       def try_find_set(target_size):
           # Returns a list of size target_size if found, else None
           current_set = []
           used_mask = 0
           
           # Heuristic: We can try to bias towards smaller numbers?
           # Or randomize the order of candidates to avoid getting stuck in one local optimum.
           # Since we are doing backtracking, we explore all possibilities (with pruning).
           # To make it faster, we can use a "fail-first" strategy on the candidates.
           # But calculating the "constraint degree" of each candidate dynamically is expensive.
           
           # Let's just iterate 1..4096.
           # To avoid deep recursion on a bad path, we can add a randomization element
           # or simply rely on the pruning.
           
           # Optimization: If we have picked `k` elements, and there are `m` candidates
           # that don't conflict, and `m < target_size - k`, we can prune.
           # We can estimate `m` by checking all 4096 candidates? That's $4096 \times$ recursion_depth.
           # $4096 \times 18 \approx 70k$ checks per node.
           # If branching factor is 10, depth 18, nodes = $10^{18}$... too many.
           # We need much stronger pruning.
           
           # Stronger Pruning:
           # The set of available sums is limited.
           # Each new element $x$ adds roughly $|S|^2/2$ sums.
           # The total number of sums is bounded by $12288$.
           # If the number of used sums + (estimated new sums) > 12288, it's not a direct prune,
           # but if the *free* space is too small to accommodate the required unique sums, we prune.
           
           # Let's implement a simpler backtracking first and see if it works for N=16.
           # If it's too slow, we switch to a randomized local search.
           
           def backtrack(k, current_set, used_mask):
               if check_time():
                   return None
               
               if k == target_size:
                   return current_set[:]
               
               # Pruning: 
               # If we have `k` elements, we need `target_size - k` more.
               # Let's count how many candidates $x \in [1, 4096]$ are valid.
               # This is expensive ($4096$ checks).
               # We can do this every few levels or if we are close to the limit.
               # For now, let's just iterate.
               
               # To speed up, we can skip candidates that are "too similar" to existing ones?
               # No, let's just iterate.
               
               # Heuristic: Try candidates in a random order?
               # No, backtracking needs determinism for correctness, but for finding *a* solution,
               # randomizing the candidate order can help escape bad branches.
               # However, we want to find the *largest* set.
               
               # Let's try iterating 1..4096.
               for x in range(1, 4097):
                   if check_time():
                       return None
                   
                   # Calculate mask for x given current_set
                   # Sums: 3x, 2x+s, x+s_i+s_j
                   mask = 1 << (3 * x)
                   
                   # Check 2x + s
                   for s in current_set:
                       s_val = 2 * x + s
                       mask |= 1 << s_val
                   
                   # Check x + s_i + s_j
                   # Use combinations_with_replacement for current_set
                   # This is O(k^2)
                   for i in range(len(current_set)):
                       for j in range(i, len(current_set)):
                           s_val = x + current_set[i] + current_set[j]
                           mask |= 1 << s_val
                   
                   # Check conflict
                   if mask & used_mask:
                       continue
                   
                   # Recurse
                   current_set.append(x)
                   result = backtrack(k + 1, current_set, used_mask | mask)
                   if result is not None:
                       return result
                   current_set.pop()
                   
               return None
           
           return backtrack(0, [], 0)

       # Try sizes
       best_set = []
       # We know 16 is possible. Let's try 19, 18, 17, 16...
       # If 19 takes too long, we might miss 17.
       # So we should set a time limit per target size.
       
       time_per_target = 10.0
       
       for target_size in [19, 18, 17, 16, 15]:
           if elapsed() > time_limit - 5:
               break
           
           target_start = elapsed()
           # We need to modify backtrack to stop if this specific target takes too long
           # and return None, allowing us to try a smaller target.
           
           # We can't easily modify the inner function's time limit without passing it.
           # Let's just run it and hope it finishes or times out globally.
           # Better: Pass a deadline to the search.
           
           def try_find_set_with_deadline(target_size, deadline):
               current_set = []
               used_mask = 0
               
               def backtrack(k, current_set, used_mask):
                   if time.time() > deadline:
                       return None
                   
                   if k == target_size:
                       return current_set[:]
                   
                   for x in range(1, 4097):
                       if time.time() > deadline:
                           return None
                       
                       mask = 1 << (3 * x)
                       
                       for s in current_set:
                           mask |= 1 << (2 * x + s)
                       
                       for i in range(len(current_set)):
                           for j in range(i, len(current_set)):
                               mask |= 1 << (x + current_set[i] + current_set[j])
                       
                       if mask & used_mask:
                           continue
                       
                       current_set.append(x)
                       result = backtrack(k + 1, current_set, used_mask | mask)
                       if result is not None:
                           return result
                       current_set.pop()
                       
                   return None
               
               return backtrack(0, [], 0)

           # Allocate 8 seconds for each attempt
           deadline = start_time + min(elapsed() + 8.0, time_limit - 2.0)
           sol = try_find_set_with_deadline(target_size, deadline)
           
           if sol:
               best_set = sol
               break
           
       # If no solution found (shouldn't happen as 16 is known), return a greedy fallback
       if not best_set:
            # Fallback to greedy
            selected = []
            used_mask = 0
            for x in range(1, 4097):
                if time.time() > time_limit:
                    break
                mask = 1 << (3 * x)
                for s in selected:
                    mask |= 1 << (2 * x + s)
                for i in range(len(selected)):
                    for j in range(i, len(selected)):
                        mask |= 1 << (x + selected[i] + selected[j])
                if not (mask & used_mask):
                    selected.append(x)
                    used_mask |= mask
            best_set = selected

       return best_set
