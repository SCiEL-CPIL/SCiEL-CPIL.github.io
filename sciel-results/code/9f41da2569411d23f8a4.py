import math
import time

def construct():
       start_time = time.time()
       N = 479001600
       
       # 1. Generate divisors
       divisors = []
       for i in range(1, int(N**0.5) + 1):
           if N % i == 0:
               divisors.append(i)
               if i != N // i:
                   divisors.append(N // i)
       
       # Filter > 12
       divisors = [d for d in divisors if d > 12]
       divisors.sort()
       
       # 2. Iterate M
       # We iterate through each divisor as a potential maximum M
       # This is efficient because the optimal M is likely small
       
       # Helper for DFS
       def find_subset(target, start_idx, depth, current_max_bound):
           # current_max_bound is M
           # We are looking for a subset of divisors[start_idx:] that are <= current_max_bound
           # and multiply to target
           
           if target == 1:
               return []
           
           if depth > 12:
               return None
           
           # Pruning: if target is prime, it must be in the list
           # Check if target is prime
           # Simple check for small numbers, or just let the loop handle it
           # The loop will check if target % d == 0. If target is prime, only d=target works.
           
           for i in range(start_idx, len(divisors)):
               d = divisors[i]
               if d > current_max_bound:
                   break
               if d > target:
                   break
               if target % d != 0:
                   continue
               
               # Try d
               res = find_subset(target // d, i + 1, depth + 1, current_max_bound)
               if res is not None:
                   return [d] + res
           
           return None

       for M in divisors:
           # Check deadline
           if time.time() - start_time > 50:
               # Fallback to a known solution or the full N
               # But we should have found a better one
               # For safety, return [N]
               return [N]
           
           # We want to find a subset of divisors <= M
           # The DFS function above uses `divisors` list and `current_max_bound`
           # We need to ensure we only pick from divisors <= M
           # The loop `if d > current_max_bound: break` handles this
           
           # Start DFS from index 0
           # But we must pick factors > 12, which is already filtered
           res = find_subset(N, 0, 0, M)
           if res is not None:
               # Found a solution with max factor <= M
               # Since we iterate M in increasing order, this M is the minimal max
               # The actual max in `res` might be < M, but M is the first bound that allowed a solution
               # Wait, if a solution with max 50 exists, and we check M=49, it fails.
               # When we check M=50, it succeeds.
               # So the returned `res` has max factor <= 50.
               # Is it possible that a solution with max 40 exists, but we missed it?
               # No, we checked M=40 earlier.
               # So the first M that works gives the minimal possible maximum.
               return res
       
       return [N]
