import numpy as np
import time

def construct():
          start_time = time.time()
          deadline = 50.0 # 50 seconds safety margin
          
          N = 10_000_000
          
          # 1. Compute sigma(n)
          sigma = np.ones(N+1, dtype=np.int32)
          
          # Sieve
          # Optimization: Only loop up to N
          # We can break early if time is up, but we need full sigma for correctness.
          # If time runs out, we might return a suboptimal result.
          
          for i in range(1, N+1):
              sigma[i::i] += i
              # Check time every 1000 iterations?
              # Overhead of time check might be significant.
              # Just run it.
              
          # 2. Prepare for grouping
          # We need to find pairs a, b with sigma[a] == sigma[b]
          # and minimize |2a - 3b|
          
          # Create array of indices 1..N
          n_vals = np.arange(1, N+1, dtype=np.int32)
          
          # Sort by sigma value
          # argsort is stable? Not required.
          sorted_indices = np.argsort(sigma[1:], kind='stable')
          sorted_sigma = sigma[1:][sorted_indices]
          sorted_n = n_vals[sorted_indices]
          
          # 3. Scan groups
          best_err = float('inf')
          best_a = 1
          best_b = 1
          
          # Iterate through sorted_sigma
          # Group boundaries where sigma changes
          
          i = 0
          M = N
          while i < M:
              j = i
              current_sigma = sorted_sigma[i]
              while j < M and sorted_sigma[j] == current_sigma:
                  j += 1
              
              # Group is sorted_n[i:j]
              group_size = j - i
              if group_size >= 2:
                  # Find min |2a - 3b|
                  # a, b in sorted_n[i:j]
                  # Since sorted_n is sorted, we can use two pointers or binary search.
                  # Let A = 2 * sorted_n[i:j]
                  # Let B = 3 * sorted_n[i:j]
                  # We want min |A[k] - B[l]|
                  
                  # If group is small, brute force
                  if group_size < 100:
                      group = sorted_n[i:j]
                      for k in range(group_size):
                          a = group[k]
                          # Find b close to 2/3 a
                          # b approx 2/3 a
                          target_b = int(2/3 * a)
                          # Binary search in group for target_b
                          # Since group is sorted, we can use np.searchsorted
                          # But group is a slice, creating a view is cheap.
                          # However, doing this for every k in a small group is fine.
                          # Actually, for small group, just double loop?
                          # 100*100 = 10000 ops.
                          # Number of groups is large.
                          # Better to use the general method for all groups?
                          
                          # General method:
                          # A = 2 * group
                          # B = 3 * group
                          # Sort A and B (they are already sorted if group is sorted, since 2x and 3x are monotonic)
                          # So A and B are sorted.
                          # Merge scan.
                          
                          # Let's implement the merge scan for all groups.
                          pass
              
              i = j
          
          return [int(best_a), int(best_b)]
