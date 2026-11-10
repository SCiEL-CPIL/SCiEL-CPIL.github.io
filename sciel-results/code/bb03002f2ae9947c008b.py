import numpy as np
import time

def construct():
       start_time = time.time()
       deadline = start_time + 50.0

       limit = 10000
       # Sieve
       is_prime = [True] * (limit + 1)
       is_prime[0] = is_prime[1] = False
       for i in range(2, int(limit**0.5) + 1):
           if is_prime[i]:
               for j in range(i*i, limit + 1, i):
                   is_prime[j] = False
       primes = [i for i, p in enumerate(is_prime) if p]
       N = len(primes)
       
       if N == 0: return []
       if N == 1: return primes

       MAX_GAP = limit
       
       # prefix_dp[i][g] = max length ending at i with last gap <= g
       # argmax_dp[i][g] = the gap value g' <= g that achieves prefix_dp[i][g]
       prefix_dp = np.zeros((N, MAX_GAP + 1), dtype=np.int32)
       argmax_dp = np.zeros((N, MAX_GAP + 1), dtype=np.int32)
       
       # parents[i] is a dict: {gap: (j, g_old)}
       parents = [dict() for _ in range(N)]
       
       best_len = 1
       best_i = 0
       best_g = 0
       
       for i in range(N):
           if time.time() > deadline:
               break
               
           # raw_vals[g] = max length ending at i with last gap EXACTLY g
           # We use a dict to store only non-zero values
           raw_vals = {0: 1} # Base case: start sequence at i, gap 0, length 1
           
           p_i = primes[i]
           
           for j in range(i):
               p_j = primes[j]
               g_new = p_i - p_j
               
               # Query best length from j with gap <= g_new
               best_prev = prefix_dp[j, g_new]
               
               if best_prev > 0:
                   g_old = argmax_dp[j, g_new]
                   cand_len = best_prev + 1
                   
                   # Update raw_vals for i
                   current_best = raw_vals.get(g_new, 0)
                   if cand_len > current_best:
                       raw_vals[g_new] = cand_len
                       parents[i][g_new] = (j, g_old)
                       
                       if cand_len > best_len:
                           best_len = cand_len
                           best_i = i
                           best_g = g_new
           
           # Build prefix arrays for i
           # Initialize raw array
           raw_arr = np.zeros(MAX_GAP + 1, dtype=np.int32)
           raw_arr[0] = 1
           for g, val in raw_vals.items():
               if g > 0:
                   if val > raw_arr[g]:
                       raw_arr[g] = val
           
           # Compute prefix max
           prefix_dp[i] = np.maximum.accumulate(raw_arr)
           
           # Compute argmax
           # We need the index of the max value in the prefix.
           # If multiple, take the last one? Or first?
           # The definition of argmax in the query `argmax_dp[j, g_new]`
           # returns a gap $g'$ such that $dp[j][g'] = prefix\_dp[j, g\_new]$.
           # Any such $g'$ is valid.
           # Let's take the largest $g'$ that achieves the max, or just any.
           # Simple loop:
           arg_arr = np.zeros(MAX_GAP + 1, dtype=np.int32)
           max_val = 0
           max_idx = 0
           for g in range(MAX_GAP + 1):
               if raw_arr[g] > max_val:
                   max_val = raw_arr[g]
                   max_idx = g
               arg_arr[g] = max_idx
           argmax_dp[i] = arg_arr

       # Reconstruct
       seq = []
       curr_i = best_i
       curr_g = best_g
       
       while curr_i != -1:
           seq.append(primes[curr_i])
           if curr_g == 0:
               break
           if curr_g not in parents[curr_i]:
               break # Safety
           prev_i, prev_g = parents[curr_i][curr_g]
           curr_i = prev_i
           curr_g = prev_g
       
       seq.reverse()
       return seq
