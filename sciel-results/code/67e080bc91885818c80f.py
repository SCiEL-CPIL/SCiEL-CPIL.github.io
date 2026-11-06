import time
import math

def construct():
       start_time = time.time()
       deadline = start_time + 55.0
       
       limit = 10000
       is_prime = [True] * (limit + 1)
       is_prime[0] = is_prime[1] = False
       for i in range(2, int(math.isqrt(limit)) + 1):
           if is_prime[i]:
               for j in range(i*i, limit + 1, i):
                   is_prime[j] = False
       
       primes = [i for i, v in enumerate(is_prime) if v]
       N = len(primes)
       
       if N < 2:
           return primes
           
       # dp[i][j] = length of longest sequence ending with primes[i], primes[j]
       # Initialize to 2 for all i < j
       dp = [[0] * N for _ in range(N)]
       parent = [[-1] * N for _ in range(N)]
       
       for i in range(N):
           for j in range(i + 1, N):
               dp[i][j] = 2
               
       max_len = 2
       best_j = 0
       best_k = 1
       
       # For each j, we process extensions to k > j
       # We need to consider all i < j
       
       for j in range(N - 1):
           if time.time() > deadline:
               break
               
           # Collect candidates i < j
           # candidate: (gap_ij, dp[i][j], i)
           candidates = []
           for i in range(j):
               if dp[i][j] > 0:
                   gap = primes[j] - primes[i]
                   candidates.append((gap, dp[i][j], i))
                   
           if not candidates:
               continue
               
           # Sort by gap
           candidates.sort(key=lambda x: x[0])
           
           # Precompute prefix max of dp values and corresponding index i
           # pref_max[m] = (max_dp_value, index_i)
           pref_max_dp = []
           pref_max_i = []
           current_max_dp = 0
           current_max_i = -1
           for gap, d, i in candidates:
               if d > current_max_dp:
                   current_max_dp = d
                   current_max_i = i
               pref_max_dp.append(current_max_dp)
               pref_max_i.append(current_max_i)
               
           # Now iterate k > j
           # gap_jk = primes[k] - primes[j]
           # We need max d such that gap_ij <= gap_jk
           # Since candidates are sorted by gap, we can binary search or use a pointer
           # Since k increases, gap_jk increases.
           # We can use a pointer `ptr` that moves forward as gap_jk increases.
           
           ptr = 0
           num_candidates = len(candidates)
           
           for k in range(j + 1, N):
               gap_jk = primes[k] - primes[j]
               
               # Advance ptr while candidates[ptr].gap <= gap_jk
               while ptr < num_candidates and candidates[ptr][0] <= gap_jk:
                   ptr += 1
                   
               # Now ptr is the count of valid candidates
               if ptr > 0:
                   # The best candidate is at index ptr - 1 in the sorted list?
                   # No, we want the max dp among the first ptr candidates.
                   # pref_max_dp[ptr-1] is the max dp among candidates[0..ptr-1]
                   best_d = pref_max_dp[ptr - 1]
                   best_i = pref_max_i[ptr - 1]
                   
                   new_len = best_d + 1
                   if new_len > dp[j][k]:
                       dp[j][k] = new_len
                       parent[j][k] = best_i
                       
                       if new_len > max_len:
                           max_len = new_len
                           best_j = j
                           best_k = k
                           
       # Reconstruct
       if max_len == 1:
           return [primes[0]]
           
       path = []
       curr_k = best_k
       curr_j = best_j
       
       # The sequence ends with ... -> curr_j -> curr_k
       path.append(primes[curr_k])
       
       # If max_len is 2, we just need curr_j
       if max_len > 2:
           while curr_j != -1:
               path.append(primes[curr_j])
               prev_i = parent[curr_j][curr_k]
               curr_k = curr_j
               curr_j = prev_i
               if curr_j == -1:
                   break
                   
       # If max_len was 2, the loop above might not run correctly if parent is -1
       # Let's handle max_len == 2 separately or ensure loop logic is robust.
       
       # Actually, if max_len == 2, parent[best_j][best_k] is -1 (initialized).
       # The loop `while curr_j != -1` will append primes[best_j], then set curr_j = -1, and break.
       # So path will be [primes[best_k], primes[best_j]].
       
       path.reverse()
       return path
