"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    LIMIT = 10000
    DEADLINE = start_time + 50  # Safety margin before 90s limit

    # 1. Generate all primes <= 10000 using Sieve of Eratosthenes (NumPy accelerated)
    sieve = np.ones(LIMIT + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(np.sqrt(LIMIT)) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    
    primes = np.where(sieve)[0]
    n = len(primes)
    
    if n < 2:
        return list(primes)

    # 2. Compute gaps between consecutive primes in the full list
    # gaps[i] = primes[i+1] - primes[i]
    gaps = np.diff(primes)

    # 3. Dynamic Programming to find Longest Non-Decreasing Subsequence (LNDS)
    # We want a subsequence of primes p_{i1}, p_{i2}, ..., p_{ik} such that
    # gaps[i_{j-1}] <= gaps[i_j] for all j.
    # Note: The "gap" in the subsequence is defined by the difference between the selected primes.
    # Wait, the task says "successive gaps NONDECREASING".
    # If we select indices i1 < i2 < i3, the gaps are (p[i2]-p[i1]) and (p[i3]-p[i2]).
    # This is NOT simply a subsequence of the `gaps` array.
    # The gap between selected primes is the sum of consecutive gaps in the prime list.
    
    # Let's re-read carefully: "increasing prime integers <=10000 with successive gaps NONDECREASING".
    # If we pick primes p1, p2, p3, the gaps are g1 = p2-p1 and g2 = p3-p2. We require g1 <= g2.
    # Since p1, p2, p3 are primes from the set of all primes <= 10000, but not necessarily consecutive in the prime list,
    # g1 is the sum of gaps between p1 and p2 in the full prime list.
    
    # This is a harder problem than LNDS on the gap array.
    # However, note that for small numbers (<=10000), the number of primes is ~1229.
    # A DP over all pairs of primes O(N^2) might be feasible if optimized, but O(N^2) states with O(N) transitions is O(N^3) which is too slow (1229^3 ~ 1.8e9).
    
    # Let's look at the structure.
    # We want to maximize the length of a sequence p_1 < p_2 < ... < p_k.
    # Constraint: (p_{j+1} - p_j) is non-decreasing.
    
    # Let's try a different DP state:
    # dp[i][j] = length of the longest valid sequence ending with primes i and j (where i < j).
    # Transition: dp[i][j] = 1 + max(dp[h][i]) for all h < i such that (i - h) <= (j - i).
    # This is O(N^3) in the worst case. N=1229. 1229^3 is too big for Python in 60s.
    
    # Can we optimize?
    # The condition is (j - i) >= (i - h) => j + h >= 2i.
    # So for a fixed pair (i, j), we need to find the max dp[h][i] for h < i and h >= 2i - j.
    
    # Let's precompute for each i, the values dp[h][i] for all h < i.
    # We can iterate j from 1 to N-1.
    # For each j, we iterate i from 0 to j-1.
    # We need max(dp[h][i]) for h in [max(0, 2i-j), i-1].
    
    # This suggests a sliding window maximum or a prefix max structure, but the range [2i-j, i-1] depends on j.
    # As j increases, the lower bound 2i-j decreases.
    
    # Let's try a simpler approach first: Is N small enough for O(N^2) with some pruning?
    # Or maybe O(N^2) is acceptable if the inner loop is tight?
    # N=1229. N^2 = 1.5e6. If the inner loop is O(1) or very small, it's fine.
    # The transition is max over h. That's O(N). So O(N^3) is the bottleneck.
    
    # Alternative:
    # Since the gaps are non-decreasing, the sequence is "convex".
    # For primes, gaps grow on average.
    
    # Let's check the size of N again.
    # pi(10000) = 1229.
    # O(N^3) is definitely too slow in pure Python.
    # We need to use NumPy or a smarter algorithm.
    
    # Let's try to formulate it as:
    # dp[i] = best length ending at prime i? No, we need the previous prime to check the gap constraint.
    # So state must be (prev_prime, curr_prime).
    
    # Let's use NumPy to vectorize the max operation.
    # dp is a 2D array of size N x N. dp[i][j] is length of sequence ending with primes[i], primes[j].
    # We fill dp for j from 1 to N-1.
    # For a fixed j, we want to compute dp[:, j].
    # dp[i][j] = 1 + max(dp[h][i]) for h < i and (primes[i]-primes[h]) <= (primes[j]-primes[i]).
    
    # Let's loop j.
    # For each j, we have a vector of i's (0..j-1).
    # We need to compute for each i in 0..j-1:
    #   max_val = max(dp[h][i] for h in range(0, i) if primes[i]-primes[h] <= primes[j]-primes[i])
    #   dp[i][j] = max_val + 1 (or 2 if no valid h)
    
    # This is still O(N^3) if done naively.
    # But N=1229.
    # Let's estimate: 1229 * 1229 * (avg i/2) ~ 1229^3 / 2 ~ 9e8 operations.
    # In Python, this is ~1000 seconds. Too slow.
    
    # We need to optimize the inner max.
    # Condition: primes[h] >= primes[i] - (primes[j]-primes[i]) = 2*primes[i] - primes[j].
    # Let L_i_j = 2*primes[i] - primes[j].
    # We need max dp[h][i] for h such that primes[h] >= L_i_j and h < i.
    # Since primes are sorted, this is a range max query on the array dp[0:i, i] (column i, rows 0 to i-1)
    # restricted to indices h where primes[h] >= L_i_j.
    
    # We can precompute for each column i, a structure that allows range max queries?
    # But the range is defined by value (primes[h]), not index.
    # However, primes are sorted, so primes[h] >= L_i_j implies h >= lower_bound(primes, L_i_j).
    # Let h_min = np.searchsorted(primes, L_i_j).
    # Then we need max(dp[h_min:i, i]).
    
    # So for each i, we can precompute a prefix max array for column i?
    # No, we need max over a suffix of the column (from h_min to i-1).
    # We can precompute suffix max for each column i.
    # suffix_max[i][k] = max(dp[k:i, i]).
    # Then the query is O(1): suffix_max[i][h_min].
    
    # Precomputation:
    # For each i, we compute dp[:, i] based on previous columns.
    # But dp[h][i] depends on dp[:, h].
    # So we must compute columns one by one.
    # When we compute column i, we have already computed columns 0..i-1.
    # So dp[0:i, i] is not known yet?
    # Wait, dp[h][i] is the length of sequence ending with h, i.
    # To compute dp[h][i], we need max(dp[k][h]) for k < h.
    # So yes, to compute column i, we need all previous columns.
    
    # Algorithm:
    # 1. Initialize dp as a 2D array of zeros, size N x N.
    # 2. For i in range(1, N):
    #      For h in range(0, i):
    #          dp[h][i] = 2 (base case: sequence of length 2)
    #      # Now we want to update dp[h][i] using previous columns.
    #      # Actually, dp[h][i] = 1 + max(dp[k][h]) for k < h and primes[h]-primes[k] <= primes[i]-primes[h].
    #      # Let's compute this for all h in 0..i-1.
    #      # For a fixed h, the condition is primes[k] >= 2*primes[h] - primes[i].
    #      # Let L = 2*primes[h] - primes[i].
    #      # k_min = np.searchsorted(primes, L).
    #      # We need max(dp[k_min:h, h]).
    #      # We can precompute suffix_max for column h?
    #      # But column h was computed in the previous step (when i was h).
    #      # So yes, we can have an array suff_max[h] which is the suffix max of column h.
    #      # suff_max[h][k] = max(dp[k:h, h]).
    
    # Let's refine:
    # dp is N x N.
    # suff_max is a list of arrays, or a 2D array.
    # suff_max[i][k] = max(dp[k:i, i]).
    
    # Step 1: For i=1, h=0.
    # dp[0][1] = 2.
    # suff_max[1] = [0, 2] (index 0 is unused for max, or max from 0 to 0 is dp[0][1]? No, suff_max[1][0] = dp[0][1]).
    # Actually, we need max(dp[k:h, h]) for k in [k_min, h-1].
    # So we need suffix max of column h, from index 0 to h-1.
    # Let's define S[h][k] = max(dp[k:h, h]).
    # S[h][h-1] = dp[h-1][h]
    # S[h][k] = max(dp[k][h], S[h][k+1])
    
    # So:
    # for i in range(1, N):
    #     # Compute column i
    #     # For each h in 0..i-1:
    #     #     L = 2*primes[h] - primes[i]
    #     #     k_min = searchsorted(primes, L)
    #     #     if k_min < h:
    #     #         best_prev = S[h][k_min]
    #     #     else:
    #     #         best_prev = 0
    #     #     dp[h][i] = best_prev + 1
    #     #     if dp[h][i] < 2: dp[h][i] = 2 # Base case
    #     
    #     # Compute S[i]
    #     # S[i][i-1] = dp[i-1][i]
    #     # for k in range(i-2, -1, -1):
    #     #     S[i][k] = max(dp[k][i], S[i][k+1])
    
    # Complexity:
    # Outer loop i: N
    # Inner loop h: i
    # Total h iterations: N^2/2.
    # Inside h loop: searchsorted (O(log N)), array access (O(1)).
    # Computing S[i]: O(i).
    # Total operations: O(N^2 log N).
    # N=1229. N^2 ~ 1.5e6. log N ~ 11. Total ~ 1.6e7 operations.
    # This is very fast in Python/NumPy.
    
    # Let's implement this.
    
    # Note: dp[h][i] represents length of sequence ending with h, i.
    # The length includes h and i.
    # Base case: sequence of two primes h, i has length 2.
    # If no valid k, dp[h][i] = 2.
    # If valid k exists, dp[h][i] = max(dp[k][h]) + 1.
    
    # We need to track the best length and reconstruct the path.
    # Let's store dp and parent pointers.
    # parent[h][i] = the k that gave the max value.
    
    primes_list = list(primes)
    N = len(primes_list)
    
    # dp[i][j] where i < j
    # We'll use a 2D list or array.
    # dp[i][j] = length
    # par[i][j] = previous index k
    
    dp = np.zeros((N, N), dtype=np.int32)
    par = np.full((N, N), -1, dtype=np.int32)
    
    # S[i] is suffix max of column i, for rows 0..i-1.
    # S[i][k] = max(dp[k:i, i])
    # We only need S for the current i to compute the next?
    # No, to compute column j, we need S[h] for all h < j.
    # So we need to store all S[h].
    # S is a list of arrays of length h.
    
    S = [None] * N
    
    for i in range(1, N):
        # Compute column i
        # h ranges from 0 to i-1
        # L_h = 2*primes[h] - primes[i]
        # k_min = searchsorted(primes, L_h)
        
        # Vectorize the searchsorted?
        # L = 2*primes[0:i] - primes[i]
        # k_mins = np.searchsorted(primes, L)
        
        L = 2 * primes[0:i] - primes[i]
        k_mins = np.searchsorted(primes, L)
        
        # For each h, get best_prev
        # best_prev[h] = S[h][k_mins[h]] if k_mins[h] < h else 0
        
        best_prevs = np.zeros(i, dtype=np.int32)
        # We can't easily vectorize the list access S[h][k].
        # But i is small (up to 1229). A loop is fine.
        
        for h in range(i):
            km = k_mins[h]
            if km < h:
                best_prevs[h] = S[h][km]
            else:
                best_prevs[h] = 0
        
        # dp[0:i, i] = best_prevs + 1
        # But base case is 2.
        # If best_prevs[h] is 0, dp is 1? No, sequence of length 1 is not valid?
        # The problem asks for "increasing prime integers".
        # A single prime has no gaps. Is it valid?
        # "successive gaps NONDECREASING".
        # Usually, a sequence of length 1 or 2 is trivially valid.
        # But we want to maximize length.
        # If we start with length 1, adding a prime makes it length 2.
        # dp[h][i] should be at least 2.
        # If best_prevs[h] is 0, it means no previous prime k.
        # So the sequence is just [h, i]. Length 2.
        # So dp[h][i] = max(2, best_prevs[h] + 1).
        # Since best_prevs[h] >= 0, best_prevs[h] + 1 >= 1.
        # If best_prevs[h] == 0, dp = 1? No, we are ending with h, i.
        # The sequence ending with h, i has at least 2 elements.
        # So dp[h][i] = best_prevs[h] + 1 if best_prevs[h] > 0 else 2.
        # Actually, if best_prevs[h] is 0, it means we didn't find a k.
        # So the longest sequence ending with k, h was length 0 (none).
        # So we start a new sequence with h, i. Length 2.
        
        vals = np.where(best_prevs > 0, best_prevs + 1, 2)
        dp[0:i, i] = vals
        
        # Update par
        # We need to know which k gave the max.
        # This is harder to vectorize.
        # Let's just store the value for now, and reconstruct by re-scanning?
        # Or store par in the loop.
        
        # Let's re-do the loop to store par.
        for h in range(i):
            km = k_mins[h]
            if km < h:
                # Find the index of the max in S[h][km:h]
                # S[h] is suffix max. We lost the index.
                # We need to store argmax or re-scan.
                # Re-scanning is O(h). Total O(N^3) again?
                # No, we only do this if we want to reconstruct.
                # For now, let's just get the value.
                # To reconstruct, we can store the argmax in a separate structure or recompute.
                # Given N=1229, re-scanning O(N^3) might be too slow if done for all.
                # But we only need to reconstruct the BEST path, not all.
                # So we can just store dp values, find the max dp[i][j], and then backtrack.
                # Backtracking requires knowing the parent.
                # So we MUST store parent.
                
                # Let's find the argmax in dp[km:h, h].
                # This is O(h).
                # Total O(N^3).
                # Is there a way to store argmax in S?
                # S_idx[h][k] = index of max in dp[k:h, h].
                
                # Let's modify S to store both value and index.
                # S_val[h][k], S_idx[h][k].
                
                pass # Placeholder
            
            # Set dp and par
            # We need to do this carefully.
            
        # Let's restart the inner loop with S_val and S_idx.
        
    # Restart implementation with S_val and S_idx
    
    S_val = [None] * N
    S_idx = [None] * N
    
    for i in range(1, N):
        L = 2 * primes[0:i] - primes[i]
        k_mins = np.searchsorted(primes, L)
        
        for h in range(i):
            km = k_mins[h]
            if km < h:
                val = S_val[h][km]
                idx = S_idx[h][km]
                if val > 0:
                    dp[h, i] = val + 1
                    par[h, i] = idx
                else:
                    dp[h, i] = 2
                    par[h, i] = -1
            else:
                dp[h, i] = 2
                par[h, i] = -1
        
        # Compute S_val[i] and S_idx[i]
        # S_val[i][k] = max(dp[k:i, i])
        # S_idx[i][k] = argmax(dp[k:i, i])
        
        s_val = np.zeros(i, dtype=np.int32)
        s_idx = np.full(i, -1, dtype=np.int32)
        
        max_v = 0
        max_i = -1
        for k in range(i-1, -1, -1):
            v = dp[k, i]
            if v > max_v:
                max_v = v
                max_i = k
            s_val[k] = max_v
            s_idx[k] = max_i
            
        S_val[i] = s_val
        S_idx[i] = s_idx
        
        # Time check
        if time.time() > DEADLINE:
            # Return best found so far
            break
            
    # Find the best length
    if time.time() > DEADLINE:
         # If we broke early, we only have partial dp.
         # But the loop breaks after computing a column.
         # So dp is valid for columns < i.
         pass

    max_len = 0
    best_i = -1
    best_j = -1
    
    # Check all dp[i][j]
    # dp is N x N.
    # We only filled upper triangle.
    for i in range(N):
        for j in range(i+1, N):
            if dp[i, j] > max_len:
                max_len = dp[i, j]
                best_i = i
                best_j = j
                
    # If no pair found, return single prime?
    # But N=1229, so pairs exist.
    # What if max_len is 0? (Should not happen)
    if max_len == 0:
        return [primes_list[0]]
        
    # Reconstruct
    seq = []
    i, j = best_i, best_j
    while True:
        seq.append(primes_list[j])
        if par[i, j] == -1:
            seq.append(primes_list[i])
            break
        j = i
        i = par[i, j]
        
    seq.reverse()
    
    return seq

# EVOLVE-BLOCK-END
