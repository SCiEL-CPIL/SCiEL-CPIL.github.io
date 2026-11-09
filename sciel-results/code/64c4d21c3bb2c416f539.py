"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _sieve_primes(limit):
    """Return numpy array of all primes <= limit."""
    if limit < 2:
        return np.array([], dtype=np.int64)
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return np.where(is_prime)[0]

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    
    # Get all primes <= 10000
    primes = _sieve_primes(10000)
    n = len(primes)
    
    if n == 0:
        return []
    if n == 1:
        return [int(primes[0])]
    
    # DP approach:
    # We want the longest subsequence of primes p_{i1} < p_{i2} < ... < p_{ik}
    # such that gaps d_j = p_{i_{j+1}} - p_{i_j} are nondecreasing: d_1 <= d_2 <= ... <= d_{k-1}
    #
    # State: dp[i][j] = length of longest valid subsequence ending at primes[i] 
    #         with the last gap being exactly primes[i] - primes[j] (where j < i)
    # Base case: a subsequence of length 1 (just primes[i]) has no gap constraint.
    #
    # Transition: dp[i][j] = 1 + max(dp[j][k] for k < j where primes[j] - primes[k] <= primes[i] - primes[j])
    #            or 2 if no such k exists (i.e., just primes[j], primes[i])
    #
    # Answer: max over all i,j of dp[i][j], also considering singletons.
    #
    # To make this efficient, for each j we precompute:
    #   best_len[j][g] = max dp[j][k] over all k < j where primes[j] - primes[k] <= g
    # But gaps can be up to ~10000, so we can use a suffix-max or prefix-max array.
    
    # Actually, let's think differently. For each pair (j, i) with j < i:
    #   gap = primes[i] - primes[j]
    #   dp[i][j] = 2 + max(0, max(dp[j][k] - 1 for k < j where primes[j] - primes[k] <= gap))
    #             = max(2, 1 + max(dp[j][k] for k < j where primes[j] - primes[k] <= gap))
    #
    # For each j, define an array over possible previous gaps. Since primes[j] - primes[k] 
    # ranges over specific values, we can for each j store:
    #   For all k < j: (gap_jk = primes[j] - primes[k], dp[j][k])
    #   Then for a query gap g, we want max dp[j][k] where gap_jk <= g.
    #   This is a prefix maximum query.
    
    # n is about 1229 (primes up to 10000), so n^2 is about 1.5M, which is manageable.
    # But we need to be careful with time.
    
    # Let's use a more direct approach with numpy vectorization where possible.
    
    # dp[i][j] for i > j: length of longest nondecreasing-gap subsequence ending with ..., primes[j], primes[i]
    # dp[i][j] = 2 if no valid predecessor, else 1 + max(dp[j][k] for k < j with primes[j]-primes[k] <= primes[i]-primes[j])
    
    # For efficiency, for each j, precompute a sorted list of (gap, dp_value) for all k < j,
    # then compute prefix max. Then for each i > j, binary search.
    
    # Actually with n=1229, O(n^2) with some constant factor should be fine within 80s.
    # Let's try a straightforward O(n^2) DP with prefix-max optimization.
    
    # dp[j][k] = length of longest valid subsequence ending with primes[k], primes[j] (k < j)
    # We'll compute row by row (for j from 0 to n-1).
    
    # For each j, we need for all k < j: the gap g_jk = primes[j] - primes[k] and dp[j][k].
    # Then for i > j: dp[i][j] = max(2, 1 + max{dp[j][k] : k < j, g_jk <= primes[i] - primes[j]})
    
    # Precompute for each j: sorted gaps and corresponding prefix max dp values.
    
    # Let's store dp as a 2D array or compute on the fly.
    # dp[j][k] depends on dp[k][*], so we need to process in order of j.
    
    # Let's use a list of arrays.
    # dp[j] will be an array of length j (for k=0..j-1), where dp[j][k] is the length.
    
    dp = [None] * n  # dp[j] is a numpy array of length j
    
    # For each j, we'll also store the sorted (gap, prefix_max_dp) for efficient querying.
    # query_data[j] = (sorted_gaps, prefix_max_dp_values)
    
    # Process j from 0 to n-1
    # For j=0: no predecessors, dp[0] is empty (no k < 0)
    # For j=1: dp[1][0] = 2 (just primes[0], primes[1])
    # For j >= 2: dp[j][k] = max(2, 1 + max{dp[k][m] : m < k, primes[k]-primes[m] <= primes[j]-primes[k]})
    
    # Wait, I need to be more careful. dp[j][k] means the subsequence ends with ..., primes[k], primes[j].
    # The gap before primes[j] is primes[j] - primes[k].
    # The gap before primes[k] (if it exists) must be <= primes[j] - primes[k].
    # dp[k][m] is the length ending with primes[m], primes[k], with last gap primes[k]-primes[m].
    # So we need primes[k]-primes[m] <= primes[j]-primes[k], i.e., primes[k]-primes[m] <= gap_jk.
    
    # So: dp[j][k] = 2 if k==0 or no valid m exists
    #         = 1 + max{dp[k][m] : m < k, primes[k]-primes[m] <= primes[j]-primes[k]}
    
    # For each k, we precompute sorted (gap_km, dp[k][m]) pairs and prefix max.
    
    # Let's restructure:
    # For each k (as the "previous" element), we have a list of (gap_km, dp[k][m]) for m < k.
    # We sort by gap_km and compute prefix max of dp[k][m].
    
    # Then for computing dp[j][k], we query: max dp[k][m] where gap_km <= (primes[j] - primes[k]).
    
    # This is O(log k) per query with binary search.
    
    # Total work: sum over j of sum over k<j of O(log k) = O(n^2 log n)
    
    # Let's implement this.
    
    import bisect
    
    # Precompute for each k: sorted gaps and prefix max dp values
    # We'll build these incrementally as we go.
    
    # For each k, store:
    #   k_gaps[k] = sorted array of gaps (primes[k] - primes[m] for m < k)
    #   k_prefmax[k] = prefix max of dp[k][m] corresponding to sorted gaps
    
    k_gaps = [None] * n
    k_prefmax = [None] * n
    
    best_len = 1  # at least one prime
    best_seq = [int(primes[0])]
    
    # We also need to track the actual sequence for reconstruction.
    # prev[j][k] = the m that achieved the max dp[k][m] for dp[j][k], or -1 if none.
    
    prev = [None] * n  # prev[j] is array of length j, prev[j][k] = m or -1
    
    for j in range(n):
        if time.time() > deadline:
            break
        if j == 0:
            continue
        
        dp_j = np.zeros(j, dtype=np.int32)
        prev_j = np.full(j, -1, dtype=np.int32)
        
        for k in range(j):
            gap_jk = int(primes[j] - primes[k])
            
            if k == 0:
                dp_j[k] = 2
                prev_j[k] = -1
            else:
                # Query: max dp[k][m] where primes[k]-primes[m] <= gap_jk
                # k_gaps[k] and k_prefmax[k] are already computed
                gaps_k = k_gaps[k]
                prefmax_k = k_prefmax[k]
                
                # Binary search: find rightmost index where gaps_k[idx] <= gap_jk
                idx = bisect.bisect_right(gaps_k, gap_jk) - 1
                if idx >= 0:
                    best_prev = int(prefmax_k[idx])
                    dp_j[k] = 1 + best_prev
                    # Need to find which m achieved this
                    # For now, store -1 and we'll reconstruct differently
                    # Actually, let's store the index into the sorted array for reconstruction
                    # This is getting complex. Let me just store the best m.
                    # We can find it by scanning, but that's O(k). 
                    # Alternative: store argmax in prefmax computation.
                    prev_j[k] = -1  # placeholder
                else:
                    dp_j[k] = 2
                    prev_j[k] = -1
            
            if dp_j[k] > best_len:
                best_len = int(dp_j[k])
        
        dp[j] = dp_j
        prev[j] = prev_j
        
        # Build k_gaps[j] and k_prefmax[j] for future use
        if j > 0:
            gaps_arr = np.array([int(primes[j] - primes[m]) for m in range(j)], dtype=np.int64)
            # Sort by gap, keeping track of original index
            order = np.argsort(gaps_arr, kind='stable')
            sorted_gaps = gaps_arr[order]
            dp_vals = dp_j[order]
            
            # Prefix max
            prefmax = np.maximum.accumulate(dp_vals)
            
            k_gaps[j] = sorted_gaps
            k_prefmax[j] = prefmax
    
    # Now reconstruct the best sequence.
    # We need to find (i, j) that maximizes dp[i][j].
    # Then backtrack.
    
    # Actually, let's redo with proper backtracking info.
    # This is getting complex. Let me use a cleaner approach.
    
    # Given time constraints, let me just return the length-based answer.
    # But we need to return the actual primes.
    
    # Let me redo this more carefully with a clean implementation.
    
    # Restart with a cleaner approach
    primes_list = primes.tolist()
    n = len(primes_list)
    
    if n <= 1:
        return primes_list
    
    # dp[i][j] = length of longest valid subsequence ending at primes[i] with previous being primes[j]
    # We'll use a dict or 2D list. n=1229, so 2D list is fine.
    
    # For backtracking: parent[i][j] = the index m such that dp[j][m] was used to compute dp[i][j]
    
    # Let's use the O(n^2 log n) approach properly.
    
    # For each index t, we maintain:
    #   For all s < t: gap_t_s = primes[t] - primes[s], dp[t][s]
    #   Sorted by gap_t_s, with prefix max of dp[t][s]
    
    # Process t from 0 to n-1.
    # When processing t, for each s < t:
    #   dp[t][s] = 2 if s == 0
    #            = 1 + max{dp[s][r] : r < s, primes[s]-primes[r] <= primes[t]-primes[s]} if s > 0
    #            = 2 if no valid r
    
    # We need to query the data structure for index s.
    
    # Let's store for each s: a list of (gap, dp_val, argmax_index) sorted by gap, with prefix max.
    
    # Actually, for backtracking we need to know which r gave the max.
    # Let's store: for each s, an array of (gap_sr, dp[s][r], r) sorted by gap_sr.
    # Then prefix max of dp[s][r], and also prefix argmax.
    
    # This is doable but let me just do it with Python lists for clarity, since n~1229.
    
    # dp[s][r] for r < s
    # Store as: dp_data[s] = list of (gap, dp_val, r) for r in range(s)
    
    # Let me just do a clean O(n^2) without the log factor, since n^2 = 1.5M and each inner op is O(1) amortized.
    
    # Alternative clean approach:
    # For each pair (s, t) with s < t, gap = primes[t] - primes[s].
    # dp[t][s] = 2 + max(0, max{dp[s][r] - 1 for r < s where primes[s]-primes[r] <= gap})
    #           = max(2, 1 + max{dp[s][r] for r < s where primes[s]-primes[r] <= gap})
    
    # For each s, precompute: for all possible gaps g, best_dp_s[g] = max{dp[s][r] for r < s where primes[s]-primes[r] <= g}
    # Since max gap is ~10000, we can use an array of size 10001.
    
    # This gives O(n * max_gap + n^2) = O(n * 10000 + n^2) which is about 12M + 1.5M = 13.5M. Fine.
    
    max_gap = 10000
    # best_dp_s[s][g] = max dp[s][r] for r < s with primes[s]-primes[r] <= g
    # We'll compute this as we go.
    
    # dp[t][s] stored in a 2D structure. Let's use a list of lists.
    # dp_table[t] = list of length t, dp_table[t][s] = dp value for ending at t with prev s.
    
    dp_table = [None] * n
    parent = [None] * n  # parent[t][s] = r that achieved the max
    
    # For each s, we maintain an array best_so_far_s of size max_gap+1
    # best_so_far_s[g] = max dp[s][r] for r < s with gap_sr <= g
    # We build this incrementally.
    
    # Actually, let's build it when we finish computing row s.
    
    # Process s from 0 to n-1:
    #   When we finish computing dp_table[s] (for all r < s), we build best_so_far_s.
    
    best_so_far = [None] * n  # best_so_far[s] = array of size max_gap+1
    
    for s in range(n):
        if time.time() > deadline:
            break
        if s == 0:
            dp_table[s] = []
            best_so_far[s] = None
            continue
        
        dp_s = [0] * s
        parent_s = [-1] * s
        
        for r in range(s):
            gap_sr = primes_list[s] - primes_list[r]
            
            if r == 0:
                dp_s[r] = 2
                parent_s[r] = -1
            else:
                # Query best_so_far[r] at index gap_sr
                bsf_r = best_so_far[r]
                if bsf_r is not None and gap_sr <= max_gap:
                    val = int(bsf_r[gap_sr])
                    if val > 0:
                        dp_s[r] = 1 + val
                        # Need to find which m achieved this
                        # This requires storing argmax, which complicates things.
                        # For now, let's just get the length and reconstruct separately.
                        parent_s[r] = -1  # placeholder
                    else:
                        dp_s[r] = 2
                        parent_s[r] = -1
                else:
                    dp_s[r] = 2
                    parent_s[r] = -1
        
        dp_table[s] = dp_s
        parent[s] = parent_s
        
        # Build best_so_far[s]
        if s > 0:
            bsf = np.zeros(max_gap + 1, dtype=np.int32)
            # For each r < s: gap_sr = primes[s]-primes[r], value = dp_s[r]
            # bsf[g] = max dp_s[r] for r < s with gap_sr <= g
            # We can do this by: for each r, update bsf[gap_sr:] = max(bsf[gap_sr:], dp_s[r])
            # But that's O(s * max_gap) which is too slow.
            # Better: create an array of (gap, value) pairs, sort by gap, then prefix max.
            
            gaps = [primes_list[s] - primes_list[r] for r in range(s)]
            vals = dp_s
            
            # Sort by gap
            order = sorted(range(s), key=lambda x: gaps[x])
            sorted_gaps = [gaps[i] for i in order]
            sorted_vals = [vals[i] for i in order]
            
            # Build prefix max
            prefmax = [0] * (max_gap + 1)
            current_max = 0
            gap_idx = 0
            for g in range(max_gap + 1):
                while gap_idx < s and sorted_gaps[gap_idx] <= g:
                    if sorted_vals[gap_idx] > current_max:
                        current_max = sorted_vals[gap_idx]
                    gap_idx += 1
                prefmax[g] = current_max
            
            best_so_far[s] = np.array(prefmax, dtype=np.int32)
    
    # Find the best (t, s) pair
    best_val = 1
    best_t = 0
    best_s = -1
    for t in range(n):
        if dp_table[t] is None:
            continue
        for s in range(t):
            if dp_table[t][s] > best_val:
                best_val = dp_table[t][s]
                best_t = t
                best_s = s
    
    # Reconstruct sequence
    # We need proper parent pointers. Let me redo with proper backtracking.
    # This is getting too complex for the time budget. Let me just return a valid sequence.
    
    # Actually, let me just use a simpler but correct approach.
    # Given the complexity, let me use a different strategy:
    # Since n=1229, we can do O(n^2) DP with a simple inner loop.
    
    # Let me restart with a clean, correct O(n^2) solution.
    
    # Clear previous work
    dp_table = [None] * n
    parent = [None] * n
    
    # dp[t][s] = length of longest valid subsequence ending at primes[t] with immediate predecessor primes[s]
    # parent[t][s] = the index r < s such that dp[s][r] was maximized, or -1
    
    # For each s, we need to answer: given gap g, what's the max dp[s][r] for r < s with primes[s]-primes[r] <= g?
    # And which r achieved it?
    
    # Let's store for each s: a sorted list of (gap_sr, dp[s][r], r) and prefix max with argmax.
    
    # Process t from 0 to n-1:
    
    # For s = 0: no predecessors
    # For s >= 1: compute dp[s][r] for all r < s
    
    # Store: s_data[s] = (sorted_gaps, sorted_dp, sorted_r, prefix_max_dp, prefix_argmax_r)
    
    s_data = [None] * n
    
    for s in range(n):
        if time.time() > deadline:
            break
        if s == 0:
            s_data[s] = None
            continue
        
        dp_s = [0] * s
        parent_s = [-1] * s
        
        for r in range(s):
            gap_sr = primes_list[s] - primes_list[r]
            
            if r == 0:
                dp_s[r] = 2
                parent_s[r] = -1
            else:
                data_r = s_data[r]
                if data_r is None:
                    dp_s[r] = 2
                    parent_s[r] = -1
                else:
                    sorted_gaps_r, prefix_max_dp_r, prefix_argmax_r = data_r
                    # Binary search for gap_sr
                    idx = bisect.bisect_right(sorted_gaps_r, gap_sr) - 1
                    if idx >= 0:
                        best_dp = prefix_max_dp_r[idx]
                        if best_dp > 0:
                            dp_s[r] = 1 + best_dp
                            parent_s[r] = prefix_argmax_r[idx]
                        else:
                            dp_s[r] = 2
                            parent_s[r] = -1
                    else:
                        dp_s[r] = 2
                        parent_s[r] = -1
        
        dp_table[s] = dp_s
        parent[s] = parent_s
        
        # Build s_data[s]
        if s > 0:
            gaps_list = [(primes_list[s] - primes_list[r], dp_s[r], r) for r in range(s)]
            gaps_list.sort(key=lambda x: x[0])
            sorted_gaps = [g for g, _, _ in gaps_list]
            dp_vals = [v for _, v, _ in gaps_list]
            r_vals = [r for _, _, r in gaps_list]
            
            # Prefix max with argmax
            prefix_max = [0] * s
            prefix_argmax = [-1] * s
            current_max = 0
            current_arg = -1
            for i in range(s):
                if dp_vals[i] > current_max:
                    current_max = dp_vals[i]
                    current_arg = r_vals[i]
                prefix_max[i] = current_max
                prefix_argmax[i] = current_arg
            
            s_data[s] = (sorted_gaps, prefix_max, prefix_argmax)
    
    # Find best
    best_val = 1
    best_t = 0
    best_s = -1
    for t in range(n):
        if dp_table[t] is None:
            continue
        for s in range(t):
            if dp_table[t][s] > best_val:
                best_val = dp_table[t][s]
                best_t = t
                best_s = s
    
    # Reconstruct
    if best_s == -1:
        return [primes_list[best_t]]
    
    seq = [best_t, best_s]
    cur_t = best_t
    cur_s = best_s
    while parent[cur_t] is not None and parent[cur_t][cur_s] != -1:
        cur_s = parent[cur_t][cur_s]
        seq.append(cur_s)
        # Find the t that has this s as predecessor
        # This is tricky. We need to know which t used s as predecessor.
        # Actually, parent[t][s] gives the r for the subsequence ending at t with prev s.
        # So the chain is: t -> s -> parent[t][s] -> ...
        # But we need to know the full chain of indices.
    
    # Hmm, the backtracking is complex. Let me just return the best length as a simple sequence.
    # Actually, let me just return a valid nondecreasing-gap sequence of the found length.
    
    # Given the complexity, let me just return the greedy result if we can't reconstruct properly.
    # But that would score 1.0. Let me try to fix the reconstruction.
    
    # The issue is: when we have dp[t][s] and parent[t][s] = r, the sequence is:
    #   ..., primes[r], primes[s], primes[t]
    # And dp[s][r] gives the length up to primes[s] with prev primes[r].
    # So we need to continue backtracking from (s, r).
    
    # But we need to know which "t" corresponds to s in the next step. It's just s.
    # The chain of indices is: t, s, r, ...
    # At each step, we have (current_t, current_s) and parent[current_t][current_s] = next_s.
    # Then the new pair is (current_s, next_s).
    
    seq = [best_t, best_s]
    ct = best_t
    cs = best_s
    while parent[ct] is not None and parent[ct][cs] != -1:
        ns = parent[ct][cs]
        seq.append(ns)
        ct = cs
        cs = ns
    
    seq.reverse()
    result = [primes_list[i] for i in seq]
    
    # Verify
    if len(result) >= 2:
        gaps = [result[i+1] - result[i] for i in range(len(result)-1)]
        if not all(gaps[i] <= gaps[i+1] for i in range(len(gaps)-1)):
            # Fallback to simple valid sequence
            result = [primes_list[0]]
    
    return result

# EVOLVE-BLOCK-END
