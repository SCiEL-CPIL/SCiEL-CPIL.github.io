import bisect
import time

def construct():
    start_time = time.time()
    DEADLINE = 55.0 # seconds
    
    # Sieve of Eratosthenes for primes <= 10000
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    L = len(primes)
    
    if L < 2:
        return primes
        
    # dp[i][j] = max length of valid subsequence ending with primes[i], primes[j] (i < j)
    # prev[i][j] = index k such that the sequence ending at i, j was extended from k, i
    # If prev[i][j] == -1, the sequence is just [primes[i], primes[j]]
    
    dp = [[0] * L for _ in range(L)]
    prev = [[-1] * L for _ in range(L)]
    
    best_len = 2
    best_end_j = -1
    best_end_i = -1
    
    # Iterate through the second-to-last element 'i' (which is 'b' in my thought process)
    # We fill dp[i][j] for j > i
    # The recurrence for dp[i][j] depends on dp[h][i] for h < i.
    # So we should iterate i from 1 to L-2?
    # Actually, dp[i][j] is computed when we are at step 'i' as the second element?
    # No, let's iterate the second element index 'i' from 1 to L-1.
    # For a fixed 'i', we compute dp[i][j] for all j > i.
    # This requires dp[h][i] for h < i to be already computed.
    # Since we iterate i in increasing order, dp[h][i] (where h < i) was computed in previous iterations of the outer loop?
    # Wait.
    # If outer loop is 'i', and we compute dp[i][j], we need dp[h][i].
    # dp[h][i] was computed when the outer loop variable was 'h'?
    # No, dp[h][i] has 'i' as the second index.
    # In my previous thought: "Loop b (second last) from 1 to L-1".
    # If outer loop is 'b' (let's call it 'i' now), we compute dp[i][j] for j > i.
    # To compute dp[i][j], we need max dp[h][i] for h < i.
    # These dp[h][i] values were computed when the outer loop was at 'h'?
    # No, dp[h][i] is an entry in the column 'i' of the matrix?
    # Let's check indices. dp[row][col] ends at primes[row], primes[col].
    # So dp[h][i] ends at primes[h], primes[i].
    # This entry is computed when we processed the pair (h, i).
    # If we iterate the second index 'i' from 1 to L-1, and for each 'i' we iterate the third index 'j' from i+1 to L-1...
    # We need dp[h][i] to be available.
    # dp[h][i] is computed when the "second index" was 'i'? No.
    # dp[h][i] is computed when we considered 'i' as the *second* element of the pair ending the sequence?
    # Yes. If we define the outer loop variable as the index of the *second* element in the pair (let's call it `mid`),
    # and inner loop as the index of the *third* element (`last`),
    # then we are computing dp[mid][last].
    # This depends on dp[prev][mid].
    # dp[prev][mid] was computed when `mid` was the `last` element in a previous iteration?
    # No. dp[prev][mid] has `mid` as the second index.
    # So dp[prev][mid] is computed in the iteration where the outer loop variable was `prev`?
    # No, if outer loop is `mid`, we compute dp[mid][last].
    # We need dp[prev][mid].
    # dp[prev][mid] was computed when the outer loop variable was `prev`?
    # If outer loop is `k`, we compute dp[k][j] for j > k.
    # So dp[prev][mid] is computed when outer loop was `prev` and inner loop was `mid`.
    # Since `prev < mid`, and we iterate `mid` (outer) in increasing order,
    # when we are at outer loop `mid`, all `dp[prev][mid]` for `prev < mid` have already been computed
    # in the iterations where outer loop was `prev` (and inner loop was `mid`).
    
    # So the algorithm is:
    # for mid in range(1, L):
    #    # Prepare suffix max of dp[0..mid-1][mid]
    #    # for last in range(mid+1, L):
    #    #    compute dp[mid][last]
    
    for mid in range(1, L):
        # Build suffix max and argmax for column 'mid' of dp, rows 0 to mid-1
        # vals[k] = dp[k][mid]
        # We want max(dp[k][mid]) for k in [h_min, mid-1]
        
        # If mid == 1, only k=0 is possible.
        # If mid > 1, we have a range.
        
        # Optimization: 
        # We can just iterate k from mid-1 down to 0.
        # suf_max[k] = max(dp[k][mid], suf_max[k+1])
        # suf_idx[k] = k if dp[k][mid] > dp[suf_idx[k+1]] else suf_idx[k+1]
        
        suf_max = [0] * (mid + 1) # suf_max[mid] is effectively -inf or 0? 
        suf_idx = [-1] * (mid + 1)
        
        # Initialize
        # suf_max[mid] = 0 (no elements)
        # suf_idx[mid] = -1
        
        current_max = 0
        current_idx = -1
        
        for k in range(mid - 1, -1, -1):
            if dp[k][mid] > current_max:
                current_max = dp[k][mid]
                current_idx = k
            suf_max[k] = current_max
            suf_idx[k] = current_idx
            
        # Now compute dp[mid][last] for last > mid
        for last in range(mid + 1, L):
            # Condition: primes[last] - primes[mid] >= primes[mid] - primes[k]
            # primes[k] >= 2 * primes[mid] - primes[last]
            target = 2 * primes[mid] - primes[last]
            
            # Find smallest k in [0, mid-1] such that primes[k] >= target
            # bisect_left on primes[0:mid]
            k_min = bisect.bisect_left(primes, target, 0, mid)
            
            if k_min < mid:
                # Valid k exists in [k_min, mid-1]
                # Best is at suf_idx[k_min]
                prev_idx = suf_idx[k_min]
                if prev_idx != -1:
                    new_len = dp[prev_idx][mid] + 1
                    dp[mid][last] = new_len
                    prev[mid][last] = prev_idx
                else:
                    # Should not happen if k_min < mid and mid > 0
                    dp[mid][last] = 2
                    prev[mid][last] = -1
            else:
                # No valid k, sequence starts here
                dp[mid][last] = 2
                prev[mid][last] = -1
                
            if dp[mid][last] > best_len:
                best_len = dp[mid][last]
                best_end_j = last
                best_end_i = mid
                
            # Deadline check
            if time.time() > DEADLINE:
                # Reconstruct and return
                break
                
        if time.time() > DEADLINE:
            break

    # If best_len is still 2 and we didn't find a better one, return a pair or single?
    # The problem asks for increasing prime integers. Length 1 is valid?
    # "successive gaps NONDECREASING". A sequence of length 1 has no gaps.
    # A sequence of length 2 has 1 gap.
    # The seed returned a list.
    # If best_len is 2, we return the pair.
    
    if best_end_j == -1:
        # Fallback
        return primes[:min(2, L)]
        
    # Reconstruct
    seq = [primes[best_end_j], primes[best_end_i]]
    i, j = best_end_i, best_end_j
    while prev[i][j] != -1:
        h = prev[i][j]
        seq.append(primes[h])
        i, j = h, i
        
    seq.reverse()
    return seq

# EVOLVE-BLOCK-END
