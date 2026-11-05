# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Sieve for primes up to 10000
    limit = 10000
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    primes = [p for p in range(2, limit + 1) if sieve[p]]
    n = len(primes)
    
    if n <= 1:
        return primes
    
    # dp[j] = list of (gap, max_length) for sequences ending at prime j
    # We'll use: best[j] = array where best[j][g] = max length of seq ending at j with last gap <= g
    # But gaps can be large. Instead, for each j, store dict: gap -> max length, then compute prefix max.
    
    # Actually, let's use: for each j, we store a sorted list of (gap_value, best_length_so_far)
    # where best_length_so_far is the max length among all sequences ending at j with last gap <= gap_value.
    
    # Simpler: dp_len[i] = max length of valid sequence ending at prime i
    # dp_prev[i] = the index of the previous prime in the optimal sequence ending at i
    # But we need to know the last gap to enforce nondecreasing gaps.
    
    # Let's do: for each i, for each j < i, the gap g = primes[i] - primes[j]
    # We need the best sequence ending at j with last gap <= g.
    
    # For each j, precompute: for all k < j, gap_kj = primes[j] - primes[k], and dp[k->j] = best length ending at j via k.
    # Then for transition j->i with gap g_ji, we need max over k<j with gap_kj <= g_ji of (dp[k->j] + 1)
    
    # This is getting complex. Let me just do O(n^3) with early termination if too slow.
    # n = 1229, n^3 = 1.86e9. In Python, this is too slow.
    
    # Alternative: O(n^2 * log(n)) using binary search on sorted gaps.
    # For each j, collect all (gap_kj, dp_len_via_k_to_j) for k < j.
    # Sort by gap_kj. Compute prefix max of dp_len.
    # Then for transition j->i, binary search for the largest gap_kj <= g_ji, get prefix max.
    
    # dp[j][k] = max length of valid sequence ending at j, with previous prime being k.
    # We want for each j, a structure that given a gap threshold G, returns max over k<j with (primes[j]-primes[k]) <= G of dp[j][k].
    
    # Let's store for each j: list of (gap_kj, dp_jk) sorted by gap_kj, and prefix max array.
    
    import bisect
    
    # dp[j][k] for k < j
    # Initialize: sequences of length 2: dp[j][k] = 2 for all k < j
    # Then for i > j: dp[i][j] = 1 + max{dp[j][k] : k < j and primes[j]-primes[k] <= primes[i]-primes[j]}
    #             = 1 + 2 = 3 if no such k, but actually if no k, then sequence is just [k, j, i] with gap_kj <= gap_ji
    #             Wait, if no k exists satisfying the condition, then we can still have sequence [j, i] of length 2.
    #             But dp[i][j] should be at least 2 (the sequence [j, i]).
    #             Actually, dp[i][j] = 1 + (max over valid k of dp[j][k]) if any valid k exists, else 2.
    
    # Let me redefine: best_end[j] = for each possible last gap, the max length.
    # Actually, let's just track for each j, a list of (gap, length) pairs and compute prefix max.
    
    # For j from 0 to n-1:
    #   For each k < j: gap = primes[j] - primes[k], length = dp[j][k]
    #   Sort these by gap, compute prefix max
    #   Store this structure for j
    
    # Then for i from 0 to n-1:
    #   For each j < i: gap_ji = primes[i] - primes[j]
    #     Look up in j's structure: max length among sequences ending at j with last gap <= gap_ji
    #     dp[i][j] = 1 + that max (or 2 if no such sequence)
    #   best_len[i] = max over j < i of dp[i][j] (or 1 if i==0)
    
    # This is O(n^2 log n) time and space.
    
    # Let's implement:
    
    # For each j, we'll build: sorted_gaps[j] = sorted list of gaps from j to earlier primes
    #                  prefix_max[j] = prefix max of dp[j][k] in that sorted order
    #                  But we need dp[j][k] which depends on earlier computations.
    
    # Process primes in order. When processing i, all j < i have their structures built.
    
    # dp[i][j] = 2 if no k < j with gap_kj <= gap_ji and dp[j][k] exists
    #             = 1 + max{dp[j][k] : k < j, gap_kj <= gap_ji} otherwise
    
    # But dp[j][k] is the length of sequence ending at j with previous prime k.
    # The sequence ending at j via k has length dp[j][k].
    # Appending i gives length dp[j][k] + 1.
    # If no valid k, the sequence is just [j, i] with length 2.
    
    # So dp[i][j] = 1 + max(2, max{dp[j][k] : valid k})? No.
    # If there's a valid k, dp[i][j] = 1 + max{dp[j][k] : valid k}
    # If no valid k, dp[i][j] = 2 (sequence [j, i])
    
    # But dp[j][k] >= 2 always (since it's a sequence of at least 2 elements ending at j).
    # So 1 + dp[j][k] >= 3.
    # And 2 < 3, so if any valid k exists, we prefer using it.
    
    # So: dp[i][j] = 2 if no valid k, else 1 + max{dp[j][k] : valid k}
    
    # Let me code this up.
    
    # For memory, dp is n x n = 1229^2 = 1.5M entries. Fine.
    
    dp = [[0] * n for _ in range(n)]  # dp[i][j] = length of best seq ending at i with prev prime j
    best_len = [1] * n  # best_len[i] = max length of valid seq ending at i
    
    # For each j, we need a data structure to query: max dp[j][k] for k < j with gap_kj <= G
    # We'll build this on the fly as we process i.
    # Actually, when we finish processing all i > j, we can build j's structure.
    # But we need j's structure when processing i > j. So we build it after computing all dp[i][j] for i > j? No, we need it during.
    
    # Alternative: process i from 0 to n-1. When processing i, for each j < i, we need j's structure.
    # j's structure depends on dp[j][k] for k < j, which are already computed (since k < j < i).
    # So we can build j's structure lazily when first needed, or precompute after computing all dp[j][*].
    
    # Let's precompute: after computing all dp[j][k] for k < j, build j's query structure.
    # We process j in order. When we're about to process i = j+1, all dp[j][k] for k < j are known.
    # So we can build j's structure at the start of processing i = j+1.
    
    # Actually, let's just build all structures upfront in a loop.
    
    # Phase 1: Compute dp[i][j] for all i > j
    # For i from 1 to n-1:
    #   For j from 0 to i-1:
    #     gap_ji = primes[i] - primes[j]
    #     Query j's structure for max dp[j][k] with gap_kj <= gap_ji
    #     dp[i][j] = 2 if no result, else 1 + result
    #     Update best_len[i]
    
    # But j's structure needs to be built before we query it.
    # j's structure depends on dp[j][k] for k < j, which are computed in earlier iterations of i.
    # So we can build j's structure after finishing the inner loop for i = j (i.e., after all dp[j][k] for k < j are computed).
    # Wait, dp[j][k] is computed when i = j and we process k < j. So after the inner loop for i = j completes, all dp[j][k] are known.
    
    # Let me restructure:
    # For i from 0 to n-1:
    #   If i > 0:
    #     For j from 0 to i-1:
    #       gap_ji = primes[i] - primes[j]
    #       # Query j's structure
    #       ...
    #   # After processing all j < i, build i's structure for future use
    #   # i's structure: for all k < i, gap_ki = primes[i] - primes[k], value = dp[i][k]
    #   # Sort by gap_ki, compute prefix max
    
    # This works!
    
    # Data structures:
    # For each i, after computing dp[i][*]:
    #   gaps_i = sorted list of (gap_ki, dp[i][k]) for k < i
    #   prefix_max_i = prefix max array
    #   We store these for query.
    
    # Query for j with threshold G:
    #   Find rightmost index in gaps_j where gap <= G
    #   Return prefix_max_j at that index, or None if no such gap
    
    # Let's implement with lists and bisect.
    
    # Store for each j: sorted_gaps[j], prefix_max[j]
    sorted_gaps = [None] * n
    prefix_max = [None] * n
    
    for i in range(n):
        if i > 0:
            for j in range(i):
                gap_ji = primes[i] - primes[j]
                # Query j's structure
                if sorted_gaps[j] is not None:
                    sg = sorted_gaps[j]
                    pm = prefix_max[j]
                    # Find rightmost index where sg[idx][0] <= gap_ji
                    # sg is sorted by gap
                    # Use bisect_right on the gap values
                    # sg is list of (gap, dp_val). We need to search on gap.
                    # Let's store separate list of gap values for bisect.
                    pass  # Need to restructure
        
        # Build i's structure
        if i > 0:
            pairs = []
            for k in range(i):
                gap_ki = primes[i] - primes[k]
                pairs.append((gap_ki, dp[i][k]))
            pairs.sort(key=lambda x: x[0])
            sorted_gaps[i] = pairs
            # Compute prefix max
            pm = [0] * len(pairs)
            pm[0] = pairs[0][1]
            for idx in range(1, len(pairs)):
                pm[idx] = max(pm[idx-1], pairs[idx][1])
            prefix_max[i] = pm
    
    # This doesn't work because we need to query before building. Let me fix the logic.
    
    # Actually, the above loop builds i's structure AFTER processing i, but queries j's structure DURING processing i.
    # j < i, so j's structure was built in an earlier iteration. That's correct!
    # The issue is just the implementation of the query.
    
    # Let me redo with proper query.
    
    # Reset
    dp = [[0] * n for _ in range(n)]
    best_len = [1] * n
    sorted_gaps = [None] * n
    prefix_max = [None] * n
    gap_values = [None] * n  # For bisect
    
    for i in range(n):
        if i > 0:
            for j in range(i):
                gap_ji = primes[i] - primes[j]
                if gap_values[j] is not None:
                    gv = gap_values[j]
                    pm = prefix_max[j]
                    # Find rightmost index where gv[idx] <= gap_ji
                    idx = bisect.bisect_right(gv, gap_ji) - 1
                    if idx >= 0:
                        dp[i][j] = 1 + pm[idx]
                    else:
                        dp[i][j] = 2
                else:
                    # j has no predecessors, so no valid k
                    dp[i][j] = 2
                
                if dp[i][j] > best_len[i]:
                    best_len[i] = dp[i][j]
        
        # Build i's structure
        if i > 0:
            pairs = []
            for k in range(i):
                gap_ki = primes[i] - primes[k]
                pairs.append((gap_ki, dp[i][k]))
            pairs.sort(key=lambda x: x[0])
            gap_values[i] = [p[0] for p in pairs]
            sorted_gaps[i] = pairs
            pm = [0] * len(pairs)
            pm[0] = pairs[0][1]
            for idx in range(1, len(pairs)):
                pm[idx] = max(pm[idx-1], pairs[idx][1])
            prefix_max[i] = pm
    
    # Find the best sequence
    max_len = max(best_len)
    end_i = best_len.index(max_len)
    
    # Reconstruct the sequence
    seq = [primes[end_i]]
    cur = end_i
    while len(seq) < max_len:
        # Find j < cur such that dp[cur][j] == best_len[cur] and gap is nondecreasing
        # Actually, we need to find the j that gives the optimal length
        target_len = best_len[cur]
        found = False
        for j in range(cur):
            if dp[cur][j] == target_len:
                # Check that the gap from j to cur is >= the gap before j in the sequence
                # Actually, dp[cur][j] already ensures validity
                seq.append(primes[j])
                cur = j
                found = True
                break
        if not found:
            break
    
    seq.reverse()
    return seq

# EVOLVE-BLOCK-END
