"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from math import isqrt

    deadline = time.time() + 55.0  # safety margin before 60s hard kill

    # ---- Sieve of primes up to 10000 ----
    N = 10000
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start::step] = [False] * (((N - start) // step) + 1)
    primes = [i for i, v in enumerate(is_prime) if v]
    prime_set = set(primes)
    prime_index = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)

    # ---- Beam search for longest sequence with nondecreasing gaps ----
    # State: (last_prime_index, last_gap, sequence_length, last_prime)
    # We track sequences by their last prime and last gap.
    # For efficiency, we use a beam approach: keep top K candidates per "state type"

    # Initial: each prime can start a sequence
    # State represented as (last_prime, last_gap, length)
    # last_gap = -1 means no previous gap (first element)
    
    # Use beam search: maintain a set of active states
    # To keep it manageable, we'll use a different approach:
    # Dynamic programming with beam pruning
    
    # Let's use a greedy + local search approach
    # First, try multiple greedy strategies with different starting points
    
    best_len = 0
    best_seq = []
    
    # Try greedy from each starting prime with various gap constraints
    # For each starting prime p_i, try to extend greedily
    # but also try to "restart" the gap requirement at various points
    
    # More systematic: beam search where we keep top-K partial sequences
    # sorted by (length, last_gap) - prefer longer sequences, and for same length, smaller last_gap
    
    BEAM_WIDTH = 50000
    
    # States: list of (last_prime_idx, last_gap, length)
    # Initialize with all primes as length-1 sequences
    # last_gap = 0 for single element (no constraint yet)
    
    # Actually, for a sequence of length 1, there's no gap constraint.
    # For length >= 2, the last gap matters.
    # Let's define: state = (last_prime_idx, last_gap, length)
    # where last_gap is the gap from the second-to-last to the last prime.
    # For length 1, last_gap = 0 (no constraint).
    
    # To make this efficient, note that the constraint is:
    # next_prime - last_prime >= last_gap
    # So for a given last_prime and last_gap, we want primes >= last_prime + last_gap
    
    # Let's use a different representation for speed:
    # Group states by (last_prime_idx, last_gap) and keep max length
    # But last_gap can be large, so this might be sparse.
    
    # Alternative: iterate through primes in order, maintaining best states.
    # For each prime p_j, for each state ending at p_i < p_j with gap g,
    # if p_j - p_i >= g, we can extend to (p_j, p_j - p_i, length+1)
    
    # This is O(num_primes^2 * beam_width) which might be too slow.
    
    # Let's use a smarter approach: for each prime, maintain the best (max length) 
    # for each possible last_gap value. Since gaps are bounded by N, this is feasible.
    
    # Actually, let's just do a well-optimized beam search:
    
    # State: (last_prime, last_gap, length)
    # We'll maintain a list of states, and for each new prime, try to extend all states.
    # Prune to keep only top BEAM_WIDTH states by (length, -last_gap)
    
    # But this is still O(num_primes * beam_width) per prime = O(num_primes^2 * beam_width)
    
    # Let's try a different angle: since N=10000 is small, maybe we can do DP more cleverly.
    
    # Key insight: the gap sequence is nondecreasing. So if we fix the starting prime and
    # the first gap, the rest is somewhat determined.
    
    # Let's try: for each pair of starting primes (p_i, p_j) with i < j,
    # the first gap is g = p_j - p_i. Then greedily extend:
    # next prime must be >= p_j + g, and the new gap is next - p_j >= g.
    # To maximize length, pick the smallest valid next prime each time.
    
    # This gives O(num_primes^2 * avg_length) which for 1229 primes is ~1.5M * ~20 = 30M, feasible.
    
    # But we can do better: for each starting prime p_i and first gap g,
    # the sequence is determined by greedily picking the smallest prime >= current + g.
    
    # Let's precompute for each prime index i and gap g, the next prime index.
    # next_prime_ge[i][g] = smallest index j > i such that primes[j] >= primes[i] + g
    
    # This is a 2D table of size num_primes x N, which is 1229 * 10000 = 12M entries.
    # We can compute this efficiently.
    
    # Actually, for a given i and g, we need the smallest j > i with primes[j] >= primes[i] + g.
    # We can binary search for this.
    
    import bisect
    
    # For each starting pair (i, j) with i < j:
    # gap = primes[j] - primes[i]
    # Then greedily extend from j with minimum gap = gap
    # Count the length
    
    # To speed up, precompute next_idx[i][g] for all i and relevant g values.
    # But g can range from 1 to N, so let's just use binary search on the fly.
    
    # Optimize: for each i, precompute for all possible target values the next prime index.
    # Actually, let's just iterate and use binary search.
    
    # Even better: for each starting prime i, iterate over possible first gaps.
    # The first gap determines j = smallest prime index > i with primes[j] >= primes[i] + g.
    # But g = primes[j] - primes[i], so j is determined by the choice of next prime.
    
    # So: for each i, for each j > i:
    #   g = primes[j] - primes[i]
    #   Extend greedily from (i, j) with min_gap = g
    #   Count length
    
    # To make extension fast, precompute for each prime index k and gap g,
    # the next prime index >= k+1 with prime >= primes[k] + g.
    
    # Let's build next_idx[k] as a function of g using binary search.
    
    # For speed, let's precompute next_prime_idx[k][g] for g in a limited range.
    # Actually, gaps in optimal sequences are likely small. Let's cap g at some value.
    
    # Let's just do the O(num_primes^2 * log(num_primes)) approach with early termination.
    
    # For each pair (i, j), extend greedily and count.
    # Use binary search to find next prime.
    
    # Precompute: for each prime index k, and for each possible gap g,
    # next_idx[k][g] = smallest index > k with primes[idx] >= primes[k] + g
    
    # Build this table:
    # For each k, for g from 1 to N:
    #   target = primes[k] + g
    #   next_idx[k][g] = bisect.bisect_left(primes, target, k+1)
    
    # This is num_primes * N = 1229 * 10000 = 12.29M lookups, each O(log n).
    # Total: ~12M * 11 = 132M operations. Might be slow in Python.
    
    # Let's optimize: for each k, we can compute next_idx[k][g] more efficiently.
    # As g increases, target increases, so next_idx is non-decreasing.
    # We can use a two-pointer approach.
    
    # For fixed k, let g range from 1 to N.
    # target = primes[k] + g
    # We want smallest j > k with primes[j] >= target.
    # As g increases by 1, target increases by 1.
    # j can only stay the same or increase.
    
    # So for each k, we can compute all next_idx[k][g] in O(num_primes + N) time.
    
    # Total: O(num_primes * (num_primes + N)) = O(1229 * (1229 + 10000)) = ~14M. Good.
    
    # Let's build this table.
    
    # next_idx[k][g] for k in 0..num_primes-1, g in 1..N
    # Store as a list of lists.
    
    # Actually, to save memory, we can compute on the fly per k.
    
    # Let's restructure: for each starting index i:
    #   For each j > i:
    #     g = primes[j] - primes[i]
    #     Extend greedily from j with min_gap = g
    #     To extend: current = j, current_gap = g
    #     Next: find smallest k > current with primes[k] >= primes[current] + current_gap
    #     new_gap = primes[k] - primes[current]
    #     current = k, current_gap = new_gap
    #     Continue until no more primes
    
    # For this, we need fast "next prime >= target" queries.
    # We can use bisect.bisect_left on the primes list.
    
    # Each extension step is O(log num_primes).
    # Total work: sum over all (i,j) pairs of the length of the greedy sequence.
    # Worst case: O(num_primes^2 * num_primes) = O(1229^3) = 1.86B. Too slow.
    
    # We need to prune. Note that for a given i and g, the greedy sequence is unique.
    # So we can memoize: for (current_idx, current_gap), the length of the greedy extension.
    
    # But current_gap varies widely.
    
    # Alternative: since the gap is nondecreasing and we're doing greedy (smallest valid next),
    # the sequence is determined by (start_idx, first_gap).
    # For a fixed start_idx i, as first_gap increases, the sequences are related.
    
    # Let's try a different approach: dynamic programming.
    # dp[i][g] = max length of a valid sequence ending at prime i with last gap g.
    # Transition: dp[i][g] = 1 + max{dp[j][g'] : j < i, g' <= g, primes[i] - primes[j] == g}
    # Wait, that's not right. The gap is primes[i] - primes[j], and we need g' <= primes[i] - primes[j].
    
    # dp[i][g] where g = primes[i] - primes[j] for some j < i.
    # dp[i][g] = 1 + max{dp[j][g'] : j < i, primes[i] - primes[j] == g, g' <= g}
    
    # But g is determined by j, so for each i and each j < i:
    #   g = primes[i] - primes[j]
    #   dp[i][g] = max(dp[i][g], 1 + max{dp[j][g'] : g' <= g})
    
    # To compute max{dp[j][g'] : g' <= g} quickly, for each j, maintain a prefix max over g'.
    
    # Let's define: for each prime index j, we have dp_j[g'] for various g' values.
    # We want, for a given g, the max of dp_j[g'] for g' <= g.
    
    # This is a 2D DP with num_primes x N states. 1229 * 10000 = 12.29M states.
    # Transitions: for each i, for each j < i, we do an O(1) lookup if we have prefix max.
    # Total transitions: O(num_primes^2) = 1.5M. Each transition is O(1) with prefix max.
    # But we need to build prefix max for each j, which is O(N) per j.
    # Total: O(num_primes * N + num_primes^2) = O(12M + 1.5M) = ~14M. Feasible.
    
    # Let's implement this DP.
    
    # dp[i] = dict or array mapping gap -> max length ending at i with that last gap.
    # For i=0 (first prime), no gap, length=1.
    
    # Actually, for the first element, there's no gap. Let's handle it separately.
    # A sequence of length 1 has no gap constraint.
    # A sequence of length >= 2 has a last gap.
    
    # Let's define:
    # best_len_ending_at[i] = max length of a valid sequence ending at prime i.
    # But we need to know the last gap to extend.
    
    # Let's use: dp[i][g] = max length of a valid sequence ending at prime i with last gap exactly g.
    # For the first prime in a sequence, we can think of it as having "no gap" or gap=0.
    
    # Transition: to extend a sequence ending at j with last gap g' to prime i (i > j):
    #   new_gap = primes[i] - primes[j]
    #   We need g' <= new_gap
    #   dp[i][new_gap] = max(dp[i][new_gap], dp[j][g'] + 1)
    
    # Also, we can start a new sequence at i: length 1.
    
    # For efficiency, for each j, we want to quickly query:
    #   max{dp[j][g'] : g' <= new_gap}
    
    # So for each j, maintain a prefix_max_j array where prefix_max_j[g] = max{dp[j][g'] : g' <= g}.
    
    # Implementation:
    # dp is a list of arrays, one per prime index.
    # dp[i] is an array of size N+1, where dp[i][g] = max length ending at i with last gap g.
    # Initialize all to 0.
    # For each i from 0 to num_primes-1:
    #   dp[i][0] = 1  # start a new sequence
    #   For each j < i:
    #     g = primes[i] - primes[j]
    #     if g > 0 and g <= N:
    #       # Query max dp[j][g'] for g' <= g
    #       # Use prefix_max[j][g]
    #       val = prefix_max[j][g] + 1
    #       if val > dp[i][g]:
    #         dp[i][g] = val
    #   After processing all j, build prefix_max[i]
    
    # Memory: num_primes * (N+1) = 1229 * 10001 = ~12.3M integers. In Python, this might be ~100MB. OK.
    # Time: O(num_primes^2 + num_primes * N) = ~1.5M + 12.3M = ~14M. Feasible.
    
    # Let's implement this.
    
    # But wait, we also need to track the actual sequence to return it.
    # We can store backpointers: for each (i, g), store which j and g' led to this state.
    
    # This adds memory but is manageable.
    
    # Let's code this up.
    
    # Actually, let me reconsider the memory. 1229 * 10001 = 12,291,029 integers.
    # In Python, each int is ~28 bytes, so ~344 MB. That might be too much.
    
    # Let's use arrays from the array module or numpy to save memory.
    # Or, we can note that most dp[i][g] values are 0 or small.
    
    # Alternative: use a dict for each i, storing only non-zero entries.
    # But then prefix max queries become slower.
    
    # Let's try with numpy arrays for efficiency.
    
    import numpy as np
    
    # dp[i][g] for i in 0..num_primes-1, g in 0..N
    # Use int32 to save memory
    dp = np.zeros((num_primes, N + 1), dtype=np.int32)
    prefix_max = np.zeros((num_primes, N + 1), dtype=np.int32)
    
    # Backpointers: for each (i, g), store the previous prime index j
    # We can use a separate array
    back_j = np.full((num_primes, N + 1), -1, dtype=np.int32)
    back_g = np.full((num_primes, N + 1), -1, dtype=np.int32)
    
    # Initialize: for each prime i, we can start a sequence of length 1
    # Represent this as gap=0, length=1
    for i in range(num_primes):
        dp[i][0] = 1
        prefix_max[i][0] = 1
    
    # Process primes in order
    for i in range(1, num_primes):
        pi = primes[i]
        for j in range(i):
            pj = primes[j]
            g = pi - pj
            if g < 1 or g > N:
                continue
            # Query max dp[j][g'] for g' <= g
            # This is prefix_max[j][g]
            best_prev = prefix_max[j][g]
            if best_prev > 0:
                new_len = best_prev + 1
                if new_len > dp[i][g]:
                    dp[i][g] = new_len
                    back_j[i][g] = j
                    back_g[i][g] = -1  # We'll figure out the actual g' later if needed
                    # Actually, we need to know which g' gave the max.
                    # For now, let's just track the best length.
        
        # Build prefix_max[i]
        # prefix_max[i][g] = max(dp[i][0], dp[i][1], ..., dp[i][g])
        running_max = 0
        for g in range(N + 1):
            if dp[i][g] > running_max:
                running_max = dp[i][g]
            prefix_max[i][g] = running_max
    
    # Find the best state
    best_len = 0
    best_i = -1
    best_g = -1
    for i in range(num_primes):
        for g in range(N + 1):
            if dp[i][g] > best_len:
                best_len = dp[i][g]
                best_i = i
                best_g = g
    
    # Reconstruct the sequence
    if best_i >= 0:
        seq = []
        i = best_i
        g = best_g
        while i >= 0:
            seq.append(primes[i])
            if g == 0:
                break
            # Find the j that led to this state
            j = back_j[i][g]
            if j < 0:
                break
            # We need to find the g' that was used. 
            # This is tricky because we didn't store it.
            # Let's recompute: we need the g' such that dp[j][g'] = best_prev and g' <= g.
            # For simplicity, let's just find any g' <= g with dp[j][g'] = prefix_max[j][g]
            # Actually, we can store back_g properly.
            
            # Let's restart with proper backtracking.
            break
        
        # The reconstruction is complex. Let's use a different approach.
        # Store the full sequence in the DP or use a simpler method.
    
    # Given the complexity, let's use a simpler beam search approach that stores sequences.
    
    # Let's abandon the DP and use a practical beam search.
    
    # Beam search: maintain a list of (sequence_tuple, last_gap)
    # For each new prime, try to extend all sequences in the beam.
    # Prune to keep top K by (length, -last_gap).
    
    # But this is O(K * num_primes) per prime, which is O(K * num_primes^2) total.
    # With K=10000 and num_primes=1229, that's ~15M operations. Feasible.
    
    # Let's implement this.
    
    # State: (tuple_of_primes, last_gap)
    # last_gap = 0 for sequences of length 1
    
    beam = []
    # Initialize with each prime as a length-1 sequence
    # But that's 1229 states. Let's start with just the first few.
    
    # Actually, let's start with all primes as length-1 sequences.
    for p in primes:
        beam.append(((p,), 0))
    
    # For each subsequent prime, extend the beam
    # But this is O(len(beam) * num_primes) which is too slow if beam is large.
    
    # Better: process primes in order, and for each prime, try to add it to existing sequences.
    
    # Let's use a different beam representation:
    # beam is a list of (last_prime, last_gap, length, sequence)
    # We process primes in order. For each prime p, we create new states by extending
    # existing states where p - last_prime >= last_gap.
    
    # To keep the beam manageable, we prune after each step.
    
    # Let's limit the beam size to BEAM_WIDTH.
    
    # Reset beam
    beam = []
    for p in primes[:100]:  # Start with first 100 primes
        beam.append((p, 0, 1, (p,)))
    
    # This is getting complicated. Let's use a simpler heuristic approach.
    
    # Given time constraints, let's use a multi-start greedy with random restarts.
    
    import random
    
    best_seq = []
    best_len = 0
    
    # Greedy: start at prime i, pick next prime as small as possible with gap >= current gap
    def greedy_extend(start_idx, min_gap):
        seq = [primes[start_idx]]
        current_idx = start_idx
        current_gap = min_gap
        while True:
            target = primes[current_idx] + current_gap
            # Find smallest prime >= target
            idx = bisect.bisect_left(primes, target, current_idx + 1)
            if idx >= num_primes:
                break
            new_gap = primes[idx] - primes[current_idx]
            seq.append(primes[idx])
            current_idx = idx
            current_gap = new_gap
        return seq
    
    # Try all starting primes with various initial gaps
    # For each starting prime i, try gaps from 1 to some max
    max_initial_gap = 200
    
    for i in range(num_primes):
        if time.time() > deadline:
            break
        for g in range(1, max_initial_gap + 1):
            target = primes[i] + g
            idx = bisect.bisect_left(primes, target, i + 1)
            if idx >= num_primes:
                continue
            actual_gap = primes[idx] - primes[i]
            seq = greedy_extend(i, actual_gap)
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = seq
    
    # Also try starting from the beginning with various gaps
    # And try some random restarts
    
    # If we found a good sequence, return it
    if best_seq:
        return best_seq
    
    # Fallback: greedy from the start
    return greedy_extend(0, 1)
    
# EVOLVE-BLOCK-END
