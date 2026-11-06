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

    # ---- Dynamic Programming: find longest nondecreasing-gap prime subsequence ----
    # dp[i] = (length, last_prime, prev_prime) for the best chain ending at primes[i]
    # We want to maximize length, and among same length, we want to keep flexibility.
    # 
    # For each prime p_i, we can extend from p_j (j < i) if gap_i = p_i - p_j >= gap_prev
    # where gap_prev is the gap before p_j in the chain ending at j.
    #
    # State: dp[i] stores the best (longest) chain ending at primes[i].
    # But "best" is tricky because a shorter chain ending at i with a smaller last-gap
    # might be more extensible. So we need to track multiple states per prime.
    #
    # Alternative: dp[i][g] = max length of chain ending at primes[i] with last gap = g
    # But g can be up to ~10000, and num_primes ~ 1229, so 1229*10000 = 12M states.
    # That's feasible.
    #
    # Actually, let's think more carefully. The gap is p_i - p_{i-1} in the chain.
    # For a chain ending at primes[i], the last gap is primes[i] - prev_prime.
    # We can precompute for each pair (j, i) with j < i, the gap = primes[i] - primes[j].
    #
    # Let's use: dp[i] = list of (last_gap, length) pairs that are Pareto-optimal
    # (no other pair has both >= last_gap and >= length).
    #
    # Actually, simpler: for each i, store the maximum length achievable ending at primes[i]
    # for each possible last gap value. But that's expensive.
    #
    # Let's try a different approach: 
    # For each starting prime, do a greedy/DP extension. But that's O(n^2) starts.
    #
    # Better: Standard DP where dp[i] = max length of valid subsequence ending at primes[i].
    # To compute dp[i], for each j < i, if we can extend the chain ending at j to i,
    # we need to know the last gap of the chain ending at j. So we need to track that.
    #
    # Let's store: best[i] = (max_length, min_last_gap_for_that_length)
    # No, that's not sufficient either.
    #
    # Let me use a 2D DP: dp[i][g] = max length of chain ending at primes[i] with last gap exactly g.
    # Then for each i, for each j < i, gap = primes[i] - primes[j], and for each g_prev <= gap,
    # dp[i][gap] = max(dp[i][gap], dp[j][g_prev] + 1)
    #
    # This is O(n^2 * max_gap) which is too slow.
    #
    # Alternative: For each i, maintain a list of Pareto-optimal (last_gap, length) states.
    # When extending from j to i with gap g = primes[i] - primes[j], we can take any state
    # from j with last_gap <= g, and create a new state (g, length+1) for i.
    #
    # Let's implement this with Pareto pruning.

    # dp[i] = list of (last_gap, length) Pareto-optimal states for chains ending at primes[i]
    # A state (g1, l1) dominates (g2, l2) if g1 <= g2 and l1 >= l2
    # We keep only non-dominated states.

    dp = [[] for _ in range(num_primes)]

    for i in range(num_primes):
        # Each prime can start a chain of length 1 with last_gap = 0 (or undefined)
        # For a single element, there's no gap constraint, so we treat it as gap=0
        # Actually, for a chain of length 1, there's no "last gap". Let's use gap=0 as sentinel.
        dp[i].append((0, 1))

        # Try to extend from all j < i
        for j in range(i):
            gap = primes[i] - primes[j]
            # For each state in dp[j] with last_gap <= gap, we can extend
            for (lg, ll) in dp[j]:
                if lg <= gap:
                    # New state for i: last_gap = gap, length = ll + 1
                    dp[i].append((gap, ll + 1))

        # Prune Pareto-optimal states: keep only those not dominated
        # Sort by last_gap ascending, then for same gap keep max length
        # Then scan and keep only states with strictly increasing length
        if dp[i]:
            # Group by last_gap, keep max length for each
            from collections import defaultdict
            gap_to_len = {}
            for (g, l) in dp[i]:
                if g not in gap_to_len or l > gap_to_len[g]:
                    gap_to_len[g] = l
            # Now sort by gap
            items = sorted(gap_to_len.items())  # (gap, length)
            # Keep Pareto frontier: increasing gap, strictly increasing length
            frontier = []
            max_len = 0
            for (g, l) in items:
                if l > max_len:
                    frontier.append((g, l))
                    max_len = l
            dp[i] = frontier

    # Find the best chain
    best_len = 0
    best_end = -1
    for i in range(num_primes):
        for (g, l) in dp[i]:
            if l > best_len:
                best_len = l
                best_end = i

    if best_end == -1:
        return [2] if primes else []

    # Reconstruct the chain
    # We need to backtrack. This is complex with Pareto states.
    # Let's use a different approach: store parent pointers.
    #
    # Actually, let's redo with parent pointers.
    # dp[i] = list of (last_gap, length, parent_j, parent_state_idx)
    # This gets complicated. Let's just use a simpler DP that stores the best chain explicitly.

    # Simpler approach: for each i, store the best chain as a tuple of primes.
    # But that's memory intensive. Let's store parent pointers.

    # Let me restart with a cleaner DP.
    # dp[i] = max length of valid subsequence ending at primes[i]
    # prev[i] = the index j < i that gives the best extension to i
    # But we need to know the last gap. So:
    # best[i] = (max_length, last_gap, parent_index)
    # When extending from j to i, gap = primes[i] - primes[j]
    # We need the last gap of the best chain ending at j to be <= gap.
    # But the best chain ending at j (by length) might not be extensible to i.
    #
    # So we need multiple states. Let's keep the Pareto approach but with parent tracking.

    # Let me try a beam search / greedy approach with multiple starts.
    # For each starting prime, greedily extend choosing the smallest possible next prime
    # that maintains nondecreasing gaps. Also try other strategies.

    # Actually, let me try a different DP formulation:
    # Let f(i, g) = max length of valid subsequence ending at primes[i] with last gap = g
    # We can compute this by iterating i from 0 to n-1, and for each i, for each j < i,
    # gap = primes[i] - primes[j], and f(i, gap) = max over g_prev <= gap of f(j, g_prev) + 1
    #
    # To make this efficient, for each j, we can precompute a prefix max over gaps.
    # But gaps vary. Let's just do O(n^2 * avg_states) with the Pareto list approach,
    # but also store parent pointers for reconstruction.

    # Let me implement a clean version with parent pointers.

    # dp[i] = list of dicts: {last_gap: (length, parent_j, parent_gap)}
    # This is getting complex. Let me just use a simpler but effective approach.

    # Approach: For each possible starting gap pattern, try to build the longest chain.
    # Or: Use the fact that the problem is small (1229 primes) and do a thorough search.

    # Let me try: for each pair of starting primes (p1, p2), greedily extend with the
    # smallest valid next prime. Also try extending with other choices.

    # Actually, let me just implement the Pareto DP with parent pointers properly.

    # State: (last_gap, length, parent_info)
    # For reconstruction, store for each (i, state_idx): parent_j, parent_state_idx

    # Let's use a list for each i, where each element is a dict with keys:
    # 'gap', 'len', 'pj' (parent j index), 'psi' (parent state index in dp[pj])
    # For the first element, pj = -1.

    dp2 = [[] for _ in range(num_primes)]

    for i in range(num_primes):
        # Base case: chain of length 1
        dp2[i].append({'gap': 0, 'len': 1, 'pj': -1, 'psi': -1})

        for j in range(i):
            gap = primes[i] - primes[j]
            for psi, state in enumerate(dp2[j]):
                if state['gap'] <= gap:
                    new_state = {
                        'gap': gap,
                        'len': state['len'] + 1,
                        'pj': j,
                        'psi': psi
                    }
                    dp2[i].append(new_state)

        # Pareto prune
        if dp2[i]:
            gap_to_best = {}
            for idx, s in enumerate(dp2[i]):
                g = s['gap']
                l = s['len']
                if g not in gap_to_best or l > gap_to_best[g]['len']:
                    gap_to_best[g] = s
            items = sorted(gap_to_best.items(), key=lambda x: x[0])
            frontier = []
            max_len = 0
            for g, s in items:
                if s['len'] > max_len:
                    frontier.append(s)
                    max_len = s['len']
            dp2[i] = frontier

    # Find best
    best_len = 0
    best_i = -1
    best_state_idx = -1
    for i in range(num_primes):
        for si, s in enumerate(dp2[i]):
            if s['len'] > best_len:
                best_len = s['len']
                best_i = i
                best_state_idx = si

    # Reconstruct
    chain = []
    i = best_i
    si = best_state_idx
    while i != -1:
        chain.append(primes[i])
        s = dp2[i][si]
        if s['pj'] == -1:
            break
        i = s['pj']
        si = s['psi']

    chain.reverse()

    # Validate
    if len(chain) >= 2:
        gaps = [chain[k+1] - chain[k] for k in range(len(chain)-1)]
        for k in range(len(gaps)-1):
            if gaps[k+1] < gaps[k]:
                # Invalid, fall back to greedy
                break
        else:
            return chain

    # Fallback: greedy
    values = []
    for p in range(2, 10001):
        if is_prime[p]:
            if len(values) < 2 or p - values[-1] >= values[-1] - values[-2]:
                values.append(p)
    return values

# EVOLVE-BLOCK-END
