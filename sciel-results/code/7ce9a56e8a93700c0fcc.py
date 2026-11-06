"""Find longest prime subsequence <=10000 with nondecreasing gaps via layered DP + pruning."""
import time
from math import isqrt
import bisect

def _sieve_primes(limit):
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i, v in enumerate(is_prime) if v]

def construct():
    start_time = time.time()
    deadline = start_time + 75.0

    primes = _sieve_primes(10000)
    n = len(primes)
    if n <= 2:
        return primes

    # Precompute: for each prime index i, and for a given min_gap,
    # we want the best (longest) subsequence ending at some prime j < i
    # with last_gap <= (primes[i] - primes[j])
    # 
    # Key structural idea: DP where state is (index, last_gap) but we
    # organize by "last gap" buckets.
    #
    # Better: for each prime i, dp[i] = best length ending at i,
    # but we track the last gap used. To speed up, for each i we only
    # need to check j where primes[j] is "close enough" or where
    # the gap condition is satisfiable.
    #
    # Structural change: Use a priority-queue based search (best-first)
    # where each node is (current_prime_idx, last_gap, length, path).
    # Upper bound: remaining primes / minimum possible gap + 1
    # This lets us prune aggressively.

    # Actually, let me use a different structural approach:
    # Group primes by "gap class". For a sequence with nondecreasing gaps,
    # if the current gap is g, the next gap must be >= g.
    # 
    # DP: best[i][g_class] = max length of valid subsequence ending at prime i
    #     where the last gap falls in class g_class.
    # But gaps range up to ~10000, too many classes.
    #
    # Alternative structural approach: Binary search on answer length L.
    # Check if a subsequence of length L exists with nondecreasing gaps.
    # This is still hard to check efficiently.
    #
    # Let me try a different structural family: 
    # Greedy + local improvement (memetic / hill climbing).
    # Start with a greedy construction, then try swaps/insertions.

    # Actually the most promising structural change: 
    # Use the fact that for nondecreasing gaps, if we fix the first two primes,
    # the rest is somewhat constrained. Enumerate first pair, then greedily
    # extend, and use DP for the extension.
    
    # Better structural idea: 
    # dp[i] = (best_length, last_gap) for sequences ending at prime i
    # But we need to track multiple (length, last_gap) pairs per i.
    # Pareto frontier per prime i: set of (length, last_gap) that are
    # not dominated (higher length AND lower last_gap is better).
    
    # This is the key structural change: Pareto frontier DP.
    # For each prime i, maintain a list of non-dominated (length, last_gap) pairs.
    # A pair (l1, g1) dominates (l2, g2) if l1 >= l2 and g1 <= g2.
    # When extending to prime j > i with gap g = primes[j]-primes[i],
    # we can use any (l, lg) from i's frontier where lg <= g,
    # producing (l+1, g) at j.

    # Pareto frontier per prime
    # frontier[i] = list of (length, last_gap) sorted by last_gap, 
    #               with lengths strictly increasing (non-dominated)
    
    frontier = [[] for _ in range(n)]
    frontier[0].append((1, 0))  # first element, no gap yet, length 1

    for i in range(1, n):
        pi = primes[i]
        new_states = []
        
        # Try extending from all j < i
        for j in range(i):
            gap = pi - primes[j]
            # From frontier[j], find all states with last_gap <= gap
            fj = frontier[j]
            # fj is sorted by last_gap (ascending), lengths strictly increasing
            # Find rightmost index where last_gap <= gap
            # Binary search
            lo, hi = 0, len(fj)
            while lo < hi:
                mid = (lo + hi) // 2
                if fj[mid][1] <= gap:
                    lo = mid + 1
                else:
                    hi = mid
            # lo is the count of states with last_gap <= gap
            # The best is the one with largest length among those,
            # which is fj[lo-1] since lengths increase with index
            if lo > 0:
                best_len = fj[lo - 1][0]
                new_states.append((best_len + 1, gap))
        
        # Also: start a new sequence of length 2 from any j (gap = pi - primes[j])
        # This is already covered if frontier[j] has (1,0) state.
        # But we also need sequences of length 1 starting at i.
        new_states.append((1, 0))  # length 1, no gap
        
        # Build Pareto frontier for i from new_states
        # Sort by last_gap ascending, then for same gap keep max length
        new_states.sort(key=lambda x: (x[1], -x[0]))
        
        # Build non-dominated frontier
        # Process in order of increasing gap; keep only if length > all previous
        frontier_i = []
        max_len_so_far = 0
        for (length, lg) in new_states:
            if length > max_len_so_far:
                frontier_i.append((length, lg))
                max_len_so_far = length
        
        frontier[i] = frontier_i
        
        if time.time() > deadline:
            break

    # Find the best overall
    best_len = 1
    best_end = 0
    for i in range(n):
        if frontier[i]:
            # The last element in frontier[i] has the max length
            if frontier[i][-1][0] > best_len:
                best_len = frontier[i][-1][0]
                best_end = i

    # Reconstruct: need to track predecessors
    # This is tricky with Pareto frontiers. Let me add parent tracking.
    
    # Actually, let me redo with parent tracking.
    # parent[i] for the best state at i.
    
    # Let me use a different approach for reconstruction:
    # Store for each i, the best (length, last_gap, prev_index) triple.
    # But Pareto frontier may have multiple useful states.
    # For reconstruction, I'll track the best state and its parent.
    
    # Redo with full tracking
    # state[i] = list of (length, last_gap, prev_idx) for non-dominated states
    # This could blow up memory. Let me limit frontier size.
    
    # For practical purposes, let me just track the single best state per i
    # that gives the longest sequence, and for ties, the smallest last_gap.
    # This is a heuristic but should work well.
    
    # Actually, the Pareto frontier approach without full parent tracking
    # makes reconstruction hard. Let me use a hybrid:
    # Track best_len[i] and best_last_gap[i] and best_prev[i] for the
    # "best" state at i, where best means max length, tie-break min gap.
    
    best_len_arr = [1] * n
    best_gap_arr = [0] * n  # 0 means "no gap" (first element)
    best_prev_arr = [-1] * n

    for i in range(1, n):
        pi = primes[i]
        best_l = 1
        best_g = 0
        best_p = -1
        
        for j in range(i):
            gap = pi - primes[j]
            # Can we extend the best state at j?
            if best_gap_arr[j] <= gap:
                if best_len_arr[j] + 1 > best_l or (best_len_arr[j] + 1 == best_l and gap < best_g):
                    best_l = best_len_arr[j] + 1
                    best_g = gap
                    best_p = j
        
        # Also consider starting fresh at i (length 1)
        # Already covered by initialization
        
        best_len_arr[i] = best_l
        best_gap_arr[i] = best_g
        best_prev_arr[i] = best_p
        
        if time.time() > deadline:
            break

    # Find best end
    max_len = max(best_len_arr)
    end_idx = best_len_arr.index(max_len)

    # Reconstruct
    seq = []
    idx = end_idx
    while idx != -1:
        seq.append(primes[idx])
        idx = best_prev_arr[idx]
    seq.reverse()

    return seq
