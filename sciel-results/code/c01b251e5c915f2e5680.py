import time
import math
from bisect import bisect_right

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    primes = [p for p in range(2, limit + 1) if is_prime[p]]
    n = len(primes)
    
    # dp[i] is a list of tuples (length, last_gap, prev_idx, prev_state_idx)
    # representing non-dominated states ending at primes[i]
    # Sorted by gap ascending (after pruning, gaps are strictly increasing)
    dp = [None] * n
    
    # Initialize dp[0]
    dp[0] = [(1, 0, -1, -1)]
    
    best_len = 1
    best_i = 0
    best_state_idx = 0
    
    for i in range(1, n):
        if time.time() > deadline:
            break
        
        # Best length and min gap for this length seen so far for dp[i]
        # We'll collect candidates and prune at the end
        candidates = []
        candidates.append((1, 0, -1, -1))
        
        pi = primes[i]
        for j in range(i):
            gap = pi - primes[j]
            states_j = dp[j]
            # states_j is sorted by gap ascending (non-dominated means gaps strictly increase)
            # We need states where G <= gap
            # Since gaps are sorted ascending, we can binary search for the last state with G <= gap
            # Find the rightmost index where G <= gap
            # Gaps are in states_j[k][1]
            # Use bisect to find how many states have G <= gap
            # Extract gaps for binary search
            # Since states_j is small, linear scan with break might be faster than building a list
            # But let's try: states_j is sorted by gap, so we can break early
            for k in range(len(states_j)):
                G = states_j[k][1]
                if G > gap:
                    break
                L = states_j[k][0]
                candidates.append((L + 1, gap, j, k))
        
        # Prune candidates: keep only non-dominated states
        # A state (L, G) dominates (L', G') if L >= L' and G <= G'
        # Sort by length descending, then gap ascending
        candidates.sort(key=lambda x: (-x[0], x[1]))
        
        non_dom = []
        min_gap_for_max_len = float('inf')
        max_len_seen = 0
        # After sorting by -length, then gap, we iterate
        # For each length level, we want the minimum gap
        # But actually, a state with higher length and smaller gap dominates all lower length states
        # So we just need: as we go through sorted candidates, keep those with gap < min_gap_encountered
        min_gap_encountered = float('inf')
        for cand in candidates:
            L, G, pi_idx, si_idx = cand
            if G < min_gap_encountered:
                non_dom.append(cand)
                min_gap_encountered = G
        
        dp[i] = non_dom
        
        # Update global best
        current_best_len = non_dom[0][0]
        if current_best_len > best_len:
            best_len = current_best_len
            best_i = i
            best_state_idx = 0

    # Reconstruct the subsequence
    seq = []
    i = best_i
    state_idx = best_state_idx
    while i != -1:
        L, G, prev_i, prev_state_idx = dp[i][state_idx]
        seq.append(primes[i])
        i = prev_i
        state_idx = prev_state_idx
        
    seq.reverse()
    return seq
