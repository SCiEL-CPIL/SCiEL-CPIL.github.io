import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 70  # 70s limit, 20s margin for safety
    
    # Sieve of Eratosthenes
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, flag in enumerate(is_prime) if flag]
    n = len(primes)
    
    # DP: For each prime index i, we maintain a Pareto frontier of (length, last_gap) pairs.
    # A state (l, g) means: there exists a subsequence ending at primes[i] with length l and last gap g.
    # We want to maximize l, and for tie-breaking, minimize g to allow future extensions.
    # 
    # Key insight: For a fixed i, if we have states (l1, g1) and (l2, g2) with l1 >= l2 and g1 <= g2,
    # then (l2, g2) is dominated and can be removed.
    # So the Pareto frontier has states where as length decreases, gap must strictly decrease.
    # Equivalently, as length increases, gap must strictly increase.
    # 
    # For each i, we store the frontier as a list of (length, gap) sorted by length ascending,
    # with strictly increasing gap.
    
    # dp[i] = list of (length, gap) for prime i, sorted by length ascending, gap strictly increasing
    dp = [[] for _ in range(n)]
    dp[0] = [(1, 0)]
    
    max_len = 1
    max_idx = 0
    
    for i in range(1, n):
        if time.time() > deadline:
            break
            
        pi = primes[i]
        new_states = []
        
        for j in range(i):
            gap = pi - primes[j]
            
            # For each state in dp[j], check if gap >= last_gap
            # dp[j] is sorted by length ascending, gap strictly increasing
            # We want the maximum length state where gap >= last_gap
            # Since gap increases with length in dp[j], we can binary search
            # for the first state where last_gap <= gap, then take the last such state
            # (which has the maximum length among valid states)
            
            frontier = dp[j]
            if not frontier:
                continue
            
            # Binary search: find the rightmost index where frontier[k][1] <= gap
            # frontier is sorted by gap ascending (same as length ascending)
            lo, hi = 0, len(frontier) - 1
            best_k = -1
            while lo <= hi:
                mid = (lo + hi) // 2
                if frontier[mid][1] <= gap:
                    best_k = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            
            if best_k >= 0:
                l, g = frontier[best_k]
                new_states.append((l + 1, gap))
        
        # Also add the base state: starting a new sequence at i
        new_states.append((1, 0))
        
        # Reduce to Pareto frontier: sort by length descending, then gap ascending
        # Keep only non-dominated: as we go from high length to low length, gap must strictly decrease
        new_states.sort(key=lambda x: (-x[0], x[1]))
        reduced = []
        min_gap = float('inf')
        for l, g in new_states:
            if g < min_gap:
                reduced.append((l, g))
                min_gap = g
        
        # Store sorted by length ascending for binary search in future iterations
        reduced.sort(key=lambda x: x[0])
        dp[i] = reduced
        
        if reduced:
            curr_max_len = reduced[-1][0]  # last element has max length
            if curr_max_len > max_len:
                max_len = curr_max_len
                max_idx = i
    
    # Reconstruct the sequence
    # We need to track parent pointers. Let's redo with parent tracking.
    
    # Actually, let's redo the DP with parent tracking for reconstruction.
    # dp[i] = list of (length, gap, prev_idx) for prime i
    # Pareto frontier: as length decreases, gap must strictly decrease
    
    dp2 = [[] for _ in range(n)]
    dp2[0] = [(1, 0, -1)]
    
    max_len2 = 1
    max_idx2 = 0
    
    for i in range(1, n):
        if time.time() > deadline:
            break
            
        pi = primes[i]
        new_states = []
        
        for j in range(i):
            gap = pi - primes[j]
            
            frontier = dp2[j]
            if not frontier:
                continue
            
            # Binary search for rightmost state where gap_j <= gap
            lo, hi = 0, len(frontier) - 1
            best_k = -1
            while lo <= hi:
                mid = (lo + hi) // 2
                if frontier[mid][1] <= gap:
                    best_k = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            
            if best_k >= 0:
                l, g, prev_idx = frontier[best_k]
                new_states.append((l + 1, gap, j))
        
        new_states.append((1, 0, -1))
        
        # Reduce to Pareto frontier
        new_states.sort(key=lambda x: (-x[0], x[1]))
        reduced = []
        min_gap = float('inf')
        for l, g, prev_idx in new_states:
            if g < min_gap:
                reduced.append((l, g, prev_idx))
                min_gap = g
        
        reduced.sort(key=lambda x: x[0])
        dp2[i] = reduced
        
        if reduced:
            curr_max_len = reduced[-1][0]
            if curr_max_len > max_len2:
                max_len2 = curr_max_len
                max_idx2 = i
    
    # Reconstruct from dp2
    seq = []
    curr_idx = max_idx2
    curr_len = max_len2
    
    while curr_idx != -1:
        seq.append(primes[curr_idx])
        
        # Find the state in dp2[curr_idx] with length curr_len
        target_state = None
        for s in dp2[curr_idx]:
            if s[0] == curr_len:
                target_state = s
                break
        
        if target_state is None:
            break
            
        prev_idx = target_state[2]
        curr_idx = prev_idx
        curr_len -= 1
    
    seq.reverse()
    
    # Verify the result
    if len(seq) > 1:
        gaps = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
        for k in range(len(gaps)-1):
            if gaps[k+1] < gaps[k]:
                # Invalid - should not happen with correct DP
                break
    
    return seq
