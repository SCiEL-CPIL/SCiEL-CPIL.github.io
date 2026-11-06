"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import heapq

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    # 1. Sieve of Eratosthenes for primes up to 10000
    LIMIT = 10000
    is_prime = [True] * (LIMIT + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, math.isqrt(LIMIT) + 1):
        if is_prime[i]:
            for j in range(i * i, LIMIT + 1, i):
                is_prime[j] = False
    primes = [p for p in range(2, LIMIT + 1) if is_prime[p]]
    N = len(primes)
    
    if N == 0:
        return []
    if N == 1:
        return primes
    
    # 2. DP approach
    # dp[i] = list of (last_gap, length) pairs representing Pareto-optimal sequences ending at primes[i]
    # A pair (g, l) dominates (g', l') if g <= g' and l >= l'
    # We maintain Pareto frontier for each prime index
    
    # For efficiency, we note that:
    # - The last gap determines what next gaps are allowed (next_gap >= last_gap)
    # - Longer sequences are always better if they have the same or smaller last gap
    # So for each prime i, we keep a list of (gap, length) where as gap increases, length must also increase
    # (otherwise the smaller gap with larger/equal length dominates)
    
    dp = [[] for _ in range(N)]  # dp[i] = list of (gap, length), sorted by gap ascending, length strictly increasing
    best_len = 1
    best_idx = 0
    
    for i in range(N):
        # Start a new sequence at primes[i] with no prior gap (gap = 0, length = 1)
        # Actually, we need to think about this differently.
        # A sequence ending at primes[i] with last gap g and length l means:
        #   p_{i-l+1}, ..., p_i where p_i - p_{i-1} = g (the last gap)
        # For the first element, there's no gap, so we can represent it as gap=0, length=1
        
        # Initialize: sequence of just primes[i]
        # We'll collect all candidate (gap, length) pairs, then compute Pareto frontier
        
        candidates = [(0, 1)]  # (last_gap, length) - gap 0 means no previous element
        
        for j in range(i):
            # If we append primes[i] after primes[j], the new gap is primes[i] - primes[j]
            new_gap = primes[i] - primes[j]
            # We can only extend sequences from dp[j] where last_gap <= new_gap
            for (prev_gap, prev_len) in dp[j]:
                if prev_gap <= new_gap:
                    candidates.append((new_gap, prev_len + 1))
        
        # Compute Pareto frontier: sort by gap ascending, keep only entries where length is strictly increasing
        candidates.sort(key=lambda x: (x[0], -x[1]))
        frontier = []
        max_len_so_far = -1
        for gap, length in candidates:
            if length > max_len_so_far:
                frontier.append((gap, length))
                max_len_so_far = length
        
        dp[i] = frontier
        
        # Track best
        if frontier:
            best_here = frontier[-1][1]  # largest length in frontier
            if best_here > best_len:
                best_len = best_here
                best_idx = i
        
        # Time check
        if time.time() > deadline - 5.0:
            break
    
    # Reconstruct the best sequence
    if best_len <= 1:
        return [primes[best_idx]]
    
    # Backtrack: we need to find the actual sequence
    # This is tricky with the Pareto frontier approach because we didn't store parent pointers
    # Let's use a different approach: store parent pointers
    
    # Actually, let me redo this with parent pointers for reconstruction
    # dp[i] = list of (gap, length, parent_idx) where parent_idx is the index of the previous prime in the sequence
    
    # Given time constraints, let me use a simpler but effective approach:
    # Use DP where dp[i] stores the best (max length) sequence ending at i for each possible last gap
    # But gaps can be large, so instead let's use the Pareto approach with parent tracking
    
    # Let me restart with a cleaner implementation that tracks parents
    
    # Clear and redo
    dp2 = [[] for _ in range(N)]  # dp2[i] = list of (gap, length, prev_idx)
    best_len2 = 1
    best_idx2 = 0
    
    for i in range(N):
        if time.time() > deadline - 3.0:
            break
        
        candidates = [(0, 1, -1)]  # (last_gap, length, prev_prime_index)
        
        for j in range(i):
            new_gap = primes[i] - primes[j]
            for (prev_gap, prev_len, prev_prev) in dp2[j]:
                if prev_gap <= new_gap:
                    candidates.append((new_gap, prev_len + 1, j))
        
        # Pareto frontier with parent tracking
        candidates.sort(key=lambda x: (x[0], -x[1]))
        frontier = []
        max_len_so_far = -1
        for gap, length, prev_idx in candidates:
            if length > max_len_so_far:
                frontier.append((gap, length, prev_idx))
                max_len_so_far = length
        
        dp2[i] = frontier
        
        if frontier:
            best_here = frontier[-1][1]
            if best_here > best_len2:
                best_len2 = best_here
                best_idx2 = i
    
    # Reconstruct sequence from best_idx2
    seq = [primes[best_idx2]]
    cur = best_idx2
    while dp2[cur]:
        # Find the entry with max length in dp2[cur]
        # The last entry in the frontier has the max length
        gap, length, prev_idx = dp2[cur][-1]
        if prev_idx == -1:
            break
        seq.append(primes[prev_idx])
        cur = prev_idx
    
    seq.reverse()
    return seq

# EVOLVE-BLOCK-END
