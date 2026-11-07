"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 12, 'maximum_start': 1000000}

def can_match(m, L, n=12):
    """
    Check if there is a perfect matching of distinct b_i in (m, m+L] 
    such that i divides b_i for i=1..n.
    Uses bitmask DP over candidates.
    """
    # Candidates are numbers in (m, m+L]
    # For each candidate c, it can be assigned to any i in 1..n such that c % i == 0.
    # We want to assign each i to a distinct candidate.
    
    # To speed up, we can prune candidates.
    # However, L is at most 144, so there are at most 144 candidates.
    # n is 12.
    # Number of masks is 2^12 = 4096.
    # For each mask, we iterate through candidates.
    # 4096 * 144 operations is very fast.
    
    # Let's map i (1..n) to bits 0..n-1.
    # For each candidate c, compute mask_i: bits set where i divides c.
    # We want to know if we can cover all bits 0..n-1 using a subset of candidates.
    # Actually, it's a matching problem: assign each i to a unique c.
    # This is equivalent to: does there exist a set of n distinct candidates c_1...c_n 
    # such that i_k divides c_k?
    # Wait, the condition is "i divides b_i". The b_i are distinct.
    # It does not say b_i is assigned to i in order. It says there exist distinct b_1...b_n 
    # such that i divides b_i.
    # So yes, it's a bipartite matching between indices {1..n} and candidates C.
    # Edge exists between i and c if c % i == 0.
    
    # DP over masks of assigned indices.
    # dp[mask] = number of candidates used so far? No, we just need to know if it's possible.
    # Actually, since we want to assign *each* i to *some* c, and each c to *at most one* i,
    # and we have exactly n indices, we need to pick n candidates.
    
    # Let's iterate through candidates in (m, m+L].
    # For each candidate c, we can assign it to any i not yet assigned, such that c % i == 0.
    
    # State: mask of assigned indices.
    # We process candidates one by one.
    # For each candidate, we can either skip it, or assign it to an available i.
    
    # Since L <= 144, the number of candidates is small.
    # We can use a set of reachable masks.
    
    reachable = {0}
    
    # Precompute divisors for i=1..n for each candidate?
    # Or just check on the fly.
    
    # To optimize, we can group candidates by their "mask" (which indices they can satisfy).
    # But since L is small, direct iteration is fine.
    
    # Let's refine: 
    # For a given m and L, candidates are m+1, ..., m+L.
    # We want to know if we can pick n distinct candidates such that we can assign them to 1..n.
    
    # Actually, a simpler DP:
    # dp[k] is the set of masks of indices that can be satisfied using the first k candidates?
    # No, candidates are not ordered by utility.
    
    # Let's use the standard subset DP for matching.
    # Let S be the set of candidates.
    # We want to know if there is a bijection from {1..n} to a subset of S.
    
    # Let's try a different angle:
    # For each mask of indices (0..2^n-1), can we satisfy this mask using some subset of candidates?
    # This is not quite right because candidates are distinct.
    
    # Let's stick to: 
    # reachable_masks: set of masks that can be formed by assigning a subset of the processed candidates.
    # We process candidates c = m+1 to m+L.
    # For each c, we compute valid_indices_mask: bits set for i in 1..n if c % i == 0.
    # New reachable masks:
    # 1. Skip c: keep old masks.
    # 2. Assign c to an i in valid_indices_mask that is not in the current mask.
    #    So for each mask in reachable, for each i in valid_indices_mask \ mask:
    #       new_mask = mask | (1 << (i-1))
    
    # This might blow up? 
    # Max masks 4096. Candidates ~144. 
    # For each candidate, we iterate through all current masks (up to 4096) and for each mask, 
    # we check valid indices.
    # 144 * 4096 * 12 is roughly 7 million ops. Very fast in Python if optimized, maybe slower.
    # Let's optimize by precomputing valid indices for each candidate? No, depends on m.
    
    # Let's try to implement this efficiently.
    
    # Optimization: 
    # If the number of reachable masks becomes large, it's fine.
    # We only care if (1<<n)-1 is in reachable at the end.
    
    full_mask = (1 << n) - 1
    if full_mask in reachable: # Not possible at start unless n=0
        pass
        
    # Actually, we can early exit if we reach full_mask.
    
    # Let's implement.
    
    # To speed up Python, we can use bit operations.
    
    # For a candidate c, valid_mask is a bitmask of i's.
    # For a current mask m_cur, the available indices are valid_mask & ~m_cur.
    # We want to add any single bit from available indices to m_cur.
    # This is equivalent to: new_masks = { m_cur | (1 << i) for i in bits(available) }
    
    # We can precompute for each possible valid_mask (0..2^n-1) and current mask?
    # No, valid_mask depends on c.
    
    # Let's just do it.
    
    # One more optimization: 
    # We don't need to keep all masks if we can prune. 
    # But 4096 is small.
    
    # Let's write the loop.
    
    # Note: We must ensure b_i are distinct. The DP naturally handles this by processing 
    # candidates one by one and each candidate can be used at most once (implicitly, 
    # as we transition from "processed c" to "processed c+1").
    # Wait, the DP state "reachable" represents masks satisfiable by *some subset* of the 
    # candidates processed so far.
    # When we process a new candidate c, we can either:
    # - Not use it: mask stays same.
    # - Use it for index i: mask gains bit i.
    # This correctly models distinctness because each candidate is considered once.
    
    # However, we need to be careful: 
    # The set `reachable` contains masks that can be formed.
    # When we add a new candidate, we generate new masks.
    # We should update `reachable` to include the new masks.
    
    # Implementation detail:
    # new_reachable = set(reachable) # Skip c
    # for mask in reachable:
    #    avail = valid_mask & ~mask
    #    while avail:
    #        bit = avail & -avail
    #        new_mask = mask | bit
    #        new_reachable.add(new_mask)
    #        avail -= bit
    # reachable = new_reachable
    
    # This is O(|reachable| * popcount(valid_mask)).
    
    # Let's code it.
    
    # Actually, we can speed this up by noting that we only care if full_mask is reachable.
    # Also, if we reach full_mask, we can return True immediately.
    
    # Let's refine the loop to be as fast as possible.
    
    # Precompute bit masks for i=1..n? 
    # bit(i) = 1 << (i-1)
    
    # For each c in range(m+1, m+L+1):
    #    valid_mask = 0
    #    for i in range(1, n+1):
    #        if c % i == 0:
    #            valid_mask |= 1 << (i-1)
            
    #    if valid_mask == 0: continue # c cannot satisfy any i
            
    #    # Expand reachable
    #    # We can do this in-place if we are careful, but creating a new set is safer.
    #    # However, sets are slow. 
    #    # Since n=12, we can use a boolean array of size 4096? 
    #    # Or a list of booleans.
    #    # Let's use a list `dp` of size 1<<n, initially dp[0]=True.
    #    # For each c:
    #       new_dp = dp[:] 
    #       for mask in range(1<<n):
    #           if dp[mask]:
    #               avail = valid_mask & ~mask
    #               while avail:
    #                   bit = avail & -avail
    #                   new_dp[mask | bit] = True
    #                   avail -= bit
    #       dp = new_dp
    #       if dp[full_mask]: return True
    
    # This is 144 * 4096 * (avg bits) ~ 144 * 4096 * 6 ~ 3.5M ops.
    # In Python, this might take ~1-2 seconds per check?
    # We have 90s. 
    # If we do binary search for each m, that's ~8 checks.
    # 8 * 2s = 16s per m.
    # We can only check ~5 m's in 90s.
    
    # We need to be faster or smarter.
    
    # Is there a faster way to check feasibility?
    # Hall's theorem? 
    # For every subset S of indices, |N(S)| >= |S|.
    # N(S) is the set of candidates divisible by at least one i in S.
    # Checking Hall's condition for all 2^12 subsets is 4096 checks.
    # For each subset S, we need to count how many candidates in (m, m+L] are divisible by any i in S.
    # This can be done using inclusion-exclusion or just checking each candidate?
    # If we check each candidate against all subsets, it's L * 2^n.
    # 144 * 4096 = 589k ops. Much faster than the DP!
    
    # Let's verify Hall's Theorem applicability.
    # We have bipartite graph G=(U, V, E) where U={1..n}, V={c1..cL}.
    # Edge (i, c) if c % i == 0.
    # Perfect matching exists iff for all S subset U, |N(S)| >= |S|.
    # Yes, this is correct.
    
    # So, for a given m, L:
    # 1. Generate candidates C = {m+1, ..., m+L}.
    # 2. For each subset S of {1..n} (represented by mask):
    #    Count how many c in C are divisible by at least one i in S.
    #    If count < popcount(S), return False.
    # 3. Return True.
    
    # How to count efficiently?
    # For each c, determine which subsets S it "satisfies" (i.e., c is in N(S)).
    # c is in N(S) if there exists i in S such that c % i == 0.
    # This is equivalent to: S is not a subset of {i : c % i != 0}.
    # Let Bad_c = {i : c % i != 0}.
    # c is NOT in N(S) iff S is a subset of Bad_c.
    # So c IS in N(S) iff S is NOT a subset of Bad_c.
    
    # We want to count for each S, how many c have S NOT subset of Bad_c.
    # Alternatively, count for each S, how many c have S subset of Bad_c.
    # Let Z[S] = number of c such that S is a subset of Bad_c.
    # Then |N(S)| = L - Z[S].
    # We need L - Z[S] >= |S| => Z[S] <= L - |S|.
    
    # How to compute Z[S] for all S?
    # For each c, compute Bad_c mask.
    # For each subset S of Bad_c, increment Z[S].
    # Sum over all c of 2^(popcount(Bad_c)) operations?
    # Bad_c can have up to 12 bits. 2^12 = 4096.
    # L * 4096 = 144 * 4096 = 589k ops.
    # This is very fast.
    
    # Algorithm for can_match(m, L):
    # 1. Z = [0] * (1<<n)
    # 2. For c in m+1 .. m+L:
    #      bad_mask = 0
    #      for i in 1..n:
    #          if c % i != 0: bad_mask |= 1 << (i-1)
    #      # Iterate all subsets of bad_mask
    #      sub = bad_mask
    #      while True:
    #          Z[sub] += 1
    #          if sub == 0: break
    #          sub = (sub - 1) & bad_mask
    # 3. For S in 0 .. (1<<n)-1:
    #      if Z[S] > L - popcount(S):
    #          return False
    # 4. return True
    
    # Wait, S=0 is trivial. |N(0)|=L >= 0.
    # We only need to check non-empty S.
    
    # Complexity: L * 2^(n - avg_good) + 2^n.
    # avg_good is number of divisors of c in 1..n.
    # For c ~ 1000, divisors in 1..12 are few (1, 2, 3, 4...). 
    # So bad_mask has many bits set. 2^12 is max.
    # This is efficient.
    
    return True # Placeholder, will implement below

def compute_f(m, n=12):
    """
    Compute the least positive L such that (m, m+L] contains distinct b_i 
    with i dividing b_i.
    """
    # L is at most 144? The problem says "raw value ... is at most 144".
    # So we can just iterate L from 1 to 144 and return the first valid one.
    # Or binary search? 
    # Is the property "exists matching for L" monotonic?
    # If there is a matching for L, is there one for L+1?
    # The interval (m, m+L+1] contains (m, m+L].
    # The candidates for L+1 are a superset of candidates for L.
    # If we had a matching for L, we can use the same b_i for L+1.
    # So yes, it is monotonic.
    # However, iterating 1..144 is only 144 checks.
    # Binary search is ~8 checks.
    # Since check is ~0.1ms, 144 checks is ~15ms.
    # Binary search is ~1ms.
    # Let's use binary search for safety, or just linear since 144 is small.
    # Linear is simpler and less error-prone regarding monotonicity edge cases (though it is monotonic).
    # Actually, 144 * 0.1ms = 14.4ms. 
    # If we do 1000 random m's, that's 14 seconds.
    # We have 90s. So we can do ~6000 m's.
    
    # Let's implement the fast check first.
    
    def check(L):
        Z = [0] * (1 << n)
        # Candidates m+1 to m+L
        # We can optimize by precomputing bad_mask for each c?
        # No, c changes with m.
        
        for c in range(m + 1, m + L + 1):
            bad_mask = 0
            # i from 1 to n
            # c % i != 0
            # We can compute this quickly.
            for i in range(1, n + 1):
                if c % i != 0:
                    bad_mask |= 1 << (i - 1)
            
            # Iterate subsets of bad_mask
            sub = bad_mask
            while True:
                Z[sub] += 1
                if sub == 0:
                    break
                sub = (sub - 1) & bad_mask
        
        # Check Hall's condition
        # For each S, Z[S] <= L - popcount(S)
        # popcount(S) can be precomputed.
        
        # Precompute popcounts for 0..2^n-1
        # This is static, do it once outside?
        
        for S in range(1, 1 << n):
            if Z[S] > L - popcounts[S]:
                return False
        return True

    # Precompute popcounts
    popcounts = [0] * (1 << n)
    for i in range(1, 1 << n):
        popcounts[i] = popcounts[i >> 1] + (i & 1)
    
    # Binary search for least L
    # Max L is 144? 
    # The problem states "raw value ... is at most 144".
    # So if no L <= 144 works, what then? 
    # "The evaluator computes f(12,m)... Maximize this independently computed L."
    # "The raw value is f(12,m), which is at most 144."
    # This implies f(12,m) is always <= 144.
    # So we can search in [1, 144].
    
    lo, hi = 1, 144
    ans = 144
    while lo <= hi:
        mid = (lo + hi) // 2
        if check(mid):
            ans = mid
            hi = mid - 1
        else:
            lo = mid + 1
            
    return ans

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    best_m = 12
    best_val = compute_f(best_m)
    
    # Random search
    # We can also try local search around good candidates.
    
    # Strategy: 
    # 1. Sample random m's.
    # 2. Keep track of best.
    # 3. If time allows, try neighbors of best.
    
    # To speed up, we can cache compute_f results? 
    # m is up to 1000000, unlikely to repeat.
    
    # Let's just random search.
    
    # We can also bias the search.
    # What m gives large L?
    # Large L means it takes a long time to find 12 distinct multiples.
    # This happens if the interval is "sparse" in terms of multiples of 1..12.
    # This is related to the Chinese Remainder Theorem or covering systems.
    # Numbers that are coprime to 1..12? 
    # If m is such that m+1..m+L are mostly coprime to small numbers?
    # Actually, if a number is coprime to 1..12, it is only divisible by 1.
    # So it can only satisfy i=1.
    # If we have many such numbers, we struggle to satisfy i=2..12.
    # So we want m such that (m, m+L] contains many numbers coprime to 2,3,4,5,6,7,8,9,10,11,12.
    # i.e., coprime to 2,3,5,7,11. (LCM of 1..12 is 27720).
    
    # Let's just random search first, it's robust.
    
    random.seed(42)
    
    while time.time() < deadline:
        m = random.randint(0, 1000000)
        val = compute_f(m)
        if val > best_val:
            best_val = val
            best_m = m
            
    return best_m

# Precompute popcounts globally for speed?
# Or inside compute_f? 
# It's better to do it once.

_N = 12
_POPCOUNTS = [0] * (1 << _N)
for _i in range(1, 1 << _N):
    _POPCOUNTS[_i] = _POPCOUNTS[_i >> 1] + (_i & 1)

def _check_L(m, L, n=12, popcounts=_POPCOUNTS):
    Z = [0] * (1 << n)
    for c in range(m + 1, m + L + 1):
        bad_mask = 0
        # Unroll for speed? 
        # i=1: c%1==0 always, so bit 0 never in bad_mask? 
        # Wait, i=1 always divides c. So bit 0 is NEVER in bad_mask.
        # So bad_mask only has bits 1..11 potentially.
        # This reduces the subset iteration space? 
        # No, bad_mask is a mask of 12 bits, but bit 0 is 0.
        # So subsets of bad_mask are subsets of {1..11}.
        # Max 2^11 = 2048 subsets.
        
        for i in range(2, n + 1):
            if c % i != 0:
                bad_mask |= 1 << (i - 1)
        
        sub = bad_mask
        while True:
            Z[sub] += 1
            if sub == 0:
                break
            sub = (sub - 1) & bad_mask
    
    for S in range(1, 1 << n):
        if Z[S] > L - popcounts[S]:
            return False
    return True

def compute_f_fast(m, n=12):
    lo, hi = 1, 144
    ans = 144
    while lo <= hi:
        mid = (lo + hi) // 2
        if _check_L(m, mid, n):
            ans = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return ans

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    best_m = 12
    best_val = compute_f_fast(best_m)
    
    random.seed(42)
    
    # We can also try specific patterns.
    # e.g. m such that m+1 is prime?
    
    while time.time() < deadline:
        m = random.randint(0, 1000000)
        val = compute_f_fast(m)
        if val > best_val:
            best_val = val
            best_m = m
            
    return best_m

# EVOLVE-BLOCK-END
