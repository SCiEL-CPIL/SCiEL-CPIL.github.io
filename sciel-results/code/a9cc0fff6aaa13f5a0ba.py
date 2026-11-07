"""Exact solution via DP over small-prime subsets."""
# EVOLVE-BLOCK-START
import time

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True

def _prime_factors(n):
    """Return set of prime factors of n."""
    ps = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            ps.add(d)
            n //= d
        d += 1
    if n > 1:
        ps.add(n)
    return ps

def _compute():
    t0 = time.time()
    N = 256
    
    # All primes <= 256
    all_primes = [p for p in range(2, N + 1) if _is_prime(p)]
    
    # Small primes (first 6): 2, 3, 5, 7, 11, 13
    small_primes = all_primes[:6]
    large_primes = all_primes[6:]
    
    sp = list(small_primes)  # [2,3,5,7,11,13]
    sp_idx = {p: i for i, p in enumerate(sp)}
    n_sp = len(sp)
    
    # For each small prime q, compute P_q = q^a where q^a <= 256
    P_q = {}
    for q in sp:
        pwr = q
        while pwr * q <= N:
            pwr *= q
        P_q[q] = pwr
    
    # Precompute prime factor sets for numbers 1..256 (only need small prime factors)
    # small_factor_mask[n] = bitmask of small primes dividing n
    small_factor_mask = [0] * (N + 1)
    for n in range(1, N + 1):
        mask = 0
        for i, q in enumerate(sp):
            if n % q == 0:
                mask |= (1 << i)
        small_factor_mask[n] = mask
    
    # For each large prime p and each subset M of small primes (bitmask),
    # compute the best value: max m such that all small-prime factors of m are in M,
    # and p*m <= 256, and gcd(m, p) = 1 (automatically true since p is large prime).
    # Value = p * m.
    
    # Precompute for each large prime p and each mask M, the best m
    # best_m[p_idx][mask] = best m (or 0 if none)
    # Actually we want to maximize p*m, so maximize m given the constraint.
    
    # For a given mask M, the allowed m values are those where small_factor_mask[m] & ~M == 0
    # i.e., all small prime factors of m are in M.
    # We want the max m in [1, floor(256/p)] satisfying this.
    
    # Precompute: for each mask M, what's the max m (from 1..256) with small_factor_mask[m] & ~M == 0?
    # But we also need m <= floor(256/p), so this depends on p.
    
    # Alternative: for each p and mask, scan m from floor(256/p) down to 1.
    # This is at most 256/p iterations, which is small.
    
    n_lp = len(large_primes)
    # best_val[lp_idx][mask] = max value p*m achievable, or -1 if impossible
    # best_m_val[lp_idx][mask] = the m that achieves it
    best_val = [[-1] * (1 << n_sp) for _ in range(n_lp)]
    best_m_arr = [[0] * (1 << n_sp) for _ in range(n_lp)]
    
    for li, p in enumerate(large_primes):
        max_m = N // p  # floor(256/p)
        for mask in range(1 << n_sp):
            # Find max m in [1, max_m] such that small_factor_mask[m] & ~mask == 0
            found = False
            for m in range(max_m, 0, -1):
                if small_factor_mask[m] & ~mask == 0:
                    best_val[li][mask] = p * m
                    best_m_arr[li][mask] = m
                    found = True
                    break
            if not found:
                # m=1 always works (no prime factors)
                best_val[li][mask] = p * 1
                best_m_arr[li][mask] = 1
    
    # DP: process large primes one by one.
    # dp[li][mask] = max total value from large primes [0..li-1], where mask is the
    #   set of small primes consumed by those primes' chosen m values.
    # For each large prime li, we choose a subset M of remaining small primes,
    #   and the value is best_val[li][M] (but M must be subset of (all_small ^ mask)).
    #   Wait, actually M can be ANY subset of small primes that is disjoint from mask.
    #   The value depends on which M we assign.
    
    # Actually, let me reconsider. For large prime p, if we assign it the subset M of small primes,
    # we get value best_val[li][M] = p * m where m is the max integer with small factors in M.
    # The constraint is that M must be disjoint from the small primes already used by previous large primes.
    
    # dp[mask] after processing some large primes = max value achieved.
    # Transition: for current large prime li, for each state mask, try each subset M of (full_mask ^ mask):
    #   new_mask = mask | M
    #   new_val = dp[mask] + best_val[li][M]
    #   dp_new[new_mask] = max(dp_new[new_mask], new_val)
    
    full_mask = (1 << n_sp) - 1
    NEG = -10**18
    dp = [NEG] * (1 << n_sp)
    dp[0] = 0
    choice = [[0] * (1 << n_sp) for _ in range(n_lp)]  # choice[li][mask] = M chosen
    
    for li in range(n_lp):
        new_dp = [NEG] * (1 << n_sp)
        new_choice = [0] * (1 << n_sp)
        for mask in range(1 << n_sp):
            if dp[mask] == NEG:
                continue
            # Available small primes: full_mask & ~mask
            avail = full_mask & ~mask
            # Enumerate all subsets M of avail
            sub = avail
            while True:
                val = best_val[li][sub]
                if val >= 0:
                    new_mask = mask | sub
                    total = dp[mask] + val
                    if total > new_dp[new_mask]:
                        new_dp[new_mask] = total
                        new_choice[new_mask] = (mask, sub)
                if sub == 0:
                    break
                sub = (sub - 1) & avail
        dp = new_dp
        for msk in range(1 << n_sp):
            choice[li][msk] = new_choice[msk]
    
    # After all large primes, for each final mask, add value of remaining small primes
    # Remaining small primes contribute P_q for q not in mask.
    total_score = NEG
    best_final_mask = 0
    for mask in range(1 << n_sp):
        if dp[mask] == NEG:
            continue
        # Add small primes not in mask
        extra = 0
        for i, q in enumerate(sp):
            if not (mask & (1 << i)):
                extra += P_q[q]
        total = dp[mask] + extra
        if total > total_score:
            total_score = total
            best_final_mask = mask
    
    # Reconstruct the solution
    # We need to find the choices that led to best_final_mask with the dp values.
    # Backtrack through choice array.
    
    # But choice[li][mask] stores (prev_mask, sub) for the transition INTO mask at step li.
    # We need to trace back from best_final_mask.
    
    # Actually, let me redo the DP storing choices more carefully.
    # Let me redo with proper backtracking storage.
    
    # Redo DP with proper choice tracking
    dp = [NEG] * (1 << n_sp)
    dp[0] = 0
    # choice[li][mask] = (prev_mask, sub) meaning: to achieve state mask after processing large prime li,
    #   we came from prev_mask before li, and assigned sub to li.
    choice = [[None] * (1 << n_sp) for _ in range(n_lp)]
    
    for li in range(n_lp):
        new_dp = [NEG] * (1 << n_sp)
        for mask in range(1 << n_sp):
            if dp[mask] == NEG:
                continue
            avail = full_mask & ~mask
            sub = avail
            while True:
                val = best_val[li][sub]
                if val >= 0:
                    new_mask = mask | sub
                    total = dp[mask] + val
                    if total > new_dp[new_mask]:
                        new_dp[new_mask] = total
                        choice[li][new_mask] = (mask, sub)
                if sub == 0:
                    break
                sub = (sub - 1) & avail
        dp = new_dp
    
    # Reconstruct
    result = []
    # Add 1
    result.append(1)
    
    # Add small primes not in best_final_mask
    for i, q in enumerate(sp):
        if not (best_final_mask & (1 << i)):
            result.append(P_q[q])
    
    # Backtrack through large primes
    current_mask = best_final_mask
    assignments = []  # (li, sub) for each large prime
    for li in range(n_lp - 1, -1, -1):
        prev_mask, sub = choice[li][current_mask]
        assignments.append((li, sub))
        current_mask = prev_mask
    assignments.reverse()
    
    for li, sub in assignments:
        p = large_primes[li]
        m = best_m_arr[li][sub]
        result.append(p * m)
    
    # Verify: all distinct, all in [1,256], pairwise coprime
    assert len(result) == len(set(result)), f"Duplicate values in result: {result}"
    assert all(1 <= x <= 256 for x in result), f"Out of range: {result}"
    from math import gcd
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                raise ValueError(f"Not coprime: {result[i]} and {result[j]}")
    
    total = sum(result)
    elapsed = time.time() - t0
    if elapsed > 50:
        # Safety: return simple valid solution
        return [1] + [p for p in all_primes]
    
    return result

# Cache the result
_RESULT = None

def construct():
    global _RESULT
    if _RESULT is None:
        _RESULT = _compute()
    return _RESULT
# EVOLVE-BLOCK-END
