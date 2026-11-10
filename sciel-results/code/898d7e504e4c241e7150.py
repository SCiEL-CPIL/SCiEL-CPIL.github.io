import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 256
    
    # Primes
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    
    # Small primes: p^2 <= 256 => p <= 16
    small_primes = [p for p in primes if p * p <= n]
    large_primes = [p for p in primes if p * p > n]
    
    num_small = len(small_primes)
    sp_to_idx = {p: i for i, p in enumerate(small_primes)}
    
    # Precompute g(U): max number <= 256 with small prime factors subset of U
    # and not divisible by any small prime in S \ U.
    # This is equivalent to: max v <= 256 such that for all p in S:
    # if p in U, v can be divisible by p.
    # if p not in U, v must NOT be divisible by p.
    
    g = [0] * (1 << num_small)
    for mask in range(1 << num_small):
        best = 0
        for v in range(1, n + 1):
            # Check if v's small prime factors are a subset of mask
            # and v is not divisible by any small prime outside mask
            valid = True
            temp = v
            for i, p in enumerate(small_primes):
                if temp % p == 0:
                    if not (mask & (1 << i)):
                        valid = False
                        break
                # We don't need to check if p is in mask and v is not divisible, that's fine.
                # We just need to ensure no "outside" prime divides v.
            if valid:
                if v > best:
                    best = v
        g[mask] = best

    # DP for f(T): max sum of pairwise coprime numbers using only small primes from T
    # f(T) = max_{U subset T, U non-empty} ( g(U) + f(T \ U) )
    # To avoid symmetry, we can just iterate all submasks.
    f = [0] * (1 << num_small)
    for mask in range(1, 1 << num_small):
        # Iterate submasks
        sub = mask
        best_val = 0
        while sub:
            # To reduce symmetry, we can require that the lowest bit of mask is in sub?
            # Or just take max. Since g(U) is fixed, and f is additive,
            # taking max over all submasks works but might be redundant.
            # Let's optimize: pick the lowest set bit in mask, say i.
            # The group containing i must be some submask U that includes i.
            lowbit = mask & -mask
            # Iterate submasks U of mask that contain lowbit
            rest_mask = mask ^ lowbit
            # U must be a submask of mask containing lowbit.
            # Let U = lowbit | V, where V is a submask of rest_mask.
            # So we iterate V submask of rest_mask.
            V = rest_mask
            while True:
                U = lowbit | V
                cand = g[U] + f[mask ^ U]
                if cand > best_val:
                    best_val = cand
                if V == 0:
                    break
                V = (V - 1) & rest_mask
        f[mask] = best_val

    # Precompute h_p(U) for each large prime p and subset U of small primes
    # h_p(U) = max v <= 256 such that:
    # 1. p | v
    # 2. small prime factors of v are subset of U
    # 3. v is not divisible by any small prime in S \ U
    
    # Note: If U is empty, v must be a multiple of p, not divisible by any small prime.
    # Since p is large, p itself is a candidate.
    
    H = {}
    for p in large_primes:
        H[p] = [0] * (1 << num_small)
        for mask in range(1 << num_small):
            best = 0
            # v must be a multiple of p
            # v = k * p
            # k * p <= 256 => k <= 256 // p
            for k in range(1, n // p + 1):
                v = k * p
                # Check small prime constraints
                valid = True
                for i, q in enumerate(small_primes):
                    if v % q == 0:
                        if not (mask & (1 << i)):
                            valid = False
                            break
                if valid:
                    if v > best:
                        best = v
            H[p][mask] = best

    # DP over large primes
    # DP[i][mask] = max sum using first i large primes, consuming small primes in mask
    # i ranges from 0 to len(large_primes)
    # mask ranges from 0 to 2^num_small - 1
    
    num_large = len(large_primes)
    # Initialize DP
    # DP[0][mask] = 0 for all mask? No, if we use 0 large primes, we consume 0 small primes.
    # But we can also "waste" small primes? No, f(S \ mask) handles the rest.
    # So DP[i][mask] should represent the max sum from large primes 0..i-1,
    # such that the set of small primes consumed is EXACTLY mask.
    
    # Actually, it's easier to let DP[i][mask] be max sum using first i large primes,
    # and the set of small primes consumed is a SUBSET of mask?
    # No, standard knapsack: DP[i][mask] = max sum using first i large primes,
    # consuming small primes in mask.
    
    # Base case: DP[0][0] = 0. DP[0][mask > 0] = -inf.
    # But we can also choose to not use any small primes for a large prime.
    
    # Let's use -inf for unreachable states.
    NEG_INF = -10**9
    DP = [[NEG_INF] * (1 << num_small) for _ in range(num_large + 1)]
    DP[0][0] = 0
    
    for i in range(1, num_large + 1):
        p = large_primes[i-1]
        h_p = H[p]
        for mask in range(1 << num_small):
            # Option 1: Don't use p
            # The small primes consumed by previous primes is 'mask'.
            # So we transition from DP[i-1][mask]
            if DP[i-1][mask] > NEG_INF:
                DP[i][mask] = max(DP[i][mask], DP[i-1][mask])
            
            # Option 2: Use p with some subset U of small primes
            # The total mask is 'mask'. The subset U must be a submask of 'mask'.
            # The previous mask was 'mask \ U'.
            # Iterate submasks U of mask
            sub = mask
            while True:
                prev_mask = mask ^ sub
                if DP[i-1][prev_mask] > NEG_INF:
                    cand = DP[i-1][prev_mask] + h_p[sub]
                    if cand > DP[i][mask]:
                        DP[i][mask] = cand
                if sub == 0:
                    break
                sub = (sub - 1) & mask

    # Find the best total sum
    # Total Sum = DP[num_large][mask] + f[all_small_mask ^ mask] + 1 (for the number 1)
    all_small_mask = (1 << num_small) - 1
    max_sum = 0
    best_mask = 0
    
    for mask in range(1 << num_small):
        if DP[num_large][mask] == NEG_INF:
            continue
        rest_mask = all_small_mask ^ mask
        total = DP[num_large][mask] + f[rest_mask] + 1
        if total > max_sum:
            max_sum = total
            best_mask = mask

    # Reconstruct the set
    # We need to recover the actual numbers.
    # 1. Recover small prime numbers from f[rest_mask]
    # 2. Recover large prime numbers from DP path
    
    result = [1]
    
    # Recover small numbers
    # We need to trace back the partition for f[rest_mask]
    # This requires storing the choice in f DP.
    # Let's re-run the f DP and store choices, or just reconstruct greedily?
    # Greedy might not be optimal if f was computed with max.
    # Let's store the best submask U for each mask in f.
    
    f_choice = [0] * (1 << num_small)
    for mask in range(1, 1 << num_small):
        lowbit = mask & -mask
        rest_mask = mask ^ lowbit
        V = rest_mask
        best_val = -1
        best_U = 0
        while True:
            U = lowbit | V
            cand = g[U] + f[mask ^ U]
            if cand > best_val:
                best_val = cand
                best_U = U
            if V == 0:
                break
            V = (V - 1) & rest_mask
        f_choice[mask] = best_U

    # Now trace back f[rest_mask]
    current_mask = all_small_mask ^ best_mask
    while current_mask:
        U = f_choice[current_mask]
        # Find the number corresponding to U
        # It's the max v <= 256 with small factors subset of U and not divisible by S\U
        # We can just find it again
        best_v = 0
        for v in range(1, n + 1):
            valid = True
            for i, q in enumerate(small_primes):
                if v % q == 0:
                    if not (U & (1 << i)):
                        valid = False
                        break
            if valid:
                if v > best_v:
                    best_v = v
        if best_v > 0:
            result.append(best_v)
        current_mask ^= U

    # Recover large numbers
    # We need to trace back DP[num_large][best_mask]
    # We didn't store choices in DP.
    # Let's re-run DP with choice storage.
    
    DP_choice = [[None] * (1 << num_small) for _ in range(num_large + 1)]
    # DP_choice[i][mask] = (prev_mask, U) if we used p_i with U, or (mask, -1) if we didn't use p_i
    
    DP = [[NEG_INF] * (1 << num_small) for _ in range(num_large + 1)]
    DP[0][0] = 0
    DP_choice[0][0] = (-1, -1) # dummy
    
    for i in range(1, num_large + 1):
        p = large_primes[i-1]
        h_p = H[p]
        for mask in range(1 << num_small):
            best_val = NEG_INF
            best_prev = -1
            best_U = -1
            
            # Option 1: Don't use p
            if DP[i-1][mask] > best_val:
                best_val = DP[i-1][mask]
                best_prev = mask
                best_U = -1 # indicates not used
            
            # Option 2: Use p with subset U of mask
            sub = mask
            while True:
                prev_mask = mask ^ sub
                if DP[i-1][prev_mask] > NEG_INF:
                    cand = DP[i-1][prev_mask] + h_p[sub]
                    if cand > best_val:
                        best_val = cand
                        best_prev = prev_mask
                        best_U = sub
                if sub == 0:
                    break
                sub = (sub - 1) & mask
            
            DP[i][mask] = best_val
            if best_val > NEG_INF:
                DP_choice[i][mask] = (best_prev, best_U)

    # Trace back
    mask = best_mask
    for i in range(num_large, 0, -1):
        prev_mask, U = DP_choice[i][mask]
        if U != -1:
            # Used large prime p = large_primes[i-1] with small primes U
            p = large_primes[i-1]
            # Find the number
            best_v = 0
            for k in range(1, n // p + 1):
                v = k * p
                valid = True
                for j, q in enumerate(small_primes):
                    if v % q == 0:
                        if not (U & (1 << j)):
                            valid = False
                            break
                if valid:
                    if v > best_v:
                        best_v = v
            if best_v > 0:
                result.append(best_v)
        mask = prev_mask

    return result
