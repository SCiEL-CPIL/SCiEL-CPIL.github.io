import math
import time
from itertools import combinations, product

_N = 256
_DEADLINE = 85.0  # 5s safety margin under 90s wall-clock

def construct():
    t0 = time.time()
    
    # 1. Sieve primes up to 256
    is_prime = [True] * (_N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(_N ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, _N + 1, i):
                is_prime[j] = False
    primes = [p for p in range(2, _N + 1) if is_prime[p]]
    
    # 2. Classify primes
    # Large: p^2 > 256 => p > 16. Large primes must appear alone.
    # Small: p <= 16. These can interact (pair with 2 or 3, or combine in a core).
    # Mid: 16 < p <= 256, but they are actually "large" in the sense that p^2 > 256.
    # Wait, if p > 16, p^2 > 256. So any prime > 16 cannot share a number with any other prime.
    # It can only be part of a number of the form p * (product of primes <= 16).
    # But if it's p * q where q is a small prime, then gcd(p*q, r) = 1 requires r not to have p or q.
    # However, the problem is pairwise coprime.
    # If we have number A = p * q (p large, q small), and number B = r (small), we need gcd(A,B)=1.
    # This means r cannot be q, and r cannot be p (which it isn't).
    # So a large prime p can be "paired" with a small prime q to form p*q, removing q from the pool of small primes available for other numbers.
    
    # Let's refine the classification:
    # P_large: primes p where p > 16. (Since 17^2 = 289 > 256).
    # P_small: primes p where p <= 16. {2, 3, 5, 7, 11, 13}
    
    P_large = [p for p in primes if p > 16]
    P_small = [p for p in primes if p <= 16]
    
    # 3. Base solution: All primes alone.
    # Sum = sum(P_large) + sum(P_small) + 1
    # This is valid.
    
    # Optimization Strategy:
    # The "value" of a prime p used alone is p.
    # If we use a large prime p in a number N = p * k (where k is a product of small primes), 
    # the value is p * k.
    # We gain p*k - p = p*(k-1) compared to using p alone.
    # But we "lose" the small primes used in k, which previously contributed sum(k_primes) to the total (as separate numbers).
    # Net gain = p*(k-1) - sum(primes_in_k).
    # We want to maximize this net gain for each large prime pairing, subject to disjoint small prime sets.
    # However, small primes can also be combined among themselves to form a single large number from the remaining small primes.
    
    # Let's break it down:
    # The small primes are {2, 3, 5, 7, 11, 13}.
    # Any large prime p > 16 can be multiplied by a subset of small primes S_p, provided the product p * prod(S_p) <= 256.
    # The small primes used in any S_p are removed from the "core" pool.
    # The remaining small primes form a "core" number N_core = prod(S_core).
    # We want to maximize:
    # Sum_{p in P_large} (p * prod(S_p)) + N_core
    # Subject to:
    # 1. S_p are disjoint subsets of P_small.
    # 2. p * prod(S_p) <= 256.
    # 3. N_core = prod(P_small \ union(S_p)).
    # 4. We can also choose S_p = empty set, meaning p is alone. In that case, contribution is p.
    #    Note: if S_p is empty, we gain 0 from pairing, and p remains in the sum as p.
    
    # Let's reframe:
    # Total Sum = sum(P_large) + sum(P_small) + 1 + sum_{p in P_large} (p * (prod(S_p) - 1)) - (sum of small primes in union(S_p) - N_core)?
    # No, simpler:
    # If a small prime q is used in S_p, it is removed from its own number (value q) and added to p's number (value multiplied by q).
    # If a small prime q is in the core, it remains in the core number.
    
    # Let's define the decision variables:
    # For each large prime p, we choose a set S_p of small primes.
    # Let U = union of all S_p.
    # The remaining small primes are R = P_small \ U.
    # The core number is N_core = prod(R) if R is not empty, else 1 (but 1 is handled separately).
    # Actually, if R is empty, N_core is 1? No, the core number is just the product. If empty, it's 1, but we already have 1 in the list.
    # Wait, the list must contain distinct integers.
    # If R is empty, we don't add a core number (or add 1, but 1 is already there).
    # If R is not empty, we add N_core = prod(R).
    
    # The total sum is:
    # 1 + sum_{p in P_large} (p * prod(S_p)) + (prod(R) if R else 0)
    # Note: if S_p is empty, p * prod(S_p) = p * 1 = p.
    # This formula works.
    
    # Since P_small is small ({2, 3, 5, 7, 11, 13}), the number of subsets is 2^6 = 64.
    # The large primes are 17, 19, ..., 251. There are about 40 large primes.
    # We cannot iterate over all combinations of subsets for 40 primes (64^40 is huge).
    
    # However, most large primes cannot pair with ANY small prime to stay under 256?
    # 17 * 2 = 34 (ok)
    # 17 * 3 = 51 (ok)
    # 17 * 5 = 85 (ok)
    # 17 * 7 = 119 (ok)
    # 17 * 11 = 187 (ok)
    # 17 * 13 = 221 (ok)
    # 17 * 2 * 3 = 102 (ok)
    # 17 * 2 * 5 = 170 (ok)
    # 17 * 2 * 7 = 238 (ok)
    # 17 * 3 * 5 = 255 (ok)
    # 17 * 2 * 3 * 5 = 510 (too big)
    
    # Let's check the largest large prime: 251.
    # 251 * 2 = 502 > 256. So 251 must be alone.
    # In general, if p * 2 > 256, i.e., p > 128, then p must be alone.
    # Primes > 128: 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251.
    # These ~23 primes must be alone.
    
    # The "active" large primes are those <= 128.
    # 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127.
    # There are 25 active large primes.
    
    # Even 25 is too many for brute force over subsets.
    # But notice that only the small primes {2, 3, 5, 7, 11, 13} are shared resources.
    # Most large primes will only be able to use 2, or 2 and 3, etc.
    # For p >= 43, p * 2 * 3 = 258 > 256. So they can use at most one small prime (2, 3, 5... if it fits).
    # Actually, 43 * 2 = 86, 43 * 3 = 129, 43 * 5 = 215.
    # 43 * 2 * 3 = 258 (no).
    # So for p >= 43, the set S_p can have at most 1 element.
    # For p in [29, 41], p * 2 * 3 might be ok?
    # 29 * 2 * 3 = 174 (ok).
    # 31 * 2 * 3 = 186 (ok).
    # 37 * 2 * 3 = 222 (ok).
    # 41 * 2 * 3 = 246 (ok).
    # 43 * 2 * 3 = 258 (no).
    # So primes 29, 31, 37, 41 can use up to 2 small primes.
    # Primes 17, 19, 23 can use up to 3 small primes?
    # 17 * 2 * 3 * 5 = 510 (no).
    # 17 * 2 * 3 = 102 (ok).
    # 17 * 3 * 5 = 255 (ok).
    # 17 * 2 * 5 = 170 (ok).
    # 17 * 2 * 7 = 238 (ok).
    # So 17 can use 3 small primes?
    # 17 * 2 * 3 = 102. Next prime is 5. 17*2*3*5 = 510.
    # So max size of S_p for 17 is 3 (e.g., {2,3,5} is 510 no. {2,3} is 102. {3,5} is 255. {2,5} is 170. {2,7} is 238. {2,3,5} no.
    # Wait, {2,3,5} sum of exponents is 3.
    # Can 17 use 4 small primes? 17 * 2 * 3 * 5 * 7 = 3570. No.
    # So max size is 3?
    # 17 * 2 * 3 * 5 = 510 > 256.
    # So max 3 primes? No, 3 primes product is 2*3*5=30. 17*30=510.
    # So max 2 primes?
    # 2*3=6. 17*6=102.
    # 3*5=15. 17*15=255.
    # 2*5=10. 17*10=170.
    # 2*7=14. 17*14=238.
    # 2*11=22. 17*22=374 > 256.
    # So 17 can use at most 2 small primes?
    # Let's check 3 primes: min product of 3 distinct small primes is 2*3*5=30. 17*30 > 256.
    # So yes, max 2 small primes for any large prime?
    # What about 17 * 2 * 3 = 102.
    # Is there a case with 3? No, min product of 3 is 30. 17*30 > 256.
    # So for ALL large primes p >= 17, |S_p| <= 2.
    
    # This simplifies things greatly!
    # Each large prime can be paired with 0, 1, or 2 small primes.
    # The small primes are {2, 3, 5, 7, 11, 13}.
    # We need to assign subsets S_p to each large prime p such that they are disjoint.
    # Since the small prime set is small, we can use DP or backtracking on the state of the small primes.
    # State: bitmask of which small primes are used.
    # There are 6 small primes, so 2^6 = 64 states.
    # We process large primes one by one.
    # For each large prime, we try all valid subsets S (size 0, 1, or 2) that are compatible with current state (disjoint).
    # We want to maximize the total sum.
    
    # Let's define:
    # small_primes = [2, 3, 5, 7, 11, 13]
    # idx_map = {p: i for i, p in enumerate(small_primes)}
    
    # For each large prime p, precompute valid subsets S (bitmasks) and the value contribution gain.
    # Actually, let's just compute the value directly.
    # Value(p, S) = p * prod(S)
    # But we also need to account for the core.
    
    # Let's use DP over the large primes, with state being the bitmask of used small primes.
    # dp[mask] = max sum of values for the processed large primes, given that the small primes in 'mask' are used.
    # At the end, for each final mask, we add the core value prod(R) where R is the complement of mask.
    # And we add 1.
    
    # Initialize dp[0] = 0.
    # For each large prime p in P_active:
    #   new_dp = copy of dp (or -inf)
    #   For each mask in 0..63:
    #     If dp[mask] is -inf, continue.
    #     For each valid subset S (bitmask) for p:
    #       If (mask & S) != 0, continue (overlap).
    #       new_mask = mask | S
    #       val = p * prod(S)
    #       new_dp[new_mask] = max(new_dp[new_mask], dp[mask] + val)
    #   dp = new_dp
    
    # After processing all active large primes:
    #   For each mask in 0..63:
    #     If dp[mask] is -inf, continue.
    #     R = complement of mask (in 6 bits)
    #     core_val = prod(R) if R else 1  # Wait, if R is empty, core is 1?
    #     # The number 1 is always in the list.
    #     # If R is not empty, we add a number prod(R).
    #     # If R is empty, we don't add a new number (the 1 is already there).
    #     # So if R is empty, core_val = 0 (since 1 is accounted for separately).
    #     # If R is not empty, core_val = prod(R).
    #     
    #     # Plus the large primes that were NOT active (p > 128).
    #     # These are always alone.
    #     sum_large_inactive = sum(p for p in P_large if p > 128)
    #     
    #     total = dp[mask] + core_val + sum_large_inactive + 1
    #     best = max(best, total)
    
    # This DP is O(|P_active| * 64 * |subsets|).
    # |P_active| ~ 25.
    # |subsets| for each p is at most 1 + 6 + 15 = 22.
    # 25 * 64 * 22 ~ 35000 operations. Very fast.
    
    # Let's implement this.
    
    P_active = [p for p in P_large if p <= 128]
    P_inactive = [p for p in P_large if p > 128]
    
    small_primes = [2, 3, 5, 7, 11, 13]
    num_small = len(small_primes)
    
    # Precompute prod for each bitmask
    prod_mask = [1] * (1 << num_small)
    for mask in range(1, 1 << num_small):
        # Find lowest set bit
        lb = mask & -mask
        idx = (lb.bit_length() - 1)
        prod_mask[mask] = prod_mask[mask ^ lb] * small_primes[idx]
    
    # Precompute valid subsets for each active large prime
    # valid_subsets[p] = list of (bitmask, value)
    valid_subsets = {}
    for p in P_active:
        subs = []
        # Size 0
        subs.append((0, p)) # p * 1
        # Size 1
        for i in range(num_small):
            bit = 1 << i
            prod_val = small_primes[i]
            if p * prod_val <= _N:
                subs.append((bit, p * prod_val))
        # Size 2
        for i in range(num_small):
            for j in range(i + 1, num_small):
                bit = (1 << i) | (1 << j)
                prod_val = small_primes[i] * small_primes[j]
                if p * prod_val <= _N:
                    subs.append((bit, p * prod_val))
        valid_subsets[p] = subs
    
    # DP
    NEG_INF = -10**18
    dp = [NEG_INF] * (1 << num_small)
    dp[0] = 0
    
    for p in P_active:
        new_dp = [NEG_INF] * (1 << num_small)
        for mask in range(1 << num_small):
            if dp[mask] == NEG_INF:
                continue
            for s_mask, val in valid_subsets[p]:
                if mask & s_mask:
                    continue
                new_mask = mask | s_mask
                new_val = dp[mask] + val
                if new_val > new_dp[new_mask]:
                    new_dp[new_mask] = new_val
        dp = new_dp
    
    best_sum = 0
    best_mask = 0
    sum_inactive = sum(P_inactive)
    
    for mask in range(1 << num_small):
        if dp[mask] == NEG_INF:
            continue
        # Calculate core
        comp_mask = ((1 << num_small) - 1) ^ mask
        if comp_mask == 0:
            core_val = 0
        else:
            core_val = prod_mask[comp_mask]
        
        total = dp[mask] + core_val + sum_inactive + 1
        if total > best_sum:
            best_sum = total
            best_mask = mask
    
    # Reconstruct the list
    # We need to know which subset was chosen for each p to build the list.
    # We can re-run the DP with backtracking, or just store the choices.
    # Given the small size, let's re-run with backtracking.
    
    # Actually, we can just store the parent pointers or recompute.
    # Let's recompute the choices by replaying the DP.
    
    # Re-initialize DP with choices
    dp_choice = [[None] * (1 << num_small) for _ in range(len(P_active) + 1)]
    dp_choice[0][0] = (0, 0) # (prev_mask, chosen_s_mask)
    
    dp_vals = [NEG_INF] * (1 << num_small)
    dp_vals[0] = 0
    
    for i, p in enumerate(P_active):
        new_dp_vals = [NEG_INF] * (1 << num_small)
        for mask in range(1 << num_small):
            if dp_vals[mask] == NEG_INF:
                continue
            for s_mask, val in valid_subsets[p]:
                if mask & s_mask:
                    continue
                new_mask = mask | s_mask
                new_val = dp_vals[mask] + val
                if new_val > new_dp_vals[new_mask]:
                    new_dp_vals[new_mask] = new_val
                    dp_choice[i + 1][new_mask] = (mask, s_mask)
        dp_vals = new_dp_vals
    
    # Backtrack
    chosen_subsets = {}
    cur_mask = best_mask
    for i in range(len(P_active) - 1, -1, -1):
        p = P_active[i]
        prev_mask, s_mask = dp_choice[i + 1][cur_mask]
        chosen_subsets[p] = s_mask
        cur_mask = prev_mask
    
    # Build the list
    result = [1]
    
    # Add inactive large primes
    for p in P_inactive:
        result.append(p)
    
    # Add active large primes with their subsets
    used_small = set()
    for p in P_active:
        s_mask = chosen_subsets[p]
        num = p
        for i in range(num_small):
            if s_mask & (1 << i):
                num *= small_primes[i]
                used_small.add(small_primes[i])
        result.append(num)
    
    # Add core
    comp_mask = ((1 << num_small) - 1) ^ best_mask
    if comp_mask != 0:
        core_num = 1
        for i in range(num_small):
            if comp_mask & (1 << i):
                core_num *= small_primes[i]
        result.append(core_num)
    
    # Verify distinctness and coprimality
    # The construction ensures disjoint small prime sets for large primes.
    # The core uses the remaining small primes.
    # So all numbers are coprime.
    # Are they distinct?
    # 1 is distinct.
    # Inactive primes are distinct.
    # Active large primes * subsets are distinct because the large primes are distinct.
    # Core is distinct from 1 (if >1) and distinct from others (since it only has small primes).
    
    # Check if any number is repeated?
    # Could two different large primes produce the same number? No, they have different large prime factors.
    # Could a large prime number equal the core? No, core has no large prime factors.
    # Could a large prime number equal 1? No.
    
    # One edge case: if a large prime p is paired with empty set, num = p.
    # This is just the prime itself.
    
    # Final check: all are ints, in 1..256, pairwise coprime.
    # Convert to list of ints.
    
    return [int(x) for x in result]
