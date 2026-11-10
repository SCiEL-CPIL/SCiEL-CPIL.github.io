import time
import math
from math import gcd

def construct():
    deadline = time.time() + 80  # safety margin before 90s limit
    n = 256

    # Sieve of Eratosthenes to find all primes up to n
    def sieve_primes(limit):
        if limit < 2:
            return []
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, limit + 1, i):
                    is_prime[j] = False
        return [i for i, p in enumerate(is_prime) if p]

    primes = sieve_primes(n)
    primes_set = set(primes)

    # For each prime p, find the maximum power p^k <= n
    max_powers = {}
    for p in primes:
        pk = p
        while pk * p <= n:
            pk *= p
        max_powers[p] = pk

    # Identify "small" primes that can potentially form beneficial composites
    # A prime p is "small" if there exists q != p such that p*q <= n (or more generally,
    # grouping primes can yield a product <= n that beats individual max powers)
    # For n=256, primes p where p <= 256/2 = 128 are candidates for pairing.
    # But we only need to consider primes where combining with others could beat
    # the sum of individual max powers.
    
    # Strategy: Enumerate all partitions of a small set of "active" primes.
    # Active primes: those p where p < n/2 (so p*2 <= n is possible, or p*q <= n for some q)
    # Actually, let's be more precise. 
    # The gain from grouping a set G of primes into a single number v (where prime_factors(v) = G)
    # is v - sum_{p in G} max_powers[p].
    # We want max over all valid groupings.
    
    # Which primes can participate in a group with size > 1?
    # If p is large, p*q > n for all q >= 2, so it can't be in a composite.
    # Threshold: p <= n//2 = 128.
    # Primes <= 128: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113.
    # That's 30 primes. 2^30 is too large for naive enumeration.
    
    # However, we can prune. 
    # If p > n/3, then p*q > n for q >= 3. So p can only pair with 2.
    # If p > n/4, p can only pair with 2 or 3? No, p*2 <= n is the main constraint.
    
    # Let's use a smarter approach: 
    # The optimal solution is a partition of the set of all primes into groups.
    # For primes p > n/2, they must be singletons.
    # For primes p in (n/3, n/2], they can only be in groups {p} or {p, 2} (since p*q > n for q>=3).
    # For primes p in (n/4, n/3], they can be in {p}, {p,2}, or {p,3}? No, p*3 > n. So {p,2} only? 
    # Wait, if p=83 (n/3 ~ 85.3), 83*2=166<=256. 83*3=249<=256. So {83,3} is possible.
    
    # Let's identify the "core" small primes that have many interaction possibilities.
    # Primes p where p <= n/2.
    # Actually, let's just brute-force the partition of the set of primes p such that p <= n/2.
    # But 30 primes is too many for Bell numbers.
    
    # Alternative: The value of a group G is max{ v <= n : prime_factors(v) = G }.
    # If G has 3 primes {p,q,r}, the max v is roughly p*q*r (if <=n).
    # For n=256, the largest product of 3 distinct primes is 2*3*5=30, 2*3*7=42, ... 
    # Even 2*3*13=78, 2*3*17=102, 2*3*19=114, 2*3*23=138, 2*3*29=174, 2*3*31=186, 2*3*37=222, 2*3*41=246.
    # 2*3*43=258 > 256.
    # So groups of size 3 are possible only with small primes.
    
    # Let's define the set of "active" primes as those p where p <= 13 (primes: 2,3,5,7,11,13).
    # For p > 13, can they form beneficial groups?
    # Check p=17. Max power 17. 
    # Group {17,2}: max v with factors 2,17 is 17*2^a <= 256. 17*16=272>256, 17*8=136. 
    # Gain: 136 - (17+128) = 136 - 145 = -9. Bad.
    # Group {17,3}: 17*3^a. 17*9=153. Gain: 153 - (17+27) = 153 - 44 = 109. Good!
    # Wait, max power of 3 is 81 (3^4=81). 
    # Sum of singletons: 17 + 81 = 98. 
    # Composite 153. Gain 55. 
    # So 17 can form a beneficial group with 3.
    
    # This implies we need to consider more primes.
    # Let's systematically check which primes can form beneficial pairs or triples.
    
    # We will use a DP or Branch and Bound over the primes.
    # But first, let's identify all "candidate groups".
    # A group is a set of distinct primes.
    # We generate all subsets of primes up to size 3 or 4 (since 2*3*5*7=210 <= 256, 2*3*5*11=330 > 256).
    # So max group size is 4 (specifically {2,3,5,7}).
    
    # List of primes up to 256:
    # We will iterate through all possible subsets of primes of size 1, 2, 3, 4.
    # For each subset, compute the max value v <= 256 with exactly those prime factors.
    # Then solve the Set Packing problem: maximize sum of values, such that each prime is used at most once.
    
    # Since the number of primes is 54, but most are large and only form singleton groups,
    # we can separate them.
    # Primes p > 256/2 = 128: Must be singletons. (131, 137, ..., 251)
    # Primes p in (128, 256]: Singletons.
    # Primes p <= 128: Can form groups.
    
    # Let's generate all valid groups (subset of primes, max value <= 256) that have positive gain
    # over the sum of individual max powers.
    
    # Actually, we should consider ALL valid groups, including singletons, and let the optimizer choose.
    # But singletons are trivial: value = max_powers[p].
    
    # Optimizer:
    # We have a set of primes P_active (those that can appear in groups of size >= 2).
    # We have a set of candidate groups G.
    # Each group g has a set of primes S_g and a value V_g.
    # We want to select a set of groups such that no prime is shared, maximizing total value.
    # Plus, for any prime not used in a selected group, we add its singleton value.
    
    # This is equivalent to: Maximize sum_{g in Selected} (V_g - sum_{p in S_g} max_powers[p]) + sum_{all p} max_powers[p].
    # So we just need to maximize the "gain" of the selected groups.
    
    # Let's compute gains.
    # Only groups with gain > 0 are worth considering.
    
    # Generate candidate groups:
    # Primes up to 128.
    # Subsets of size 2, 3, 4.
    
    small_primes = [p for p in primes if p <= 128]
    
    candidate_groups = []  # List of (frozenset(primes), value, gain)
    
    # Helper to find max number <= n with given prime factors
    def max_val_with_factors(factor_set, n):
        # The number must be of form prod(p_i ^ a_i)
        # We want to maximize the value <= n.
        # Since factor set is small, we can iterate.
        # Actually, the max value is just the largest multiple of the radical (product of primes) that is <= n?
        # No, because exponents can vary.
        # Example: factors {2,3}. Numbers: 6, 12, 18, 24, 36, 48, 54, 72, ...
        # We want the largest one <= 256.
        # It's not simply 256 // 6 * 6. It could be 240 (2^4 * 3 * 5? No, 5 not in set).
        # For {2,3}, numbers are 2^a 3^b.
        # We can iterate b from 1 while 3^b <= n, then for each b, a can be up to log2(n/3^b).
        
        best = 0
        # Recursive generation of divisors
        def gen_divisors(idx, current):
            nonlocal best
            if idx == len(factor_set):
                if current > best and current <= n:
                    best = current
                return
            p = list(factor_set)[idx]
            val = current
            while val <= n:
                gen_divisors(idx + 1, val)
                val *= p
        
        gen_divisors(0, 1)
        return best

    # This recursive approach might be slow if called many times.
    # But the number of candidate groups is small.
    # Primes <= 128: ~30 primes.
    # P(30,2) + P(30,3) + P(30,4) is large (435 + 4060 + 27405 ~ 32k).
    # For each, we run a small DFS. It should be fast enough in Python?
    # 32k * (small constant) might take a few seconds.
    
    # Let's optimize max_val_with_factors.
    # For a fixed set of primes, the max value is determined.
    # We can precompute for all pairs, triples, quadruples.
    
    # Actually, we can just iterate through all numbers 1..256, factor them, and group them.
    # This is much faster!
    
    # For each number v in 1..256:
    # Find its prime factors.
    # This defines a group S_v.
    # We want to pick a set of numbers such that their prime factors are disjoint.
    # And maximize sum of v.
    
    # This is a weighted set packing problem.
    # Elements: Primes.
    # Sets: For each v in 1..256, the set of prime factors of v.
    # Weight: v.
    # But note: we can pick at most one number for a given set of prime factors?
    # No, we can pick different numbers with the same prime factors?
    # No, because if we pick v1 and v2 with the same prime factors, they are not coprime.
    # So for each set of prime factors S, we can only pick ONE number v such that prime_factors(v)=S.
    # And we should pick the maximum such v.
    
    # So, step 1: For each subset of primes S (that is the factor set of some v<=256), find max v.
    # Step 2: Solve weighted set packing on these groups.
    
    # Let's build the groups.
    groups = {}  # frozenset(primes) -> max_value
    
    for v in range(1, n + 1):
        # Factorize v
        temp = v
        factors = set()
        d = 2
        while d * d <= temp:
            if temp % d == 0:
                factors.add(d)
                while temp % d == 0:
                    temp //= d
            else:
                d += 1
        if temp > 1:
            factors.add(temp)
        
        # Note: v=1 has no prime factors.
        # We treat 1 as a special group with empty factor set.
        # 1 is coprime to everything.
        
        if not factors:
            # v=1
            key = frozenset()
            if key not in groups or groups[key] < v:
                groups[key] = v
        else:
            key = frozenset(factors)
            if key not in groups or groups[key] < v:
                groups[key] = v
    
    # Now we have a list of (frozenset(primes), value).
    # We need to select a subset of these such that the prime sets are pairwise disjoint.
    # Maximize sum of values.
    
    # Note: The empty set (v=1) is disjoint from everything. So we should always include it.
    # But wait, if we include 1, it uses no primes.
    # So we can always add 1 to the sum.
    
    # Let's remove the empty set group and handle 1 separately.
    one_val = groups.get(frozenset(), 1)
    del groups[frozenset()]
    
    # Filter groups: We only need to consider groups that are "better" than their individual prime powers?
    # Not necessarily. The optimizer will decide.
    # But we can prune groups that are strictly worse than the sum of max_powers of their primes.
    # Actually, the set packing will naturally prefer better combinations.
    # However, if a group has value < sum(max_powers[p] for p in S), it's never beneficial to pick it
    # instead of picking the singletons.
    # So we can remove such groups to reduce search space.
    
    filtered_groups = []
    for S, val in groups.items():
        sum_singles = sum(max_powers[p] for p in S)
        if val > sum_singles:
            filtered_groups.append((S, val))
    
    # If filtered_groups is empty, the answer is sum(max_powers.values()) + 1.
    # Otherwise, we need to solve the set packing problem.
    
    # The number of filtered groups is small.
    # Let's count.
    # For n=256, likely only a few dozen groups have positive gain.
    
    # Exact solution via Backtracking / Branch and Bound.
    # Sort groups by gain (val - sum_singles) descending? Or just value?
    # We want to maximize total value.
    
    # Map primes to indices for bitmasking?
    # Number of active primes (those appearing in filtered_groups) is small.
    active_primes = set()
    for S, val in filtered_groups:
        active_primes.update(S)
    
    if not active_primes:
        return list(max_powers.values()) + [1]
    
    # Map active primes to 0..k-1
    prime_list = list(active_primes)
    prime_to_idx = {p: i for i, p in enumerate(prime_list)}
    k = len(prime_list)
    
    # Convert groups to bitmasks
    group_masks = []
    group_vals = []
    group_gains = []
    
    for S, val in filtered_groups:
        mask = 0
        for p in S:
            mask |= (1 << prime_to_idx[p])
        group_masks.append(mask)
        group_vals.append(val)
        sum_singles = sum(max_powers[p] for p in S)
        group_gains.append(val - sum_singles)
    
    # Sort by gain descending to find good solutions early
    indices = sorted(range(len(group_masks)), key=lambda i: group_gains[i], reverse=True)
    group_masks = [group_masks[i] for i in indices]
    group_vals = [group_vals[i] for i in indices]
    group_gains = [group_gains[i] for i in indices]
    
    n_groups = len(group_masks)
    
    best_gain = 0
    best_selection = [0] * n_groups
    
    # Branch and Bound
    def bb(idx, current_mask, current_gain, selection):
        nonlocal best_gain, best_selection
        if idx == n_groups:
            if current_gain > best_gain:
                best_gain = current_gain
                best_selection = selection[:]
            return
        
        # Upper bound: current_gain + sum of remaining positive gains
        # Simple bound: sum of all remaining gains
        remaining_gain = sum(group_gains[idx:])
        if current_gain + remaining_gain <= best_gain:
            return
        
        # Option 1: Skip this group
        bb(idx + 1, current_mask, current_gain, selection)
        
        # Option 2: Take this group (if no conflict)
        if (current_mask & group_masks[idx]) == 0:
            selection[idx] = 1
            bb(idx + 1, current_mask | group_masks[idx], current_gain + group_gains[idx], selection)
            selection[idx] = 0
    
    # Initial selection: all zeros
    selection = [0] * n_groups
    bb(0, 0, 0, selection)
    
    # Construct result
    result = [1]  # Always include 1
    
    # Add singletons for inactive primes (primes not in active_primes)
    for p in primes:
        if p not in active_primes:
            result.append(max_powers[p])
    
    # Add selected groups
    for i in range(n_groups):
        if best_selection[i]:
            # Find the original set of primes for this group
            # We lost the original S in the sorting?
            # We can recover it from the mask.
            mask = group_masks[i]
            S = set()
            for j, p in enumerate(prime_list):
                if mask & (1 << j):
                    S.add(p)
            # Find the value associated with this S
            # We have group_vals[i]
            result.append(group_vals[i])
    
    # Sort result
    result = list(set(result))  # Ensure distinct (should be, but just in case)
    result.sort()
    
    # Validate
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                # Fallback
                return list(max_powers.values()) + [1]
    
    return result

if __name__ == "__main__":
    res = construct()
    print(sum(res))
