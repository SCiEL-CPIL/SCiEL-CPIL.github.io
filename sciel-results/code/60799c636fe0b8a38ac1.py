import time
from math import isqrt, gcd
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    def time_check():
        return time.time() < deadline

    N = 256

    # Sieve primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]

    # Max prime power for each prime
    prime_powers = {}
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        prime_powers[p] = power

    # Identify small primes (those where combining can help)
    # A prime p is "small" if there exists q != p with p*q <= N
    # For p >= 17, p*2 = 34 but we need p*q <= 256 with q prime
    # Actually, for p > 16, p^2 > 256, so p^k = p for k>=1
    # But p*q could be <= 256 if q is small enough
    # For p = 17, 17*2=34, 17*3=51, ..., 17*13=221, 17*17=289>256
    # So even large primes can form products with small primes
    # But the gain: p*q - p - q = (p-1)(q-1) - 1
    # For p=17, q=2: 34-17-2=15 > 0
    # Hmm, so we need to consider all primes potentially

    # Actually, let me reconsider. The base is prime powers.
    # For p >= 17, prime_powers[p] = p (since p^2 > 256)
    # Gain of p*q vs p+q: p*q - p - q
    # This is positive when (p-1)(q-1) > 1, i.e., always for p,q >= 2
    # Wait: p*q - p - q = (p-1)(q-1) - 1
    # For p=2,q=2: but we need distinct primes
    # For p=2,q=3: 6-2-3=1>0
    # For p=2,q=17: 34-2-17=15>0
    # So ALL pairs of distinct primes have positive gain for their product vs sum of prime powers!
    # But we can't use all of them because each prime can only be in one number.

    # So the problem is: partition primes into groups, each group's product <= 256,
    # maximizing sum of products. Plus include 1.

    # For a group of primes S, the value is prod(S) if prod(S) <= 256, else invalid.
    # We want to maximize sum of prod(S_i) over the partition.

    # This is a set partitioning problem. With 52 primes, it's large.
    # But note: for large primes (p > 128), they can only be alone or with 2.
    # p > 128: p*2 > 256, so they must be alone. Value = p.
    # 64 < p <= 128: p*2 <= 256, so can pair with 2. Value = 2p if paired, p if alone.
    #   Gain of pairing: 2p - p = p > 0, so always better to pair with 2 if possible.
    # But 2 can only pair with one prime.

    # Let me think about this more carefully with the matching/DP approach.
    # The current approach only considers pairs. Let me extend to triples and quadruples
    # for small primes where the product stays <= 256.

    # Small primes that can form products of 3+:
    # 2*3*5=30, 2*3*5*7=210, 2*3*5*7*11=2310>256
    # 2*3*7=42, 2*3*11=66, 2*3*13=78, 2*3*17=102, 2*3*19=114, 2*3*23=138, 2*3*29=174, 2*3*31=186, 2*3*37=222, 2*3*41=246
    # 2*5*7=70, 2*5*11=110, 2*5*13=130, 2*5*17=170, 2*5*19=190, 2*5*23=230
    # 2*7*11=154, 2*7*13=182, 2*7*17=238
    # 2*11*13=286>256
    # 3*5*7=105, 3*5*11=165, 3*5*13=195, 3*5*17=255
    # 3*7*11=231
    # 5*7*11=385>256

    # So there are many possible triples. The key question is whether a triple is better
    # than the best pair matching for those primes.

    # For primes {a,b,c}:
    # Triple: a*b*c, gain = a*b*c - a - b - c (vs prime powers, but for large primes pp=p)
    # Best pair: max of {a*b + c, a*c + b, b*c + a} minus (a+b+c)
    #   = max{a*b-c, a*c-b, b*c-a} ... no
    #   Pair (a,b) + c alone: a*b + c, gain vs base = a*b + c - a - b - c = a*b - a - b
    #   Triple: a*b*c, gain = a*b*c - a - b - c
    #   Triple better if a*b*c - a - b - c > a*b - a - b => a*b*c - c > a*b => c(a*b-1) > a*b => c > a*b/(a*b-1)
    #   For a=2,b=3: c > 6/5 = 1.2, so any c >= 2 works. But c must be distinct prime.
    #   So triple (2,3,5): 30 vs pair(2,3)+5=6+5=11, or pair(2,5)+3=10+3=13, or pair(3,5)+2=15+2=17
    #   30 > 17, so triple is better.

    # This means we need to consider hyperedges. Let me use a DP over subsets of "small" primes.
    # Define small primes as those <= 16: {2, 3, 5, 7, 11, 13}
    # For primes > 16, they can only be in groups of size 1 or 2 (with 2, or with 3 if small enough)
    # Actually, prime p > 16 can pair with 2 if 2p <= 256, i.e., p <= 128
    # And with 3 if 3p <= 256, i.e., p <= 85
    # And with 5 if 5p <= 256, i.e., p <= 51
    # Etc.

    # This is getting complex. Let me use a different approach:
    # Model as maximum weight set packing on primes, where each candidate number
    # is a set of primes with weight = the number's value.
    # Use DP over bitmasks for the small primes, and handle large primes greedily.

    # Actually, let me just extend the current matching DP to also consider triples
    # among the small primes {2,3,5,7,11,13}.

    small_primes = [2, 3, 5, 7, 11, 13]
    large_primes = [p for p in primes if p not in small_primes]

    # For small primes, compute all valid groups (subsets with product <= 256)
    # and their gains vs using them as individual prime powers
    small_pp = {p: prime_powers[p] for p in small_primes}

    # Generate all subsets of small primes with product <= 256
    small_prime_set = set(small_primes)
    valid_groups = {}  # bitmask -> (product, gain)
    for mask in range(1, 1 << 6):
        prod = 1
        base_sum = 0
        valid = True
        for i in range(6):
            if mask & (1 << i):
                p = small_primes[i]
                prod *= p
                base_sum += small_pp[p]
                if prod > 256:
                    valid = False
                    break
        if valid and prod <= 256:
            gain = prod - base_sum
            if gain > 0:
                valid_groups[mask] = (prod, gain)

    # For large primes, they can form pairs with small primes or with each other
    # But let's first solve the small prime partition, then handle large primes

    # DP over subsets of small primes
    # dp[mask] = max gain from partitioning the set of small primes in mask
    # using valid groups

    @lru_cache(maxsize=None)
    def dp_small(mask):
        if mask == 0:
            return 0
        # Find lowest set bit
        i = (mask & -mask).bit_length() - 1
        # Option 1: leave i alone (as prime power)
        best = dp_small(mask ^ (1 << i))
        # Option 2: group i with some subset of remaining
        rem = mask ^ (1 << i)
        # Try all subsets of rem to group with i
        sub = rem
        while sub > 0:
            group_mask = sub | (1 << i)
            if group_mask in valid_groups:
                _, g = valid_groups[group_mask]
                val = g + dp_small(rem ^ sub)
                if val > best:
                    best = val
            sub = (sub - 1) & rem
        return best

    if not time_check():
        # Fallback: just prime powers
        return [1] + [prime_powers[p] for p in primes]

    # Now handle large primes. They can:
    # 1. Be alone (value = p)
    # 2. Pair with a small prime (if the small prime isn't already in a group that uses it)
    # 3. Pair with another large prime (if product <= 256)

    # This is getting very complex. Let me simplify:
    # For large primes, the best they can do is pair with 2 (gain = 2p - p = p) 
    # or be alone. Since 2 can only pair with one, we pick the large prime with max p
    # that can pair with 2 (i.e., 2p <= 256, p <= 128).

    # But wait, 2 might be in a small prime group. Let me handle this more carefully.

    # Actually, let me just use the full DP approach with all primes.
    # 52 primes is too many for bitmask DP. Let me separate.

    # Key insight: large primes (p > 16) have prime_power = p.
    # They can only be in groups of size 1 or 2 (since p*q > 256 for p,q > 16, 
    # except 17*17=289>256, so no two large primes can pair).
    # A large prime p can pair with a small prime s if s*p <= 256.
    # Or with another large prime q if p*q <= 256. For p,q > 16, min is 17*19=323>256.
    # So large primes can ONLY pair with small primes, not with each other.

    # So the problem decomposes:
    # - Small primes {2,3,5,7,11,13} can form groups of size up to 4
    # - Large primes can either be alone or paired with a small prime
    # - The pairing of large primes with small primes competes with small prime grouping

    # Let me extend the DP to include the option of pairing a small prime with a large prime.
    # For each small prime s, the "best" large prime to pair with is the largest p such that s*p <= 256.
    # But that large prime can only be used once.

    # This is a matching problem between small primes and large primes, combined with
    # the small prime grouping. It's complex but the small prime set is only 6 elements.

    # Let me enumerate all possible "fates" for each small prime:
    # - In a group with other small primes
    # - Paired with a large prime
    # - Alone (as prime power)

    # For the large primes, once we decide which small primes pair with which large primes,
    # the remaining large primes are just alone.

    # The large primes that can pair with small primes:
    # s=2: p <= 128, large primes: 17,19,...,127
    # s=3: p <= 85, large primes: 17,...,83
    # s=5: p <= 51, large primes: 17,...,47
    # s=7: p <= 36, large primes: 17,19,23,29,31
    # s=11: p <= 23, large primes: 17,19,23
    # s=13: p <= 19, large primes: 17,19

    # For each small prime s, the best large prime to pair with is the largest available.
    # But they compete for the same large primes.

    # Given the complexity, let me just use the current approach (pairs only) but also
    # check if replacing a pair with a triple among small primes is better.

    # Actually, the current program already scores 1.2121. Let me try to improve it
    # by considering triples of small primes.

    # Let me rebuild: use DP over all 6 small primes with hyperedges (groups of size 1-4)
    # and for each small prime, also the option to pair with the best large prime.

    # For simplicity, let me precompute for each small prime s, the best large prime to pair with
    # (largest p > 16 with s*p <= 256). Then treat it as a "virtual" partner.

    # But large primes can't be shared. So I need to handle the matching.

    # Given time constraints, let me try a simpler improvement:
    # After finding the optimal pair matching, check if any triple of small primes
    # would give a better total than the pairs they're in.

    # Actually, let me just implement the full hyperedge DP for small primes,
    # and for large primes, pair each unpaired small prime with the best available large prime.

    # Step 1: DP over small primes with hyperedges
    # Step 2: For small primes not in any hyperedge group, pair with best large prime
    # Step 3: Remaining large primes are alone

    # But the DP needs to account for the large prime pairing option.
    # For each small prime s, the "alone" value is not just small_pp[s], but 
    # max(small_pp[s], best_large_pair_value) where best_large_pair_value = max p with s*p<=256, p>16
    # But the large prime used is consumed.

    # This is a joint optimization. Let me just enumerate all possible groupings of small primes
    # (there are 2^6=64 subsets, and for each, the number of partitions is manageable)
    # and for each grouping, compute the best assignment of remaining small primes to large primes.

    # Number of set partitions of 6 elements is Bell(6) = 203. For each, check validity.
    # For each valid partition, the "unpaired" small primes can each take a large prime.
    # The best assignment of large primes to unpaired small primes is a bipartite matching,
    # but since each small prime can take at most one large prime and vice versa, and the
    # weight is just the large prime's value, we want to assign the largest available large
    # primes to the small primes that can use them.

    # Actually, for a small prime s, the large primes it can pair with are those p > 16 with s*p <= 256.
    # The value of pairing is s*p (vs s alone = small_pp[s]).
    # Gain of pairing = s*p - small_pp[s].
    # We want to maximize total gain, which is equivalent to maximizing sum of p assigned.
    # This is a bipartite matching problem, but with 6 small primes and up to ~30 large primes,
    # we can solve it greedily: sort small primes by the threshold, assign largest available large prime.

    # Let me implement this.

    # Precompute for each small prime s, the list of large primes it can pair with
    small_to_large = {}
    for s in small_primes:
        candidates = [p for p in large_primes if s * p <= 256]
        small_to_large[s] = candidates

    # Generate all set partitions of small_primes
    # For each partition, check if all groups are valid (product <= 256)
    # For valid partitions, compute:
    #   - Sum of products for groups
    #   - For unpaired small primes (groups of size 1), assign best large primes
    #   - Total = sum(group_products) + sum(assigned_large_prime * s for pairs) + sum(remaining_large_primes)

    # Wait, I'm overcomplicating. Let me just use the current approach and add
    # a post-processing step that tries to replace pairs with triples.

    # Actually, let me just implement the hyperedge DP properly.

    # For the 6 small primes, enumerate all possible "configurations":
    # Each small prime is either in a group with other small primes, or paired with a large prime, or alone.
    # The large prime pairing is a matching.

    # Given the small size, let me just brute force all partitions of the 6 small primes
    # into groups (each group product <= 256), and for each partition, solve the
    # bipartite matching for the remaining small primes.

    from itertools import product

    # All possible group masks for small primes (product <= 256, size >= 2)
    group_masks = []
    for mask in range(1, 64):
        prod = 1
        for i in range(6):
            if mask & (1 << i):
                prod *= small_primes[i]
        if prod <= 256 and bin(mask).count('1') >= 2:
            group_masks.append(mask)

    # For each partition of {0..5}, check validity
    # A partition is a set of disjoint masks covering all 6 bits
    # Use recursive generation

    def generate_partitions(remaining, current):
        if remaining == 0:
            yield current
            return
        # Find lowest set bit
        i = (remaining & -remaining).bit_length() - 1
        # Try all subsets of remaining containing i
        sub = remaining
        while sub > 0:
            if sub & (1 << i):
                prod = 1
                for j in range(6):
                    if sub & (1 << j):
                        prod *= small_primes[j]
                if prod <= 256:
                    for rest in generate_partitions(remaining ^ sub, current + [sub]):
                        yield rest
            sub = (sub - 1) & remaining

    best_total = -1
    best_config = None

    for partition in generate_partitions(63, []):
        # partition is a list of masks
        # Groups of size >= 2 are "merged", size 1 are "unpaired"
        unpaired = []
        group_sum = 0
        for mask in partition:
            if bin(mask).count('1') == 1:
                i = mask.bit_length() - 1
                unpaired.append(small_primes[i])
            else:
                prod = 1
                for i in range(6):
                    if mask & (1 << i):
                        prod *= small_primes[i]
                group_sum += prod

        # For unpaired small primes, assign large primes
        # Greedy: sort unpaired by threshold descending, assign largest available large prime
        if not unpaired:
            large_sum = sum(large_primes)
        else:
            # For each unpaired small prime s, the large primes it can take
            # We want to maximize sum of assigned large primes
            # Greedy: process small primes in order of decreasing threshold
            # (i.e., those with fewer options first)
            unpaired_sorted = sorted(unpaired, key=lambda s: min(small_to_large[s]) if small_to_large[s] else 0)
            available_large = list(large_primes)
            assigned_sum = 0
            for s in unpaired_sorted:
                # Find largest available large prime that s can pair with
                candidates = [p for p in available_large if s * p <= 256]
                if candidates:
                    best_p = max(candidates)
                    assigned_sum += best_p
                    available_large.remove(best_p)
                # Else, s is alone
            # Remaining large primes are alone
            large_sum = assigned_sum + sum(available_large)

        total = group_sum + large_sum + 1  # +1 for the integer 1
        # Also add the small prime powers for unpaired small primes that didn't get a large prime
        # Wait, I'm double counting. Let me redo.

        # Total = 1 + sum(group_products) + sum(for unpaired small primes: either s*p or small_pp[s]) + sum(remaining large primes)

        # Let me recompute:
        unpaired = []
        group_sum = 0
        for mask in partition:
            if bin(mask).count('1') == 1:
                i = mask.bit_length() - 1
                unpaired.append(small_primes[i])
            else:
                prod = 1
                for i in range(6):
                    if mask & (1 << i):
                        prod *= small_primes[i]
                group_sum += prod

        if not unpaired:
            total = 1 + group_sum + sum(large_primes)
        else:
            unpaired_sorted = sorted(unpaired, key=lambda s: min(small_to_large[s]) if small_to_large[s] else 0)
            available_large = list(large_primes)
            pair_sum = 0
            alone_small_sum = 0
            for s in unpaired_sorted:
                candidates = [p for p in available_large if s * p <= 256]
                if candidates:
                    best_p = max(candidates)
                    pair_sum += s * best_p
                    available_large.remove(best_p)
                else:
                    alone_small_sum += small_pp[s]
            total = 1 + group_sum + pair_sum + alone_small_sum + sum(available_large)

        if total > best_total:
            best_total = total
            best_config = (partition, unpaired)

    if not time_check():
        return [1] + [prime_powers[p] for p in primes]

    # Reconstruct the result
    partition, _ = best_config
    result = [1]
    used_large = set()

    for mask in partition:
        if bin(mask).count('1') == 1:
            i = mask.bit_length() - 1
            s = small_primes[i]
            # Find best large prime
            candidates = [p for p in large_primes if s * p <= 256 and p not in used_large]
            if candidates:
                best_p = max(candidates)
                result.append(s * best_p)
                used_large.add(best_p)
            else:
                result.append(small_pp[s])
        else:
            prod = 1
            for i in range(6):
                if mask & (1 << i):
                    prod *= small_primes[i]
            result.append(prod)

    for p in large_primes:
        if p not in used_large:
            result.append(p)

    # Validate
    if len(set(result)) != len(result):
        return [1] + [prime_powers[p] for p in primes]
    for x in result:
        if x < 1 or x > N or not isinstance(x, int):
            return [1] + [prime_powers[p] for p in primes]
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                return [1] + [prime_powers[p] for p in primes]

    return result
