import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin

    N = 256

    # Sieve primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, 257):
        if i * i > N:
            break
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

    n = len(primes)

    # For each pair of primes (p,q), compute max p^a * q^b <= N with a>=1, b>=1
    def max_pair_product(p, q, N=256):
        best = 0
        pa = p
        while pa <= N:
            qb = q
            while pa * qb <= N:
                prod = pa * qb
                if prod > best:
                    best = prod
                qb *= q
            pa *= p
        return best

    # For each triple of primes (p,q,r), compute max p^a * q^b * r^c <= N with a,b,c>=1
    def max_triple_product(p, q, r, N=256):
        best = 0
        pa = p
        while pa <= N:
            qb = q
            while pa * qb <= N:
                rc = r
                while pa * qb * rc <= N:
                    prod = pa * qb * rc
                    if prod > best:
                        best = prod
                    rc *= r
                qb *= q
            pa *= p
        return best

    # Build all potential merges (pairs and triples) with positive gain
    # For pairs: gain = max_pair_product - prime_powers[p] - prime_powers[q]
    # For triples: gain = max_triple_product - prime_powers[p] - prime_powers[q] - prime_powers[r]
    
    # We'll handle this as a maximum weight matching problem on a graph where:
    # - Nodes are primes
    # - Edges represent merges (pairs or triples)
    # - We need to select a set of non-overlapping merges maximizing total gain
    
    # Since triples make this a 3-set packing problem (NP-hard in general),
    # but with N=256 the number of primes is small (52), we can use a 
    # branch-and-bound or DP approach.
    
    # First, let's identify all potential merges
    pair_merges = []  # (gain, frozenset of prime indices, product)
    for i in range(n):
        p = primes[i]
        Pp = prime_powers[p]
        for j in range(i + 1, n):
            q = primes[j]
            Pq = prime_powers[q]
            if p * q > N:
                break
            max_prod = max_pair_product(p, q)
            g = max_prod - Pp - Pq
            if g > 0:
                pair_merges.append((g, frozenset([i, j]), max_prod))
    
    triple_merges = []  # (gain, frozenset of prime indices, product)
    for i in range(n):
        p = primes[i]
        Pp = prime_powers[p]
        for j in range(i + 1, n):
            q = primes[j]
            Pq = prime_powers[q]
            if p * q > N // 2:  # min r is 2, so p*q*r > N
                break
            for k in range(j + 1, n):
                r = primes[k]
                Pr = prime_powers[r]
                if p * q * r > N:
                    break
                max_prod = max_triple_product(p, q, r)
                g = max_prod - Pp - Pq - Pr
                if g > 0:
                    triple_merges.append((g, frozenset([i, j, k]), max_prod))
    
    # Combine all merges
    all_merges = pair_merges + triple_merges
    all_merges.sort(key=lambda x: x[0], reverse=True)
    
    # Branch and bound to find max weight set packing
    # With 52 primes, we need to be efficient
    # Use the fact that only small primes can be in merges
    
    # Identify which primes can participate in any merge
    active_primes = set()
    for g, indices, prod in all_merges:
        for idx in indices:
            active_primes.add(idx)
    
    # For the branch and bound, we only need to consider active primes
    # Inactive primes always contribute their prime power
    
    # Map active primes to compact indices
    active_list = sorted(active_primes)
    node_map = {node: idx for idx, node in enumerate(active_list)}
    m = len(active_list)
    
    # Build adjacency: for each active prime index, which merges include it
    merges_by_prime = [[] for _ in range(m)]
    for mi, (g, indices, prod) in enumerate(all_merges):
        for idx in indices:
            ci = node_map[idx]
            merges_by_prime[ci].append(mi)
    
    # DP over bitmask for the active primes
    # State: mask of which active primes are "used" (in a selected merge)
    # We want to select a set of non-overlapping merges maximizing total gain
    
    from functools import lru_cache
    
    # Precompute for each merge its bitmask over active primes
    merge_masks = []
    for g, indices, prod in all_merges:
        mask = 0
        for idx in indices:
            ci = node_map[idx]
            mask |= (1 << ci)
        merge_masks.append(mask)
    
    # Branch and bound with memoization
    # Since m could be up to ~20, bitmask DP is feasible
    if m <= 20:
        # DP: dp[mask] = max gain achievable using only primes in mask
        size = 1 << m
        dp = [0] * size
        
        # For each merge, update dp
        # Actually, let's do it differently: enumerate all subsets of merges
        # But that's too many. Instead, use the standard set packing DP.
        
        # dp[mask] = max total gain from a set of non-overlapping merges 
        #              whose union of primes is a subset of mask
        # Transition: dp[mask] = max over all merges m where merge_mask[m] ⊆ mask:
        #              gain[m] + dp[mask ^ merge_mask[m]]
        
        # Initialize dp[0] = 0
        # For mask from 1 to size-1:
        #   dp[mask] = dp[mask without lowest set bit]  (option: don't use that prime)
        #   for each merge where merge_mask ⊆ mask and lowest set bit of mask is in merge:
        #     dp[mask] = max(dp[mask], gain[merge] + dp[mask ^ merge_mask])
        
        for mask in range(1, size):
            # Find lowest set bit
            lowbit = mask & (-mask)
            i = lowbit.bit_length() - 1
            remaining = mask ^ lowbit
            
            # Option 1: don't use prime i in any merge
            dp[mask] = dp[remaining]
            
            # Option 2: use prime i in some merge
            for mi in merges_by_prime[i]:
                g, indices, prod = all_merges[mi]
                mmask = merge_masks[mi]
                if mmask & remaining == mmask:  # merge_mask ⊆ mask
                    # Actually we need mmask ⊆ mask, and i is in mmask
                    # remaining = mask ^ lowbit, so mmask ⊆ mask means mmask & ~mask == 0
                    if mmask & ~mask == 0:
                        new_mask = mask ^ mmask
                        val = g + dp[new_mask]
                        if val > dp[mask]:
                            dp[mask] = val
        
        full_mask = (1 << m) - 1
        total_gain = dp[full_mask]
        
        # Reconstruct which merges were selected
        # Walk through the DP
        selected_merges = []
        mask = full_mask
        while mask:
            lowbit = mask & (-mask)
            i = lowbit.bit_length() - 1
            remaining = mask ^ lowbit
            
            # Check if we used prime i in a merge
            found = False
            for mi in merges_by_prime[i]:
                g, indices, prod = all_merges[mi]
                mmask = merge_masks[mi]
                if mmask & ~mask == 0:
                    new_mask = mask ^ mmask
                    if g + dp[new_mask] == dp[mask]:
                        selected_merges.append(mi)
                        mask = new_mask
                        found = True
                        break
            if not found:
                mask = remaining
    else:
        # Fallback: greedy or branch and bound for larger m
        # Use branch and bound
        total_gain = 0
        selected_merges = []
        
        def branch_and_bound(merge_idx, used_mask, current_gain, current_selection):
            nonlocal total_gain, selected_merges
            if time.time() > deadline:
                return
            if current_gain > total_gain:
                total_gain = current_gain
                selected_merges = list(current_selection)
            
            if merge_idx >= len(all_merges):
                return
            
            # Skip this merge
            branch_and_bound(merge_idx + 1, used_mask, current_gain, current_selection)
            
            # Take this merge if no conflict
            g, indices, prod = all_merges[merge_idx]
            mmask = merge_masks[merge_idx]
            if mmask & used_mask == 0:
                branch_and_bound(merge_idx + 1, used_mask | mmask, 
                                current_gain + g, current_selection + [merge_idx])
        
        branch_and_bound(0, 0, 0, [])
    
    # Build result
    result = [1]
    
    # Track which prime indices are in selected merges
    matched_prime_indices = set()
    product_values = []
    for mi in selected_merges:
        g, indices, prod = all_merges[mi]
        for idx in indices:
            matched_prime_indices.add(idx)
        product_values.append(prod)
    
    # Add unmatched primes as their max prime power
    for i, p in enumerate(primes):
        if i not in matched_prime_indices:
            result.append(prime_powers[p])
    
    # Add matched max-products
    result.extend(product_values)
    
    # Validate
    if len(set(result)) != len(result):
        return [1] + [prime_powers[p] for p in primes]
    
    for x in result:
        if not isinstance(x, int) or x < 1 or x > N:
            return [1] + [prime_powers[p] for p in primes]
    
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                return [1] + [prime_powers[p] for p in primes]
    
    return result
