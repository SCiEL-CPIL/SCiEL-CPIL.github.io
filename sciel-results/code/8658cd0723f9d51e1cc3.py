"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    target = math.factorial(12)
    deadline = time.time() + 85.0
    
    # Generate all divisors of target
    divisors = set()
    for i in range(1, int(math.isqrt(target)) + 1):
        if target % i == 0:
            divisors.add(i)
            divisors.add(target // i)
            
    # Filter divisors > 12 and sort
    candidates = sorted([d for d in divisors if d > 12])
    cand_set = set(candidates)
    n_cands = len(candidates)
    
    # Precompute prime factors of target for pruning
    def prime_factors(n):
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors
    
    target_pf = prime_factors(target)
    primes_in_target = sorted(target_pf.keys())
    num_primes = len(primes_in_target)
    
    # Build prime index lookup for O(1) access
    prime_idx = {p: i for i, p in enumerate(primes_in_target)}
    
    # Convert prime factor dict to vector
    def pf_to_vec(pf):
        vec = [0] * num_primes
        for p, e in pf.items():
            vec[prime_idx[p]] = e
        return tuple(vec)
    
    target_vec = pf_to_vec(target_pf)
    
    # Precompute for each candidate its prime factorization as vector
    cand_pf_vec = [None] * n_cands
    for i, c in enumerate(candidates):
        cand_pf_vec[i] = pf_to_vec(prime_factors(c))
    
    # Precompute prefix sums of prime vectors for pruning
    # prefix_pf[i] = sum of prime vectors of candidates[0..i-1]
    prefix_pf = [None] * (n_cands + 1)
    prefix_pf[0] = (0,) * num_primes
    for i in range(1, n_cands + 1):
        prev = prefix_pf[i-1]
        curr = cand_pf_vec[i-1]
        prefix_pf[i] = tuple(prev[j] + curr[j] for j in range(num_primes))
    
    # Precompute suffix products for quick range product checks
    suffix_prod = [1] * (n_cands + 1)
    for i in range(n_cands - 1, -1, -1):
        suffix_prod[i] = suffix_prod[i + 1] * candidates[i]
    
    # Cache for prime factorizations of values encountered during search
    pf_cache = {}
    
    def get_pf_vec(n):
        if n in pf_cache:
            return pf_cache[n]
        pf = prime_factors(n)
        vec = pf_to_vec(pf)
        pf_cache[n] = vec
        return vec
    
    def can_form(rem_vec, start_idx, end_idx):
        """Check if rem_vec can be formed by distinct candidates from start_idx to end_idx-1.
        Uses prefix sum pruning."""
        if all(v == 0 for v in rem_vec):
            return True
        if start_idx >= end_idx:
            return False
        # Check if rem_vec can be covered by the sum of all available candidates
        for j in range(num_primes):
            if rem_vec[j] > prefix_pf[end_idx][j] - prefix_pf[start_idx][j]:
                return False
        return True
    
    best_factors = [target]  # Fallback
    best_max = target
    
    def find_subset(remaining, start_idx, end_idx, used_vec, chosen, current_max):
        """DFS to find distinct candidates from [start_idx, end_idx) that multiply to remaining.
        used_vec tracks prime exponents used. chosen tracks selected indices."""
        nonlocal best_max
        
        if time.time() > deadline:
            return None
        
        if remaining == 1:
            return list(chosen)
        
        if start_idx >= end_idx:
            return None
        
        # Pruning: if remaining < candidates[start_idx], can't form it
        if remaining < candidates[start_idx]:
            return None
        
        # Check if remaining divides target and is a valid divisor
        if target % remaining != 0:
            return None
        
        # Compute remaining prime vector
        rem_vec = get_pf_vec(remaining)
        
        # Check if we can still form the remaining
        if not can_form(rem_vec, start_idx, end_idx):
            return None
        
        # Try each candidate from start_idx to end_idx-1
        for i in range(start_idx, end_idx):
            f = candidates[i]
            if f > remaining:
                break
            if remaining % f != 0:
                continue
            
            # Check prime vector feasibility
            new_used = tuple(used_vec[j] + cand_pf_vec[i][j] for j in range(num_primes))
            # Check we don't exceed target's prime exponents
            feasible = True
            for j in range(num_primes):
                if new_used[j] > target_vec[j]:
                    feasible = False
                    break
            if not feasible:
                continue
            
            chosen.append(i)
            res = find_subset(remaining // f, i + 1, end_idx, new_used, chosen, current_max)
            if res is not None:
                return res
            chosen.pop()
        
        return None
    
    # Iterate K in increasing order
    for k_idx in range(n_cands):
        if time.time() > deadline:
            break
        
        K = candidates[k_idx]
        end_idx = k_idx + 1
        
        # Quick check: product of all candidates[0..k_idx] must be >= target
        if suffix_prod[0] < target:
            continue
        
        # Try to find a subset
        sol = find_subset(target, 0, end_idx, (0,) * num_primes, [], K)
        if sol is not None:
            return [candidates[i] for i in sol]
    
    return best_factors
# EVOLVE-BLOCK-END
