import math
import time

def construct():
    start = time.time()
    deadline = start + 85  # safety margin before 90s kill
    
    N = 256
    
    # Sieve for primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N+1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N+1) if is_prime[i]]
    
    small_primes = [p for p in primes if p <= 16]  # 2,3,5,7,11,13
    large_primes = [p for p in primes if p > 16]   # 17,19,...,251
    
    # For each number 2..256, compute its prime factorization (set of primes)
    def factorize(x):
        factors = set()
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                factors.add(p)
                while x % p == 0:
                    x //= p
        if x > 1:
            factors.add(x)
        return factors
    
    # For each number, store: (small_prime_set, large_prime_set, value)
    num_info = {}
    for v in range(2, N+1):
        fac = factorize(v)
        small_part = frozenset(p for p in fac if p in small_primes)
        large_part = frozenset(p for p in fac if p in large_primes)
        num_info[v] = (small_part, large_part, v)
    
    # Precompute for each subset S of small_primes, the list of numbers with small_prime_set = S
    # Each entry: (large_prime_or_None, value)
    small_prime_groups = {}
    for v in range(2, N+1):
        sp, lp, val = num_info[v]
        if sp not in small_prime_groups:
            small_prime_groups[sp] = []
        large_p = None
        if lp:
            large_p = list(lp)[0]
        small_prime_groups[sp].append((large_p, val))
    
    # For each group, sort by value descending
    for sp in small_prime_groups:
        small_prime_groups[sp].sort(key=lambda x: -x[1])
    
    # Generate all set partitions of subsets of small_primes
    def get_all_partitions(elements):
        """Yield all partitions of subsets of elements (including empty)."""
        if not elements:
            yield []
            return
        first = elements[0]
        rest = elements[1:]
        
        for part in get_all_partitions(rest):
            # Option 1: don't include first
            yield part
            
            # Option 2: include first in a new block
            yield part + [[first]]
            
            # Option 3: include first in an existing block
            for i in range(len(part)):
                new_part = [list(b) for b in part]
                new_part[i] = new_part[i] + [first]
                yield new_part
    
    # Precompute all skeleton configurations (deduplicated)
    all_configs = set()
    for part in get_all_partitions(small_primes):
        normalized = tuple(sorted(tuple(sorted(block)) for block in part))
        all_configs.add(normalized)
    
    # For a given config, solve the assignment problem:
    # - Each block in config gets one number from small_prime_groups[block]
    # - No large prime can be used twice
    # - Remaining large primes can be used as standalone numbers
    # We use a greedy approach with backtracking for small conflicts
    
    def solve_for_config(config):
        """
        config: tuple of tuples, each inner tuple is a subset of small_primes
        Returns: (total_sum, selected_numbers_list)
        """
        if time.time() > deadline:
            return (-1, [])
        
        # Get the blocks that have available numbers
        blocks_with_numbers = []
        for block in config:
            B = frozenset(block)
            if B in small_prime_groups and small_prime_groups[B]:
                blocks_with_numbers.append(B)
        
        if not blocks_with_numbers:
            # No small prime blocks, just add all large primes
            selected = [1]
            total = 1
            for p in large_primes:
                selected.append(p)
                total += p
            return (total, selected)
        
        # For each block, get the best number (by value) and track large prime usage
        # We'll use a greedy approach: process blocks in order of their best value
        # For ties or conflicts, try alternatives
        
        # Sort blocks by their best available value (descending)
        block_best = {}
        for B in blocks_with_numbers:
            # Find best value among all options (ignoring conflicts for now)
            best_val = max(val for (lp, val) in small_prime_groups[B])
            block_best[B] = best_val
        
        sorted_blocks = sorted(blocks_with_numbers, key=lambda B: -block_best[B])
        
        # Greedy assignment with conflict resolution
        used_large = set()
        selected = [1]
        total = 1
        
        for B in sorted_blocks:
            # Try to find the best number for this block that doesn't conflict
            assigned = False
            for (lp, val) in small_prime_groups[B]:
                if lp is None or lp not in used_large:
                    selected.append(val)
                    total += val
                    if lp is not None:
                        used_large.add(lp)
                    assigned = True
                    break
            
            if not assigned:
                # No valid number for this block; skip it (shouldn't happen often)
                pass
        
        # Add remaining large primes as standalone numbers
        for p in large_primes:
            if p not in used_large:
                selected.append(p)
                total += p
                used_large.add(p)
        
        return (total, selected)
    
    # Now iterate over all configs and find the best
    best_total = -1
    best_selected = [1]
    
    for config in all_configs:
        if time.time() > deadline:
            break
        
        total, selected = solve_for_config(config)
        if total > best_total:
            best_total = total
            best_selected = selected
    
    # Verify the result
    for i in range(len(best_selected)):
        for j in range(i+1, len(best_selected)):
            if math.gcd(best_selected[i], best_selected[j]) != 1:
                best_selected = [1]
                best_total = 1
                break
    
    return [int(x) for x in best_selected]
