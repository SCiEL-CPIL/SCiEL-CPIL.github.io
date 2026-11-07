import math
import time
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 80  # safety margin before 90s kill
    
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
    large_primes = [p for p in primes if p > 16]   # 17,19,...,256
    
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
    
    # Generate all set partitions of subsets of small_primes
    def get_all_partitions(elements):
        """Yield all partitions of subsets of elements (including empty)."""
        if not elements:
            yield []
            return
        first = elements[0]
        rest = elements[1:]
        
        # Get all partitions of subsets of rest
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
    
    # Precompute all skeleton configurations
    all_configs = set()
    for part in get_all_partitions(small_primes):
        normalized = tuple(sorted(tuple(sorted(block)) for block in part))
        all_configs.add(normalized)
    
    def solve_for_config(config):
        """
        config: tuple of tuples, each inner tuple is a subset of small_primes
        Returns: (total_sum, selected_numbers_list)
        """
        if time.time() > deadline:
            return (-1, [])
        
        # Determine which small primes are used in skeletons
        used_small = set()
        for block in config:
            for p in block:
                used_small.add(p)
        
        # Precompute small_prime_groups
        small_prime_groups = {}
        for v in range(2, N+1):
            sp, lp, val = num_info[v]
            if sp not in small_prime_groups:
                small_prime_groups[sp] = []
            large_p = None
            if lp:
                large_p = list(lp)[0]
            small_prime_groups[sp].append((large_p, val))
        
        for sp in small_prime_groups:
            small_prime_groups[sp].sort(key=lambda x: -x[1])
        
        used_large = set()
        selected = []
        total = 1  # include 1
        
        # Process each block in the config
        for block in config:
            B = frozenset(block)
            if B not in small_prime_groups:
                continue
            best_val = -1
            best_large = None
            for (lp, val) in small_prime_groups[B]:
                if lp is None or lp not in used_large:
                    if val > best_val:
                        best_val = val
                        best_large = lp
                    break
            if best_val > 0:
                selected.append(best_val)
                total += best_val
                if best_large is not None:
                    used_large.add(best_large)
        
        for p in large_primes:
            if p not in used_large:
                selected.append(p)
                total += p
                used_large.add(p)
        
        return (total, selected)
    
    # Find best from greedy approach
    best_total = -1
    best_selected = [1]
    
    for config in all_configs:
        if time.time() > deadline:
            break
        
        total, selected = solve_for_config(config)
        if total > best_total:
            best_total = total
            best_selected = selected
    
    # Local search polish: try to improve by swapping or adding numbers
    def is_coprime_pairwise(nums):
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                if math.gcd(nums[i], nums[j]) != 1:
                    return False
        return True
    
    def try_improve(selected, total):
        """Try to improve the solution via local search."""
        if time.time() > deadline:
            return selected, total
        
        selected_set = set(selected)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try adding a number
            for v in range(2, N+1):
                if v in selected_set:
                    continue
                # Check if v is coprime with all selected
                compatible = all(math.gcd(v, s) == 1 for s in selected)
                if compatible:
                    new_selected = selected + [v]
                    new_total = total + v
                    # Verify
                    if is_coprime_pairwise(new_selected):
                        selected = new_selected
                        selected_set = set(selected)
                        total = new_total
                        improved = True
            
            # Try swapping: remove one, add one or two
            for i in range(len(selected)):
                if time.time() > deadline:
                    break
                removed = selected[i]
                remaining = [x for j, x in enumerate(selected) if j != i]
                remaining_set = set(remaining)
                
                # Try adding one number
                for v in range(2, N+1):
                    if v in remaining_set:
                        continue
                    if v == removed:
                        continue
                    compatible = all(math.gcd(v, s) == 1 for s in remaining)
                    if compatible and v > removed:
                        new_selected = remaining + [v]
                        new_total = total - removed + v
                        if is_coprime_pairwise(new_selected):
                            selected = new_selected
                            selected_set = set(selected)
                            total = new_total
                            improved = True
                            break
                
                # Try adding two numbers (replace one with two)
                if time.time() > deadline:
                    break
                for v1 in range(2, N+1):
                    if v1 in remaining_set or v1 == removed:
                        continue
                    if any(math.gcd(v1, s) != 1 for s in remaining):
                        continue
                    for v2 in range(2, N+1):
                        if v2 in remaining_set or v2 == removed or v2 == v1:
                            continue
                        if any(math.gcd(v2, s) != 1 for s in remaining):
                            continue
                        if math.gcd(v1, v2) != 1:
                            continue
                        if v1 + v2 > removed:
                            new_selected = remaining + [v1, v2]
                            new_total = total - removed + v1 + v2
                            if is_coprime_pairwise(new_selected):
                                selected = new_selected
                                selected_set = set(selected)
                                total = new_total
                                improved = True
                                break
                    if improved:
                        break
        
        return selected, total
    
    # Apply local search improvement
    if time.time() < deadline - 5:
        best_selected, best_total = try_improve(best_selected, best_total)
    
    # Final verification
    if not is_coprime_pairwise(best_selected):
        best_selected = [1]
        best_total = 1
    
    return [int(x) for x in best_selected]
