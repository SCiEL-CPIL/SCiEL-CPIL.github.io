import math
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

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
        
    # Precompute Base value for each group (best number with no large prime)
    group_base = {}
    for sp, items in small_prime_groups.items():
        max_val_none = 0
        for lp, val in items:
            if lp is None:
                if val > max_val_none:
                    max_val_none = val
        group_base[sp] = max_val_none

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
        
    # Sum of all large primes
    sum_large_primes = sum(large_primes)
    
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
                
        # Base sum: 1 (for the number 1) + sum of all large primes
        # We will adjust this based on the assignments
        base_sum = 1 + sum_large_primes
        
        if not blocks_with_numbers:
            # No small prime blocks, just add all large primes (already in base_sum)
            # Wait, base_sum includes sum_large_primes.
            # If we don't use any small prime numbers, we just keep all large primes.
            # The numbers selected are [1] + large_primes.
            selected = [1] + large_primes
            return (base_sum, selected)
        
        # For each block, we have options:
        # 1. Use a number with no large prime (Gain = value - 0? No, relative to base)
        #    Base assumes we have all large primes as standalone.
        #    If we pick a number with no large prime, we add its value.
        #    Gain = value.
        # 2. Use a number with large prime p.
        #    We remove p from standalone set (loss p) and add value (gain value).
        #    Net Gain = value - p.
        
        # Let's calculate the "Base Gain" for each block if we don't use a large prime.
        # This is group_base[B].
        # If we use a large prime p, the gain is (value - p).
        # We want to maximize total gain.
        
        # We formulate this as an assignment problem.
        # Rows: blocks_with_numbers
        # Cols: large_primes that are "available" (appear in some block's options with positive delta) + Dummy Col
        
        # First, determine which large primes are relevant.
        # A large prime p is relevant if there is a block B and a number v in G_B with L_v={p}
        # such that (value - p) > group_base[B].
        
        relevant_primes = set()
        # Map from prime to index
        prime_to_idx = {}
        
        # For each block, find the best value for each large prime
        block_prime_best = {} # {block: {prime: best_value}}
        for B in blocks_with_numbers:
            block_prime_best[B] = {}
            for lp, val in small_prime_groups[B]:
                if lp is not None:
                    if lp not in block_prime_best[B]:
                        block_prime_best[B][lp] = val
                    else:
                        if val > block_prime_best[B][lp]:
                            block_prime_best[B][lp] = val
                    
                    # Check if this option is better than base
                    base_val = group_base[B]
                    if val - lp > base_val:
                        relevant_primes.add(lp)
                        
        if not relevant_primes:
            # No large prime assignment is beneficial
            # Just use the best number with no large prime for each block
            total_gain = 0
            selected = [1]
            for B in blocks_with_numbers:
                best_val = group_base[B]
                # Find the actual number
                # We need to select the number.
                # If group_base[B] > 0, there is a number.
                # We should pick the one with max value.
                # Since we stored group_base, we can just find the number.
                # But we need to ensure it's coprime with others?
                # Numbers with no large prime only share small primes.
                # Since blocks are disjoint in small primes, they are coprime.
                # So we can just pick the best one.
                
                # Find the number
                num_to_add = None
                for lp, val in small_prime_groups[B]:
                    if lp is None and val == best_val:
                        num_to_add = val
                        break
                if num_to_add is not None:
                    selected.append(num_to_add)
                    total_gain += num_to_add
                # If best_val is 0, we don't add anything.
                
            # Add all large primes
            for p in large_primes:
                selected.append(p)
                total_gain += p
                
            return (1 + total_gain, selected)
            
        # Sort relevant primes for consistent indexing
        sorted_relevant_primes = sorted(relevant_primes)
        for i, p in enumerate(sorted_relevant_primes):
            prime_to_idx[p] = i
            
        num_blocks = len(blocks_with_numbers)
        num_primes = len(sorted_relevant_primes)
        
        # Cost matrix for Hungarian Algorithm
        # Size: num_blocks x (num_primes + 1)
        # The +1 column is for "Don't assign to any relevant prime" (i.e., use base option)
        # Cost = -Gain
        
        cost_matrix = np.full((num_blocks, num_primes + 1), np.inf)
        
        # Fill in costs
        for i, B in enumerate(blocks_with_numbers):
            base_val = group_base[B]
            
            # Option: Use base (no large prime)
            # Gain = base_val
            # Cost = -base_val
            cost_matrix[i, num_primes] = -base_val
            
            # Option: Use a large prime
            for p, val in block_prime_best[B].items():
                if p in prime_to_idx:
                    j = prime_to_idx[p]
                    gain = val - p
                    # Only consider if gain > base_val?
                    # Actually, Hungarian will pick the best.
                    # If gain < base_val, the cost will be higher (less negative) than the base column.
                    # So it will naturally prefer the base column if base is better.
                    cost_matrix[i, j] = -gain
                    
        # Solve Assignment
        # linear_sum_assignment minimizes cost
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        total_gain = 0
        selected = [1]
        used_primes = set()
        
        # Map block index to block
        block_list = blocks_with_numbers
        
        # Determine assignments
        # For each row i, it is assigned to col j.
        # If j == num_primes, it means "Use Base".
        # Else, it means "Use Prime sorted_relevant_primes[j]".
        
        assignments = {} # {block: (type, value, prime)}
        
        for i, j in zip(row_ind, col_ind):
            B = block_list[i]
            base_val = group_base[B]
            
            if j == num_primes:
                # Use Base
                # Find the number
                num_to_add = None
                for lp, val in small_prime_groups[B]:
                    if lp is None and val == base_val:
                        num_to_add = val
                        break
                if num_to_add is not None:
                    assignments[B] = ('base', num_to_add, None)
                    total_gain += num_to_add
                # If base_val is 0, we don't add anything.
            else:
                p = sorted_relevant_primes[j]
                val = block_prime_best[B][p]
                assignments[B] = ('prime', val, p)
                total_gain += (val - p) # Net gain relative to having p standalone
                used_primes.add(p)
                
        # Construct the selected list
        # Add numbers from assignments
        for B, (typ, val, p) in assignments.items():
            selected.append(val)
            
        # Add remaining large primes (those not used in assignments)
        for p in large_primes:
            if p not in used_primes:
                selected.append(p)
                total_gain += p
                
        total_sum = 1 + total_gain
        
        return (total_sum, selected)
        
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
