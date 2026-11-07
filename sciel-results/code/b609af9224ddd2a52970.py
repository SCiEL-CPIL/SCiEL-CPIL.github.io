"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import product

P = {'n': 256}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    n = P['n']
    
    # Identify small and large primes
    # Small primes: p <= sqrt(256) = 16
    small_primes = [2, 3, 5, 7, 11, 13]
    
    # Generate all large primes > 16 up to 256
    def is_prime(x):
        if x < 2: return False
        if x == 2: return True
        if x % 2 == 0: return False
        for i in range(3, int(x**0.5) + 1, 2):
            if x % i == 0:
                return False
        return True
        
    large_primes = [x for x in range(17, n + 1) if is_prime(x)]
    large_primes.sort(reverse=True) # Largest first
    M = len(large_primes)
    
    # Precompute candidates for each subset of small primes
    # Subset represented by bitmask
    # For a given mask, find all c <= 256 composed of primes in mask
    # And for each c, find the best large prime q to pair with it (largest q <= 256/c)
    # Or q=1 (standalone)
    
    # Actually, we don't need to precompute for all masks.
    # We will iterate over partitions.
    
    # To iterate over partitions of the set of small primes (indices 0..5)
    # We can use a recursive generator or just iterate all masks and check disjointness?
    # Number of primes is 6.
    # We can represent the partition as a list of masks.
    
    # Better: Iterate over all ways to assign each small prime to a "group".
    # There are G groups. This is complex.
    
    # Alternative: Iterate over all subsets of small primes for the first group,
    # then all subsets of remaining for the second, etc.
    
    # Let's use a recursive search over the indices of small primes.
    
    best_sum = 0
    best_config = []
    
    # Sort small primes to ensure consistent ordering? Not strictly needed.
    
    def get_candidates(mask):
        """
        Return list of (c, best_q) for a given mask of small primes.
        c is the small part.
        best_q is the largest large prime <= 256/c, or 1 if we choose standalone.
        Actually, we return a list of options: (c, q) where q is a specific large prime index or -1 for standalone.
        But we need to ensure distinct large primes.
        
        Let's return a list of (c, max_q_val, max_q_idx).
        If we pick this option, we consume large prime max_q_idx.
        """
        options = []
        # Generate all c from primes in mask
        primes_in_mask = [small_primes[i] for i in range(6) if mask & (1 << i)]
        
        # Start with c=1
        # We need to generate all products of powers of these primes <= 256
        # Since number of primes is small, we can use recursion or nested loops.
        # Max 6 primes.
        
        # Recursive generation of c
        def gen_c(idx, current_c):
            if idx == len(primes_in_mask):
                # Add option (current_c, ...)
                # Find largest q in large_primes such that current_c * q <= 256
                # If current_c * 1 <= 256, standalone is always an option (q=1)
                # We want to consider both standalone and paired.
                
                # Standalone: value = current_c, q_idx = -1
                options.append((current_c, -1, current_c)) # c, q_idx, value
                
                # Paired:
                if current_c > 1: # If c=1, pairing with q is just q, which is covered by standalone large primes later?
                                 # No, if we use c=1 in a group, it means we "assign" the group to a large prime.
                                 # But c=1 doesn't consume small primes.
                                 # Our partition logic assigns primes to groups.
                                 # A group with no primes (mask=0) is not part of the partition of the 6 primes.
                                 # So c will always be > 1 if mask != 0.
                                 # If mask=0, we don't form a number from small primes.
                                 # So we only call this for non-empty masks?
                                 # In the partition recursion, we pick non-empty subsets.
                                 
                    # Find largest q
                    max_q_val = 0
                    max_q_idx = -1
                    for q_idx, q_val in enumerate(large_primes):
                        if current_c * q_val <= 256:
                            max_q_val = q_val
                            max_q_idx = q_idx
                            break # Since sorted desc, first is largest
                        else:
                            continue
                    
                    if max_q_val > 0:
                        options.append((current_c, max_q_idx, current_c * max_q_val))
                return
            
            p = primes_in_mask[idx]
            power = 1
            while current_c * power <= 256:
                gen_c(idx + 1, current_c * power)
                power *= p
                if power > 256:
                    break
        
        if primes_in_mask:
            gen_c(0, 1)
        else:
            # Should not happen in partition of non-empty set
            options.append((1, -1, 1))
            
        return options

    # Recursive partition search
    # used_mask: bitmask of small primes already assigned
    # used_large_mask: bitmask of large primes already assigned (M bits)
    # current_sum: sum so far
    # current_numbers: list of numbers chosen
    
    # M can be up to 48. Bitmask of 48 bits is fine in Python int.
    
    def search(used_small_mask, used_large_mask, current_sum, current_numbers, depth):
        nonlocal best_sum, best_config
        
        if time.time() > deadline:
            return
            
        if used_small_mask == (1 << 6) - 1:
            # All small primes assigned.
            # Add all remaining large primes as standalone numbers
            total = current_sum
            final_numbers = list(current_numbers)
            for q_idx, q_val in enumerate(large_primes):
                if not (used_large_mask & (1 << q_idx)):
                    total += q_val
                    final_numbers.append(q_val)
            
            if total > best_sum:
                best_sum = total
                best_config = final_numbers
            return
            
        # Find the first unused small prime
        first_unused = -1
        for i in range(6):
            if not (used_small_mask & (1 << i)):
                first_unused = i
                break
                
        if first_unused == -1:
            return
            
        # Iterate over all subsets of remaining small primes that include first_unused
        # to form the next group
        remaining_mask = ((1 << 6) - 1) ^ used_small_mask
        # Subsets of remaining_mask that contain bit first_unused
        # We can iterate sub = remaining_mask; sub; sub = (sub-1) & remaining_mask
        # But we must include first_unused.
        
        # Get list of indices in remaining_mask
        rem_indices = [i for i in range(6) if remaining_mask & (1 << i)]
        
        # Generate all subsets of rem_indices that contain first_unused
        # This is equivalent to generating all subsets of rem_indices \ {first_unused} and adding first_unused
        
        others = [i for i in rem_indices if i != first_unused]
        
        # Iterate over all masks for 'others'
        # Number of others is at most 5. 2^5 = 32.
        
        # Use recursion or loop
        def gen_subsets(indices, idx, current_sub_mask):
            if idx == len(indices):
                full_sub_mask = current_sub_mask | (1 << first_unused)
                
                # Get candidates for this group
                candidates = get_candidates(full_sub_mask)
                
                for c, q_idx, val in candidates:
                    if q_idx != -1:
                        # Check if large prime is available
                        if used_large_mask & (1 << q_idx):
                            continue
                        new_large_mask = used_large_mask | (1 << q_idx)
                    else:
                        new_large_mask = used_large_mask
                        q_idx = -1 # Not used
                    
                    new_small_mask = used_small_mask | full_sub_mask
                    new_sum = current_sum + val
                    new_numbers = current_numbers + [val] # Store value for now, reconstruct later?
                    # We need to store the actual number for the final list.
                    # val is the number.
                    
                    search(new_small_mask, new_large_mask, new_sum, new_numbers, depth + 1)
                return
            
            gen_subsets(indices, idx + 1, current_sub_mask)
            gen_subsets(indices, idx + 1, current_sub_mask | (1 << indices[idx]))
            
        gen_subsets(others, 0, 0)
        
    search(0, 0, 0, [], 0)
    
    if not best_config:
        # Fallback
        return [1] + [x for x in range(2, 257) if is_prime(x)]
        
    # Convert to int and ensure list of ints
    return [int(x) for x in best_config]

# EVOLVE-BLOCK-END
