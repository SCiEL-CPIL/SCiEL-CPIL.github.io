# EVOLVE-BLOCK-START
import sys

def construct():
    """
    Returns a strictly increasing sequence of integers in [2, 128] 
    maximizing length such that all contiguous block products are distinct.
    Uses Backtracking Search with prime exponent vectors and strong pruning.
    """
    
    # Primes up to 128
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    num_primes = len(primes)
    
    # Precompute factor vectors
    factor_vecs = {}
    for n in range(2, 129):
        n_tmp = n
        vec = [0] * num_primes
        for i, p in enumerate(primes):
            while n_tmp % p == 0:
                vec[i] += 1
                n_tmp //= p
            if n_tmp == 1:
                break
        factor_vecs[n] = tuple(vec)
    
    zero_vec = tuple([0] * num_primes)
    
    # Global best solution
    best_len = 0
    best_seq = []
    
    def dfs(idx, current_values, current_prefixes, current_products):
        nonlocal best_len, best_seq
        
        # idx is the next number to consider
        if idx > 128:
            if len(current_values) > best_len:
                best_len = len(current_values)
                best_seq = current_values[:]
            return
        
        # Pruning: Upper bound on possible length
        # Remaining numbers: 128 - idx + 1
        if len(current_values) + (128 - idx + 1) <= best_len:
            return

        # Try to include idx first (heuristic: aim for max length quickly)
        x = idx
        x_vec = factor_vecs[x]
        last_prefix = current_prefixes[-1]
        new_prefix = tuple(last_prefix[i] + x_vec[i] for i in range(num_primes))
        
        k = len(current_values)
        valid = True
        new_blocks = []
        for j in range(k + 1):
            old_prefix = current_prefixes[j]
            block_vec = tuple(new_prefix[i] - old_prefix[i] for i in range(num_primes))
            if block_vec in current_products:
                valid = False
                break
            new_blocks.append(block_vec)
            
        if valid:
            new_values = current_values + [x]
            new_prefixes = current_prefixes + [new_prefix]
            new_products = set(current_products)
            new_products.update(new_blocks)
            
            dfs(idx + 1, new_values, new_prefixes, new_products)
            
        # Try to skip idx
        dfs(idx + 1, current_values, current_prefixes, current_products)
        
    dfs(2, [], [zero_vec], set())
    
    return best_seq

# EVOLVE-BLOCK-END
