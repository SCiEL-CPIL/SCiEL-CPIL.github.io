import time

# EVOLVE-BLOCK-START

def construct():
    t0 = time.time()
    deadline = 80.0
    
    P = 101
    D = list(range(1, 11))
    
    # Precompute modular inverses
    inv = [0] * 11
    for x in D:
        inv[x] = pow(x, -1, P)
    
    # min_size_for_residue[r] stores the minimum size of a subset summing to r
    # best_mask_for_residue[r] stores the mask of that subset
    min_size_for_residue = [101] * 101
    best_mask_for_residue = [0] * 101
    
    # Iterate over all non-empty subsets of D
    # There are 2^10 - 1 = 1023 subsets
    for mask in range(1, 1 << 10):
        res = 0
        size = 0
        # Calculate sum of inverses for this mask
        for x in D:
            if mask & (1 << (x - 1)):
                res = (res + inv[x]) % P
                size += 1
        
        # Update best subset for this residue if current size is smaller
        if size < min_size_for_residue[res]:
            min_size_for_residue[res] = size
            best_mask_for_residue[res] = mask
    
    # Find the minimum K such that all residues are covered by subsets of size <= K
    # Since we stored the minimum size for each residue, K is simply the max of these minimum sizes
    max_min_size = 0
    for r in range(101):
        if min_size_for_residue[r] > max_min_size:
            max_min_size = min_size_for_residue[r]
            
    # Construct the result lists
    result = []
    for r in range(101):
        mask = best_mask_for_residue[r]
        terms = []
        for x in D:
            if mask & (1 << (x - 1)):
                terms.append(x)
        result.append(terms)
        
    return result

# EVOLVE-BLOCK-END
