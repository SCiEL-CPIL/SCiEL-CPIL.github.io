import itertools

def construct():
    # Precompute modular inverses of 1..10 modulo 101
    p = 101
    invs = [0] * 11
    for x in range(1, 11):
        invs[x] = pow(x, -1, p)
    
    # For each residue 0..100, we want the shortest list of distinct denominators in 1..10
    # whose inverses sum to that residue mod 101.
    # We do this by iterating over all subsets of {1..10} (2^10 = 1024 subsets) and
    # recording the minimum size subset that achieves each residue.
    
    best_len = [11] * p  # 11 is impossible (max is 10)
    best_subset = [None] * p
    
    # Iterate over all masks from 0 to 1023
    for mask in range(1, 1 << 10):
        # Count bits and compute sum of inverses
        bits = bin(mask).count('1')
        s = 0
        m = mask
        x = 1
        while m:
            if m & 1:
                s += invs[x]
            x += 1
            m >>= 1
        residue = s % p
        
        if bits < best_len[residue]:
            best_len[residue] = bits
            # Reconstruct the subset
            subset = []
            m = mask
            x = 1
            while m:
                if m & 1:
                    subset.append(x)
                x += 1
                m >>= 1
            best_subset[residue] = subset
    
    # All residues should be covered. If any are not, they'd have best_len 11,
    # but with 1024 subsets and 101 residues, pigeonhole guarantees coverage.
    return best_subset
