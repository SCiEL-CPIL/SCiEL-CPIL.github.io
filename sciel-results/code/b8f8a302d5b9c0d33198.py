import itertools
import time

def construct():
    n = 17
    # Differences are 1..8, paired with 17-d
    # We partition {1..8} into 4 sets of size 2
    # Actually, the sets can be of any size, but sum of sizes must be 8.
    # To maximize symmetry and chance, pairs are good.
    # But let's try all partitions of {1..8} into 4 non-empty subsets.
    # Number of partitions is small.
    
    def has_c6_circulant(D):
        # D is a set of allowed steps (integers 1..16)
        # Check for a cycle of length 6
        # s0=0, s1..s5 distinct and non-zero, s6=0
        # Steps d1..d6 in D
        # Note: The cycle can traverse steps in any order.
        # We fix the first step to break rotational symmetry? No, D is fixed.
        # We just check all sequences.
        for d1 in D:
            s1 = d1 % 17
            if s1 == 0: continue
            for d2 in D:
                s2 = (s1 + d2) % 17
                if s2 == 0 or s2 == s1: continue
                for d3 in D:
                    s3 = (s2 + d3) % 17
                    if s3 == 0 or s3 in (s1, s2): continue
                    for d4 in D:
                        s4 = (s3 + d4) % 17
                        if s4 == 0 or s4 in (s1, s2, s3): continue
                        for d5 in D:
                            s5 = (s4 + d5) % 17
                            if s5 == 0 or s5 in (s1, s2, s3, s4): continue
                            for d6 in D:
                                s6 = (s5 + d6) % 17
                                if s6 == 0:
                                    # Check if s1..s5 are distinct and non-zero
                                    # We already checked distinctness in loop
                                    # s6=0 means we are back to start
                                    # We must ensure the path is a simple cycle
                                    # The condition s1..s5 distinct and non-zero ensures vertices 0, s1, s2, s3, s4, s5 are distinct
                                    return True
        return False

    # Generate all partitions of {1..8} into 4 subsets
    # We can represent a partition as a tuple of 4 sets
    # To avoid duplicates, we can use a canonical representation
    # Or just iterate and check, and break on first valid one
    
    items = list(range(1, 9))
    
    # We will try to find a valid partition
    # Since the number of partitions is small, we can just try them all
    
    # A partition is a list of 4 sets.
    # We can generate them by assigning each item to one of 4 bins, then normalizing.
    # But 4^8 = 65536. This is fast enough.
    
    best_partition = None
    for assignment in itertools.product(range(4), repeat=8):
        # assignment[i] is the bin for item i+1
        bins = [set() for _ in range(4)]
        for i, bin_idx in enumerate(assignment):
            bins[bin_idx].add(i + 1)
        
        # Check if all bins are non-empty
        if any(len(b) == 0 for b in bins):
            continue
            
        # Check if any bin has a C6
        valid = True
        for b in bins:
            D = set()
            for d in b:
                D.add(d)
                D.add(17 - d)
            if has_c6_circulant(D):
                valid = False
                break
        
        if valid:
            best_partition = bins
            break
            
    if best_partition is None:
        # Fallback to n=9 if no partition found (should not happen)
        n = 9
        return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
        
    # Construct the coloring
    colors = [0] * (n * (n - 1) // 2)
    # Map edge (i,j) to color
    # Edge (i,j) with i<j has difference d = (j-i) % 17
    # If d > 8, use 17-d
    # Find which bin contains d or 17-d
    
    # Create a lookup table for difference -> color
    diff_to_color = {}
    for c, b in enumerate(best_partition):
        for d in b:
            diff_to_color[d] = c
            diff_to_color[17 - d] = c
            
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % 17
            if d == 0: d = 17 # Not possible for i<j
            # d is in 1..16
            colors[idx] = diff_to_color[d]
            idx += 1
            
    return {'n': n, 'colors': colors}
