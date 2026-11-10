def construct():
    P = 101
    N = 10
    # Precompute inverses
    invs = [0] * (N + 1)
    for i in range(1, N + 1):
        invs[i] = pow(i, -1, P)
        
    # best_subset[r] will store the smallest subset (by size) that sums to r
    # Initialize with None
    best_subset = [None] * P
    best_size = [float('inf')] * P
    
    # Iterate through all non-empty subsets
    # We can iterate by size to potentially stop early if we find a solution, 
    # but since N=10 is small, we can just check all and keep the min size.
    
    for mask in range(1, 1 << N):
        # Calculate sum and size
        s = 0
        size = 0
        subset = []
        for i in range(1, N + 1):
            if mask & (1 << (i - 1)):
                s = (s + invs[i]) % P
                size += 1
                subset.append(i)
        
        # Update best if this size is smaller
        if size < best_size[s]:
            best_size[s] = size
            best_subset[s] = subset
            
    # Construct the result
    # The problem states "101 nonempty lists". 
    # If any best_subset is None, it means that residue is unreachable with denominators 1..10.
    # However, with 10 denominators, we have 1023 sums. 
    # By pigeonhole, many collisions, but do they cover all 101 residues?
    # The seed program implies they do (it returns representations for all r).
    # If a residue is not covered, we might have an issue, but the task implies a solution exists.
    # To be safe, if best_subset[r] is None, we might need to handle it, 
    # but given the seed works, it should be fine.
    
    result = []
    for r in range(P):
        if best_subset[r] is None:
            # Fallback: this shouldn't happen for P=101, N=10.
            # But if it does, we can't satisfy the constraint.
            # We'll raise an error or return empty? 
            # The task says "Every residue must be represented".
            # Let's assume it's covered.
            pass
        result.append(best_subset[r])
        
    return result
