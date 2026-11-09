import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    # B[k] is a bitset where bit s is 1 if sum s is achievable with exactly k elements
    B = [1]  # B[0] = 1 (sum 0 with 0 elements)
    elements = []
    
    for x in range(1, 257):
        if time.time() > deadline:
            break
        
        # Check if x can be added
        can_add = True
        n = len(B)  # Current max cardinality
        
        # Precompute Total = union of all B[j]
        Total = 0
        for j in range(n):
            Total |= B[j]
        
        # For each k, check if (B[k] << x) intersects with U_{k+1} = Total & ~B[k+1]
        # Note: B[k+1] might not exist yet if k+1 == n, so we treat it as 0
        for k in range(n):
            new_sums = B[k] << x
            # U_{k+1} is all sums not in B[k+1]
            # If k+1 < n, B[k+1] exists. If k+1 == n, B[k+1] is 0 (not yet created)
            if k + 1 < n:
                U = Total & ~B[k+1]
            else:
                U = Total  # Since B[k+1] doesn't exist, all sums are in U
            
            # Check intersection
            if new_sums & U:
                can_add = False
                break
        
        if can_add:
            # Add x to the set
            elements.append(x)
            # Update bitsets
            # New B[k+1] = B[k+1] | (B[k] << x)
            # We need to extend B if necessary
            if len(B) == n:
                B.append(0)  # B[n] = 0 initially
            
            for k in range(n - 1, -1, -1):
                B[k + 1] |= (B[k] << x)
    
    return elements
