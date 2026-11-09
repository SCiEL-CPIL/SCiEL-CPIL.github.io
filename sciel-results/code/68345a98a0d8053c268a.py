def construct():
    from fractions import Fraction
    import math
    import heapq

    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of denominators
    D = 1
    for d in denominators:
        D = D * d // math.gcd(D, d)
        
    # x[i] is the contribution of index i to the scaled sum
    x = [D // d for d in denominators]
    
    # Start state: deficit D, mask 0
    # We want to minimize deficit > 0
    
    # Priority queue: (deficit, mask)
    heap = [(D, 0)]
    visited = {0}
    
    best_deficit = D
    best_mask = 0
    
    # Limit on number of states to process to ensure time limit
    # 100,000 states should be fast enough
    limit = 100000
    processed = 0
    
    while heap and processed < limit:
        d, mask = heapq.heappop(heap)
        processed += 1
        
        # Update best if this deficit is smaller
        if d < best_deficit:
            best_deficit = d
            best_mask = mask
            
        # If we found a deficit of 1, it's the theoretical minimum
        if d == 1:
            break
            
        # Generate neighbors
        for i in range(n):
            if not (mask & (1 << i)):
                new_d = d - x[i]
                if new_d > 0:
                    new_mask = mask | (1 << i)
                    if new_mask not in visited:
                        visited.add(new_mask)
                        heapq.heappush(heap, (new_d, new_mask))
                        
    # Convert best_mask to list of indices
    result = []
    for i in range(n):
        if best_mask & (1 << i):
            result.append(i)
            
    return result
