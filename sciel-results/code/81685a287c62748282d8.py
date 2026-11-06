def construct():
    """
    Constructs a set of n=12 distinct positive integers such that all 
    subset sums are distinct, minimizing the largest integer.
    
    This implementation uses a greedy algorithm that iteratively selects 
    the smallest positive integer not yet used that preserves the 
    distinctness of all current subset sums.
    """
    n = 12
    current_set = []
    subset_sums = {0}
    
    for _ in range(n):
        candidate = 1
        while True:
            # Check if candidate is distinct from already chosen numbers
            if candidate in current_set:
                candidate += 1
                continue
            
            # Calculate new subset sums if we add candidate
            new_sums = {s + candidate for s in subset_sums}
            
            # Check for collision with existing subset sums
            if not new_sums.intersection(subset_sums):
                # No collision, accept candidate
                current_set.append(candidate)
                subset_sums = subset_sums.union(new_sums)
                break
            else:
                candidate += 1
                
    return current_set
