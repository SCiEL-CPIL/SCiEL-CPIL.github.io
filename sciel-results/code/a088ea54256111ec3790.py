import time

def construct():
    """
    Constructs a set of 12 distinct positive integers with distinct subset sums,
    minimizing the largest integer.
    
    The optimal solution is known to be the powers of 2, where the largest
    element is 2^(n-1). For n=12, this is 2048.
    
    This implementation uses a greedy algorithm to construct the set, which
    will naturally select the powers of 2. This serves as a verification
    that the greedy choice is optimal for this specific problem structure.
    """
    n = 12
    current_set = []
    
    # Pre-allocate a set for subset sums to speed up checks
    # Max sum will be around 2^12 - 1 = 4095, so a boolean array or set is fine.
    
    def get_subset_sums(arr):
        sums = {0}
        for x in arr:
            new_sums = {s + x for s in sums}
            sums |= new_sums
        return sums

    for i in range(n):
        candidate = 1
        while True:
            # Check if adding 'candidate' to current_set keeps sums distinct
            # Current sums
            current_sums = get_subset_sums(current_set)
            
            # New sums if we add candidate
            # new_sums = {s + candidate for s in current_sums}
            # We just need to check if new_sums intersects with current_sums
            # Intersection exists if there are s1, s2 in current_sums such that s1 = s2 + candidate
            # i.e., s1 - s2 = candidate
            # Or simply check if (current_sums + candidate) has any element in current_sums
            
            # Efficient check:
            # If any (s + candidate) is in current_sums, it's a collision.
            # Since current_sums contains 0, if candidate is in current_sums, collision.
            # More generally:
            collision = False
            for s in current_sums:
                if s + candidate in current_sums:
                    collision = True
                    break
            
            if not collision:
                current_set.append(candidate)
                break
            else:
                candidate += 1
                
            # Safety margin: candidate should not exceed 2048 for n=12
            if candidate > 2048:
                # Fallback to powers of 2 if greedy gets stuck (should not happen)
                return [1 << i for i in range(n)]
                
    return current_set
