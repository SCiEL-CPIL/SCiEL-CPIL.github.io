def _is_valid(elements):
    n = len(elements)
    if n == 0:
        return True
    
    # Check for duplicate sums
    # If any duplicate, invalid
    sums = _all_subset_sums(elements)
    if len(sums) != (1 << n) - 1:
        return False
        
    max_sum = max(sums)
    sum_set = sums
    
    for a in sum_set:
        multiple = 2 * a
        while multiple <= max_sum:
            if multiple in sum_set:
                return False
            multiple += a
    return True
