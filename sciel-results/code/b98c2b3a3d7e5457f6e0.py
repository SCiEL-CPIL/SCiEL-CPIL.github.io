def _incremental_check(existing_sums, new_elem):
    """Check if adding new_elem keeps all subset sums valid.
    Returns the updated sums set if valid, None otherwise."""
    # New sums: {new_elem} ∪ {new_elem + s : s in existing_sums}
    # Check no collision with existing sums
    if new_elem in existing_sums:
        return None
    new_sums = {new_elem}
    for s in existing_sums:
        ns = s + new_elem
        if ns in existing_sums:
            return None
        new_sums.add(ns)
    
    all_sums = existing_sums | new_sums
    
    # Check divisibility among all sums
    s_list = sorted(all_sums)
    for i in range(len(s_list)):
        for j in range(i+1, len(s_list)):
            if s_list[j] % s_list[i] == 0:
                return None
    return all_sums
