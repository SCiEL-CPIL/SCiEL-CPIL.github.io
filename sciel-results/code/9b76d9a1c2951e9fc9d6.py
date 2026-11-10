def _get_all_sums(S):
    """Get all nonempty subset sums. Returns None if any collision."""
    sums = set()
    # Start with empty subset sum = 0
    current_sums = [0]
    for x in S:
        new_sums = [s + x for s in current_sums]
        for s in new_sums:
            if s in sums:
                return None
            sums.add(s)
        current_sums = new_sums
    
    # Remove 0 (empty subset)
    if 0 in sums:
        sums.discard(0)
    return sums
