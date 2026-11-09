def find_violation(S):
    """
    Find a violation in S. Returns x if x is subsumed by others, else None.
    """
    S_set = set(S)
    for x in S:
        target = L_weights[x]
        weights = [L_weights[y] for y in S_set if y != x]
        if subset_sum_exists(weights, target):
            return x
    return None
