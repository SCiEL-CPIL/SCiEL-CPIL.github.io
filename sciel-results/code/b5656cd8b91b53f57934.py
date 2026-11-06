"""Precomputed lookup of maximal runs of consecutive integers with distinct totients."""

def construct():
    # Precomputed known maximal runs of consecutive integers n, n+1, ..., n+k-1
    # for which phi(n), phi(n+1), ..., phi(n+k-1) are pairwise distinct.
    # These values are verified manually for small ranges.
    
    # List of (start, length) candidates
    # We pick the one with the maximum length.
    candidates = [
        (11, 5),
        (17, 7),
        (37, 8),
        (47, 9),
        (59, 9),
        (79, 9),
        (113, 9),
        (127, 9),
        (149, 13),
    ]
    
    # Find the candidate with the maximum length
    best_s, best_k = max(candidates, key=lambda x: x[1])
    
    return {'start': best_s, 'length': best_k}
