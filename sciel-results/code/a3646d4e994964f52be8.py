import itertools

def max_sidon_size(S):
    """
    Computes the size of the largest subset of S with all unordered pair sums unique.
    S is a list of 12 distinct integers.
    """
    n = len(S)
    # Precompute all pair sums for the whole set? No, we need subsets.
    # Since n=12, we can iterate over subsets by size, starting from largest.
    # Max possible size is 12.
    # We check if there exists a Sidon subset of size k.
    
    # Helper to check if a list is Sidon
    def is_sidon(subset):
        sums = set()
        for i in range(len(subset)):
            for j in range(i, len(subset)):
                s = subset[i] + subset[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    # Check sizes from 12 down to 1
    for k in range(n, 0, -1):
        for subset in itertools.combinations(S, k):
            if is_sidon(list(subset)):
                return k
    return 1

def construct():
    # Candidate 1: The seed
    candidates = [
        list(range(12)),
        # Some other dense sets
        list(range(0, 12)),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12],
        [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12],
        [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
        [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12],
        [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12],
        [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    ]
    
    best_set = candidates[0]
    best_score = float('inf')
    
    for c in candidates:
        # Ensure distinct and in range
        if len(set(c)) != 12:
            continue
        if min(c) < 0 or max(c) > 128:
            continue
            
        size = max_sidon_size(c)
        if size < best_score:
            best_score = size
            best_set = c
            
    return best_set
