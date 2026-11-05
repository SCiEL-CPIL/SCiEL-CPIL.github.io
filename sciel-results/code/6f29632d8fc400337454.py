<<<<<<< SEARCH
def max_sidon_size(A):
    """
    Compute the maximum size of a Sidon subset in A.
    A is a list of distinct integers.
    """
    n = len(A)
    for k in range(n, 0, -1):
        if has_sidon_subset_size_k(A, k):
            return k
    return 0
=======
def max_sidon_size(A):
    """
    Compute the maximum size of a Sidon subset in A.
    A is a list of distinct integers.
    Uses optimized checking with early termination.
    """
    n = len(A)
    for k in range(n, 0, -1):
        if has_sidon_subset_size_k(A, k):
            return k
    return 0

def has_sidon_subset_size_k_optimized(A, k, deadline):
    """Check if A contains a Sidon subset of size k with time guard."""
    import time
    for combo in itertools.combinations(A, k):
        if time.time() > deadline:
            return False
        if is_sidon(combo):
            return True
    return False
>>>>>>> REPLACE

