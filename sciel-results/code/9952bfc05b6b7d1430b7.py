"""Program to find the smallest n in [1, 1000000] with phi(n) divisible by 120."""
# EVOLVE-BLOCK-START

def _compute_phi_array(limit):
    """Compute phi(n) for all n up to limit using a sieve method."""
    phi = list(range(limit + 1))
    for i in range(2, limit + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, limit + 1, i):
                phi[j] -= phi[j] // i
    return phi

def construct():
    """Return the smallest n in [1, 1000000] such that phi(n) % 120 == 0."""
    limit = 1000000
    phi = _compute_phi_array(limit)
    for n in range(1, limit + 1):
        if phi[n] % 120 == 0:
            return n
    return None
# EVOLVE-BLOCK-END
