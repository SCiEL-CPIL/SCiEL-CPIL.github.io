"""
OneMax 100-bit direct construction.
Returns the global optimum: a 100-character string of '1's.
"""
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Return the optimal 100-bit string for OneMax: all bits set to 1.
    The global optimum for OneMax is trivially all ones.
    """
    t0 = time.monotonic()
    t_limit = t0 + 145.0

    # Direct construction of the global optimum: 100 ones.
    # This is an O(1) operation in terms of logical complexity (string multiplication).
    # No search, relaxation, or iterative refinement is needed or beneficial.
    candidate = "1" * 100

    # Explicit deadline guard: check elapsed time against a safety margin.
    # The task limit is 150s. We use a 5s margin, so 145s.
    # This check is defensive; the operation above is instantaneous.
    if time.monotonic() < t_limit:
        return candidate

    return candidate
# EVOLVE-BLOCK-END
