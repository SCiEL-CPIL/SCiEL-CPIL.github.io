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
    # Capture start time for the deadline guard
    t_start = time.monotonic()
    # Safety margin: task limit is 150s, use 148s as the hard stop for returning
    deadline = t_start + 148.0

    # Direct construction of the global optimum: 100 ones.
    # This is an O(1) operation in terms of logical complexity (string multiplication).
    # No search, relaxation, or iterative refinement is needed or beneficial.
    # We construct the string explicitly to ensure it meets the interface requirements.
    n_bits = 100
    best_candidate = "1" * n_bits

    # Explicit deadline guard: check elapsed time against a safety margin.
    # This check is defensive; the operation above is instantaneous.
    # If somehow we are close to the limit (which is not possible for this operation),
    # we return the best candidate found (which is the optimal one).
    if time.monotonic() > deadline:
        return best_candidate

    # Return the optimal binary string
    return best_candidate
# EVOLVE-BLOCK-END
