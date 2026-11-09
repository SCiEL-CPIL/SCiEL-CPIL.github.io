"""OneMax direct binary construction (100 bits).

Constructs a 100-bit binary string maximizing ones.
The global optimum is all ones, giving a score of 100.
"""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    Returns a string of N '1' characters, which is the unique global optimum
    for the OneMax objective with N bits.

    Uses a structural change: represents the solution as a large integer
    and formats it, rather than using string repetition.
    """
    t0 = time.monotonic()
    # Safety margin: return well before the 150s wall-clock kill.
    deadline = t0 + 140.0

    # Structural Change: Use integer bit manipulation.
    # Create an integer with N bits set to 1.
    # (1 << N) - 1 creates a number with N ones in binary.
    val = (1 << N) - 1
    
    # Format the integer as a binary string of length N, zero-padded.
    # Since val has N ones, it will be a string of N '1's.
    optimal = format(val, '0{}b'.format(N))

    # Deadline guard: return the result if within time budget.
    if time.monotonic() < deadline:
        return optimal

    # Fallback (unreachable for O(1) work, but satisfies contract):
    return optimal
# EVOLVE-BLOCK-END
