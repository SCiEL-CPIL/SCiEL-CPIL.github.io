import time

N = 100

# Pre-compute the optimal solution at import time.
# Strategy: Model the binary string as a 128-bit integer by shifting a 1 left 127 times 
# and masking with (1 << 128) - 1, then extract the rightmost 100 bits.
# This leverages Python's native big-integer arithmetic.
# However, for maximum safety and speed, we just use the result of this logic 
# or the simpler equivalent, pre-computed.
# Let's follow the directive's specific generation logic for the constant.
_MASK = (1 << 128) - 1
_VAL = (1 << 127) # This is not quite right for all ones.
# The directive says: "shifting a 1 left 127 times and masking... then extract rightmost 100 bits"
# Shifting 1 left 127 times gives 2^127. This is a single bit.
# To get all ones, we need (1 << 128) - 1.
# Let's re-read: "Model the binary string as a 128-bit integer by shifting a 1 left 127 times and masking with (1 << 128) - 1"
# This phrasing is slightly ambiguous. Shifting 1 left 127 times creates a number with a single 1 at the 127th position.
# Masking it with (1<<128)-1 doesn't change it.
# Extracting the rightmost 100 bits of 2^127 would give a string of zeros followed by a 1 at the end?
# No, bin(2^127) is '1' followed by 127 zeros.
# The rightmost 100 bits of that would be '0' * 127 + '1' -> wait.
# bin(2^127) = '0b1' + '0'*127.
# Rightmost 100 bits: '0'*127 + '1' is 128 bits.
# Rightmost 100 bits are '0' * 127 ... no.
# Let's assume the directive meant to generate the all-ones pattern.
# The all-ones pattern for 100 bits is (1 << 100) - 1.
# bin((1 << 100) - 1) gives '0b' + '1'*100.
# So bin((1 << 100) - 1)[-100:] gives '1'*100.

# Let's stick to the most robust and simple method for the constant, 
# but acknowledge the integer approach in comments or logic if it doesn't break correctness.
# Actually, the directive says "This leverages Python's native big-integer arithmetic to generate the all-ones pattern in a single low-level operation".
# (1 << 100) - 1 is the standard way to get a 100-bit mask of all ones.
# Let's use that.

_OPTIMAL_STR = bin((1 << N) - 1)[-N:]

# Deadline guard setup
_START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0

def construct():
    """
    Return a binary string of length N maximizing the number of ones.
    
    This implementation uses direct construction of the known global optimum.
    The result is pre-computed at module load time for O(1) retrieval.
    An explicit deadline guard is included to ensure compliance with wall-clock limits.
    """
    # Check deadline guard
    if time.time() - _START_TIME > (TIME_LIMIT - SAFETY_MARGIN):
        return _OPTIMAL_STR
    
    return _OPTIMAL_STR
