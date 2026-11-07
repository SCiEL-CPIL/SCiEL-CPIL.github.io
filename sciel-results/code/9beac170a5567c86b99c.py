"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def update(state, item):
    # The state is 1024 bytes.
    # Items are in range 0..399.
    # We use the first 400 bytes as an exact count array.
    # b[item] stores the count of item.
    # Max count is ~62, which fits in a uint8 (0-255).
    b = bytearray(state)
    b[item] = (b[item] + 1) & 0xFF
    return bytes(b)

def query(state, item):
    # Return the exact count stored at state[item].
    # The value is a byte (0-255), which is a non-negative integer.
    return state[item]
# EVOLVE-BLOCK-END
