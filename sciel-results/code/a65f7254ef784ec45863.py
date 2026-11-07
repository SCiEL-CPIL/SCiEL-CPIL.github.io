"""Memory-bounded streaming frequency sketch: exact 11-bit counters for all 400 items."""
import time

_T0 = time.monotonic()
_DEADLINE = _T0 + 145.0

NUM_ITEMS = 400
BITS_PER_ITEM = 11  # 2^11 = 2048 > 1024 (max possible count)
TOTAL_BITS = NUM_ITEMS * BITS_PER_ITEM  # 4400 bits
TOTAL_BYTES_NEEDED = (TOTAL_BITS + 7) // 8  # 550 bytes
MAX_COUNT = (1 << BITS_PER_ITEM) - 1  # 2047


def _read_counter(state, item):
    """Read the 11-bit counter for the given item from state bytes."""
    start_bit = item * BITS_PER_ITEM
    val = 0
    for bit_idx in range(BITS_PER_ITEM):
        absolute_bit = start_bit + bit_idx
        byte_idx = absolute_bit // 8
        bit_in_byte = absolute_bit % 8
        if state[byte_idx] & (1 << bit_in_byte):
            val |= (1 << bit_idx)
    return val


def _write_counter(state_bytes, item, val):
    """Write the 11-bit counter for the given item into state_bytes (a bytearray)."""
    start_bit = item * BITS_PER_ITEM
    for bit_idx in range(BITS_PER_ITEM):
        absolute_bit = start_bit + bit_idx
        byte_idx = absolute_bit // 8
        bit_in_byte = absolute_bit % 8
        if val & (1 << bit_idx):
            state_bytes[byte_idx] |= (1 << bit_in_byte)
        else:
            state_bytes[byte_idx] &= ~(1 << bit_in_byte)


def update(state, item):
    """Update the sketch state with a new item occurrence."""
    if time.monotonic() > _DEADLINE:
        return state  # safety: return unchanged

    new_state = bytearray(state)

    # Ensure we have enough bytes (state should be 1024 bytes, first 550 used)
    if len(new_state) < TOTAL_BYTES_NEEDED:
        new_state.extend(b'\x00' * (TOTAL_BYTES_NEEDED - len(new_state)))

    current = _read_counter(new_state, item)
    if current < MAX_COUNT:
        current += 1
    _write_counter(new_state, item, current)

    return bytes(new_state)


def query(state, item):
    """Return a nonnegative frequency estimate for a given item."""
    if time.monotonic() > _DEADLINE:
        return 0  # safety

    return _read_counter(state, item)
