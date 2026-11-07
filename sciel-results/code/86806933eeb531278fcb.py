"""Domain-adaptive frequency sketch: hot exact counters + widened CM sketch."""
import struct

# EVOLVE-BLOCK-START
def update(state, item):
    state = bytearray(state)
    if item < 20:
        # Hot counter: 20 items, 1 byte each at offset 0..19
        state[item] = min(255, state[item] + 1)
    else:
        # CM sketch: 16 rows x 64 cols of uint8 at offset 20..1043
        # 16 * 64 = 1024 bytes, but we only have 1024 - 20 = 1004 bytes available
        # So we use 15 rows x 64 = 960 bytes, leaving 44 bytes unused or for padding
        # Actually: 1024 - 20 = 1004. 15*64 = 960. We can fit 15 rows.
        # Let's use 15 rows to stay within budget.
        for row in range(15):
            idx = ((item * 1035493289 + 31 * row) % 2147483647) % 64
            offset = 20 + row * 64 + idx
            if offset < 1024:
                state[offset] = min(255, state[offset] + 1)
    return bytes(state)

def query(state, item):
    if item < 20:
        return float(state[item])
    else:
        estimates = []
        for row in range(15):
            idx = ((item * 1035493289 + 31 * row) % 2147483647) % 64
            offset = 20 + row * 64 + idx
            if offset < 1024:
                estimates.append(float(state[offset]))
        if not estimates:
            return 0.0
        return min(estimates)
# EVOLVE-BLOCK-END
