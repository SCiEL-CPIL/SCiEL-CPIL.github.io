import time

_start_time = time.time()
_DEADLINE = 300.0

def priority(vector, n):
    # Deadline guard: bail out if we're too close to the limit
    if time.time() - _start_time > _DEADLINE - 10:
        return 0.0
    # Morton code: interleave bits of coordinates to create a space-filling curve order
    # This spreads out the selection, avoiding the dense line structures of lexicographic order
    result = 0
    for i in range(n):
        val = vector[i]
        bit0 = val & 1
        bit1 = (val >> 1) & 1
        result |= (bit0 << (2*i)) | (bit1 << (2*i + 1))
    return result
