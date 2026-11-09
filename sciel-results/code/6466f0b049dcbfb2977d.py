import time

_P = 1009
_M = 5

def _precompute():
    roots = {}
    for x in range(1, _P):
        r = (x * x) % _P
        if r != 0 and r not in roots:
            roots[r] = x
    return roots

# Precompute at module load
_ROOTS = _precompute()
_ROOT_SET = set(_ROOTS.keys())

def construct():
    start_time = time.monotonic()
    deadline = start_time + 55.0

    # Use bitset for efficient scanning
    # bit i is set if i is a non-zero QR mod _P
    bitset = 0
    for r in _ROOT_SET:
        bitset |= (1 << r)

    # We need to find smallest s in [1, 1004] such that
    # bits s, s+1, s+2, s+3, s+4 are all set.
    # This is equivalent to: (bitset >> s) & 0b11111 == 0b11111

    max_s = _P - _M + 1  # 1005, so s in 1..1004

    for s in range(1, max_s):
        if time.monotonic() > deadline:
            raise RuntimeError("Deadline exceeded")

        # Check if 5 consecutive bits starting at position s are all set
        if (bitset >> s) & 0b11111 == 0b11111:
            return {
                'start': s,
                'roots': [
                    _ROOTS[s],
                    _ROOTS[s + 1],
                    _ROOTS[s + 2],
                    _ROOTS[s + 3],
                    _ROOTS[s + 4]
                ]
            }

    raise RuntimeError("No run found")
