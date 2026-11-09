"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _greedy_construct(n=12, deadline=None):
    # Start with the smallest positive integer.
    seq = [1]
    # Maintain the set of all subset sums of the current sequence.
    sums = {0, 1}

    while len(seq) < n:
        if deadline is not None and time.time() > deadline:
            break
        # Try the smallest candidate not already used.
        c = 1
        while True:
            if c in seq:
                c += 1
                continue
            # Collision check: for every existing subset sum s, s + c must be new.
            ok = True
            for s in sums:
                if s + c in sums:
                    ok = False
                    break
            if ok:
                # Accept c; update subset sums.
                seq.append(c)
                new_sums = set()
                for s in sums:
                    new_sums.add(s + c)
                sums |= new_sums
                break
            c += 1
    return seq

def construct():
    # Safety margin: stop searching ~3s before the 90s wall-clock kill.
    start = time.time()
    deadline = start + 87.0
    result = _greedy_construct(12, deadline)

    # If greedy didn't finish in time, fall back to a valid (but larger) binary set.
    if len(result) < 12:
        result = [1 << i for i in range(12)]

    return result
# EVOLVE-BLOCK-END
