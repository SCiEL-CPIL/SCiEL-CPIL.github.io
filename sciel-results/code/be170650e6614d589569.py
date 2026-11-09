"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import lcm
    import time

    deadline = time.time() + 60.0
    LIMIT = 256

    # Numbers in [2,256] that divide 256; at most one may be selected.
    divisors = [x for x in range(2, LIMIT + 1) if LIMIT % x == 0]
    others = [x for x in range(2, LIMIT + 1) if LIMIT % x != 0]

    # Optimal: take every non-divisor (mutually compatible) plus the single
    # divisor with the largest reciprocal (smallest value).
    if divisors:
        best_divisor = min(divisors)
        selected = others + [best_divisor]
    else:
        selected = list(others)

    # Safety: verify the pairwise lcm constraint; fall back to a greedy
    # construction if (unexpectedly) violated, keeping a time guard.
    def valid(sel):
        for i in range(len(sel)):
            for j in range(i + 1, len(sel)):
                if lcm(sel[i], sel[j]) <= LIMIT:
                    return False
        return True

    if not valid(selected):
        sel = []
        for x in range(2, LIMIT + 1):
            if time.time() > deadline:
                break
            if all(lcm(x, y) > LIMIT for y in sel):
                sel.append(x)
        selected = sel

    return selected
# EVOLVE-BLOCK-END
