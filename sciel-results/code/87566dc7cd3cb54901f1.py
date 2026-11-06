"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    DEADLINE = 110.0

    moduli = list(range(20, 201))

    def is_disjoint(m, r, chosen):
        for mp, rp in chosen:
            g = gcd(m, mp)
            if (r - rp) % g == 0:
                return False
        return True

    chosen = []
    for m in moduli:
        if time.time() > DEADLINE:
            break
        for r in range(m):
            if is_disjoint(m, r, chosen):
                chosen.append([m, r])
                break

    return chosen
# EVOLVE-BLOCK-END
