"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 128}

def construct():
    n = P['n']
    # S_small = {1, ..., 32}
    # S_large = {64, ..., 128}
    # Max pair sum in S_small is 32+31=63 < 64.
    # Min pair sum in S_large is 64+65=129 > 128.
    # Cross sums x+L (x in S_small, L in S_large) are > 63 and < 128+32=160.
    # If x+L in S, it must be in S_large (since >63).
    # Forbidden triple requires x+y in S, x+L in S, y+L in S.
    # x+y < 64, so x+y not in S_large. x+y not in S_small. So x+y not in S.
    # Thus no forbidden triples.
    small_part = list(range(1, 33))
    large_part = list(range(64, n + 1))
    return small_part + large_part
# EVOLVE-BLOCK-END
