"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
# Precompute modular inverses of 1..10 mod 101
_INV = [0] + [pow(x, -1, 101) for x in range(1, 11)]

# Precompute all subsets of {1,...,10}: map residue -> best (size, terms_tuple)
# There are 2^10 - 1 = 1023 non-empty subsets.
# For each residue, keep only the minimum-size subset (ties broken arbitrarily).
_BEST_FOR_RESIDUE = [None] * 101
for mask in range(1, 1 << 10):
    terms = []
    r = 0
    for x in range(1, 11):
        if (mask >> (x - 1)) & 1:
            terms.append(x)
            r += _INV[x]
    r %= 101
    size = len(terms)
    if _BEST_FOR_RESIDUE[r] is None or size < _BEST_FOR_RESIDUE[r][0]:
        _BEST_FOR_RESIDUE[r] = (size, tuple(terms))


def construct():
    result = [list(_BEST_FOR_RESIDUE[r][1]) for r in range(101)]
    
    # Verify
    for r in range(101):
        s = sum(_INV[x] for x in result[r]) % 101
        assert s == r, f"Residue {r}: got sum {s}"
        assert len(set(result[r])) == len(result[r]), f"Row {r} has duplicate denominators"
        assert all(1 <= x <= 10 for x in result[r]), f"Row {r} has out-of-range denominator"
    
    return result

# EVOLVE-BLOCK-END
