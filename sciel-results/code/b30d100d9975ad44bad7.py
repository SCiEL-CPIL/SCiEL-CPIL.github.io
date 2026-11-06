"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

_INV = {x: pow(x, -1, 101) for x in range(1, 11)}

# Precompute all subsets: mask -> (residue, size, terms_tuple)
_SUBSETS = []  # list of (size, residue, terms_tuple)
for mask in range(1, 1 << 10):
    terms = tuple(x for x in range(1, 11) if (mask >> (x - 1)) & 1)
    r = 0
    for x in terms:
        r += _INV[x]
    r %= 101
    _SUBSETS.append((len(terms), r, terms))

def construct():
    deadline = time.time() + 55  # safety margin
    
    # Precompute: for each residue, what's the minimum size subset that achieves it
    min_size_for_residue = {}
    terms_for_residue = {}
    # Process subsets in order of increasing size
    for size, r, terms in sorted(_SUBSETS, key=lambda x: x[0]):
        if r not in min_size_for_residue:
            min_size_for_residue[r] = size
            terms_for_residue[r] = terms
    
    # Find minimum k such that all 101 residues are covered
    # We need all residues to have a representation of size <= k
    if len(min_size_for_residue) < 101:
        # Some residues are not achievable at all (shouldn't happen with 1023 subsets)
        # Fallback
        representations = {}
        for size, r, terms in _SUBSETS:
            if r not in representations:
                representations[r] = list(terms)
        return [representations[r] for r in range(101)]
    
    # The minimum k is the maximum of min_size_for_residue over all residues
    k_min = max(min_size_for_residue.values())
    
    # Build result using the minimum-size representation for each residue
    result = []
    for r in range(101):
        result.append(list(terms_for_residue[r]))
    
    return result
# EVOLVE-BLOCK-END
