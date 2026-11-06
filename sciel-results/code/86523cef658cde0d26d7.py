"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

_P = 101
_N = 10

# Precompute modular inverses
_INV = {x: pow(x, -1, _P) for x in range(1, _N + 1)}

# Precompute all subsets: for each size, list of (residue, terms_tuple)
# subsets_by_size[s] = list of (r, terms)
_subsets_by_size = [[] for _ in range(_N + 1)]
for mask in range(1, 1 << _N):
    terms = tuple(x for x in range(1, _N + 1) if (mask >> (x - 1)) & 1)
    r = sum(_INV[x] for x in terms) % _P
    s = len(terms)
    _subsets_by_size[s].append((r, terms))

def construct():
    deadline = time.time() + 55  # safety margin

    # For each size s from 1 to 10, try to cover all 101 residues using only subsets of size s
    # We want the smallest s such that all residues can be covered.
    # For a given s, we need to assign to each residue r a subset of size s that sums to r.
    # This is a set cover / assignment problem. Since we just need existence (any assignment works),
    # we can check: for each residue r, is there at least one subset of size s with sum r?
    # If yes for all r, then s is feasible.
    
    # First, find the minimum feasible k
    for k in range(1, _N + 1):
        # Check if every residue is achievable with size k
        achievable = set()
        for r, terms in _subsets_by_size[k]:
            achievable.add(r)
        if len(achievable) == _P:
            # All residues are achievable with size k
            # Assign the first found subset for each residue
            result = [None] * _P
            for r, terms in _subsets_by_size[k]:
                if result[r] is None:
                    result[r] = list(terms)
            # Verify all are filled
            if all(x is not None for x in result):
                return result
    
    # Fallback: use the original approach (minimum size per residue independently)
    min_size_for_residue = {}
    terms_for_residue = {}
    for size in range(1, _N + 1):
        for r, terms in _subsets_by_size[size]:
            if r not in min_size_for_residue:
                min_size_for_residue[r] = size
                terms_for_residue[r] = terms
    
    result = []
    for r in range(_P):
        if r in terms_for_residue:
            result.append(list(terms_for_residue[r]))
        else:
            # Should not happen, but fallback
            result.append([1])
    return result
# EVOLVE-BLOCK-END
