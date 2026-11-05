"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

_START = time.time()
_DEADLINE = 85.0


def _elapsed():
    return time.time() - _START


def construct():
    MOD = 101
    N = 10

    # Precompute modular inverses
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, -1, MOD)

    # Enumerate all non-empty subsets, compute (size, residue, mask)
    mask_info = []
    for mask in range(1, 1 << N):
        size = 0
        s = 0
        m = mask
        while m:
            lsb = m & (-m)
            x = lsb.bit_length()
            s += inv[x]
            size += 1
            m -= lsb
        residue = s % MOD
        mask_info.append((size, residue, mask))

    # Sort by size so we process smallest subsets first
    mask_info.sort(key=lambda t: t[0])

    # For each residue, record the minimum-size subset that achieves it
    best_for_residue = {}  # residue -> (size, mask)
    for size, residue, mask in mask_info:
        if residue not in best_for_residue:
            best_for_residue[residue] = (size, mask)

    # Check if all residues are covered
    all_covered = all(r in best_for_residue for r in range(MOD))

    if not all_covered:
        # Fallback: use all subsets
        for mask in range(1, 1 << N):
            size = bin(mask).count('1')
            s = 0
            m = mask
            while m:
                lsb = m & (-m)
                x = lsb.bit_length()
                s += inv[x]
                m -= lsb
            residue = s % MOD
            if residue not in best_for_residue:
                best_for_residue[residue] = (size, mask)

    # Build rows
    rows = [None] * MOD
    for r in range(MOD):
        if r in best_for_residue:
            size, mask = best_for_residue[r]
            terms = [x for x in range(1, N + 1) if (mask >> (x - 1)) & 1]
            rows[r] = terms
        else:
            # Should not happen, but fallback
            rows[r] = list(range(1, N + 1))

    return rows

# EVOLVE-BLOCK-END
