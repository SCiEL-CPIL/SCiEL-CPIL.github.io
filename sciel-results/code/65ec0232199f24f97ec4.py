import time

_START = time.time()
_DEADLINE = 89.0  # Safety margin before 90s limit

MOD = 101
N = 10
_FULL = 1 << N  # 1024


def construct():
    elapsed = time.time() - _START
    if elapsed > _DEADLINE:
        # Emergency fallback
        inv = [0] * (N + 1)
        for x in range(1, N + 1):
            inv[x] = pow(x, -1, MOD)
        first = {}
        for mask in range(1, _FULL):
            s = 0
            for i in range(N):
                if mask & (1 << i):
                    s = (s + inv[i + 1]) % MOD
            if s not in first:
                first[s] = mask
        rows = []
        for r in range(MOD):
            m = first.get(r, 1)
            denoms = [i + 1 for i in range(N) if m & (1 << i)]
            rows.append(denoms)
        return rows

    # 1. Compute modular inverses
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, -1, MOD)

    # 2. Precompute sum and popcount for all 1024 masks
    sums = [0] * _FULL
    pop = [0] * _FULL
    for mask in range(1, _FULL):
        lsb = mask & (-mask)
        idx = lsb.bit_length()  # 1-based index
        prev = mask ^ lsb
        sums[mask] = (sums[prev] + inv[idx]) % MOD
        pop[mask] = pop[prev] + 1

    # 3. For each residue, find the minimum popcount mask that achieves it
    min_pop_for_residue = [100] * MOD
    best_mask_for_residue = [0] * MOD

    for mask in range(1, _FULL):
        r = sums[mask]
        p = pop[mask]
        if p < min_pop_for_residue[r]:
            min_pop_for_residue[r] = p
            best_mask_for_residue[r] = mask

    # 4. Determine the minimum k such that all residues have a mask of size <= k
    current_max = max(min_pop_for_residue)

    # 5. Construct rows from the optimal assignment
    # Pre-compute denominator lists for all masks to avoid repeated bit-checking
    mask_to_denoms = [None] * _FULL
    for mask in range(_FULL):
        denoms = []
        for i in range(N):
            if mask & (1 << i):
                denoms.append(i + 1)
        mask_to_denoms[mask] = denoms

    rows = []
    for r in range(MOD):
        mask = best_mask_for_residue[r]
        rows.append(mask_to_denoms[mask])

    # Verify all rows are nonempty
    for r in range(MOD):
        if not rows[r]:
            for mask in range(1, _FULL):
                if sums[mask] == r:
                    denoms = [i + 1 for i in range(N) if mask & (1 << i)]
                    rows[r] = denoms
                    break

    return rows
