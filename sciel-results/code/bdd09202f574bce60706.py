"""Global greedy set-cover with bitset acceleration + local polish, deadline-guarded."""
import time

_BASE = 51975
_DEADLINE = 150.0
_SAFETY = 10.0  # seconds to leave before the evaluator kills us


def _divisors(n):
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            if i > 1:
                divs.append(i)
            j = n // i
            if j != i and j > 1:
                divs.append(j)
        i += 1
    return divs


def _precompute_masks(divs):
    # masks[(m, r)] = int bitset of residues x in [0, base) with x % m == r
    # cnt[(m, r)]  = base // m (constant popcount of the mask)
    masks = {}
    for m in divs:
        step = _BASE // m
        for r in range(m):
            # build bitset for residue r mod m
            # positions r, r+m, r+2m, ...
            bits = 0
            pos = r
            while pos < _BASE:
                bits |= (1 << pos)
                pos += m
            masks[(m, r)] = bits
    return masks


def construct():
    start = time.time()
    deadline = start + _DEADLINE - _SAFETY

    divs = _divisors(_BASE)
    # Sort by coverage count ascending (smaller moduli first) as a tie-breaker preference,
    # but greedy will select by marginal gain each step.
    divs.sort(key=lambda m: _BASE // m)

    masks = _precompute_masks(divs)

    all_ones = (1 << _BASE) - 1
    covered = 0  # bitset of covered residues
    chosen = []  # list of (m, r)

    # Precompute list of candidate (m, r) with their base masks
    # We'll evaluate marginal gain = bit_count(mask & ~covered)
    candidates = [(m, r, masks[(m, r)]) for m in divs for r in range(m)]

    # Greedy set cover
    while True:
        if time.time() > deadline:
            break
        if covered == all_ones:
            break
        uncovered = all_ones ^ covered
        best_gain = 0
        best_key = None
        # Iterate candidates; pick max marginal gain, tie-break by smaller m
        for m, r, mask in candidates:
            gain = (mask & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_key = (m, r)
                if gain == _BASE // m and m == min(c[0] for c in candidates if c[1] == r):
                    pass
        if best_gain == 0:
            break
        m, r = best_key
        chosen.append((m, r))
        covered |= masks[(m, r)]

    # Local polish: try swapping each chosen class with a better one
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            uncovered = all_ones ^ covered
            for i in range(len(chosen)):
                if time.time() > deadline:
                    break
                m_i, r_i = chosen[i]
                # Remove class i
                without_i = covered ^ masks[(m_i, r_i)]  # since it was fully covered
                # Actually covered may have overlaps; removing is not XOR.
                # Recompute: covered_without = OR of all chosen except i
                # This is expensive; approximate by checking if any other class covers those residues.
                # For simplicity, do a proper recompute only for promising swaps.
                # Try each candidate replacement
                best_gain = (masks[(m_i, r_i)] & uncovered).bit_count()  # current contribution
                for m, r, mask in candidates:
                    if (m, r) == (m_i, r_i):
                        continue
                    # gain of adding (m,r) on top of covered_without_i
                    # covered_without_i = covered minus mask_i's exclusive parts
                    # Approximation: new_gain = bit_count(mask & (all_ones ^ covered)) + overlap with mask_i
                    # This is an underestimate/overestimate; do exact for top candidates only.
                    new_gain = (mask & (all_ones ^ covered)).bit_count()
                    if new_gain > best_gain + 1:  # need strictly better
                        # Exact check: remove i, add (m,r)
                        new_covered = 0
                        for j, (mj, rj) in enumerate(chosen):
                            if j != i:
                                new_covered |= masks[(mj, rj)]
                        new_covered |= mask
                        if new_covered.bit_count() > covered.bit_count():
                            chosen[i] = (m, r)
                            covered = new_covered
                            improved = True
                            best_gain = (masks[(m, r)] & (all_ones ^ covered)).bit_count()
                            break

    return [[m, r] for m, r in chosen]
