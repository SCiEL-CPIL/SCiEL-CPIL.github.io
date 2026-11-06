import time

BASE = 51975  # 3^3 * 5^2 * 7 * 11

_DIVISORS = None
_MASKS = None  # dict: (m, r) -> bitmask int

def _get_divisors():
    global _DIVISORS
    if _DIVISORS is None:
        divs = set()
        # 51975 = 3^3 * 5^2 * 7 * 11
        for a in range(4):  # 3^0..3^3
            p3 = 3 ** a
            for b in range(3):  # 5^0..5^2
                p15 = p3 * (5 ** b)
                for c in range(2):  # 7^0..7^1
                    p105 = p15 * (7 ** c)
                    for d in range(2):  # 11^0..11^1
                        v = p105 * (11 ** d)
                        if v > 1:
                            divs.add(v)
        _DIVISORS = sorted(divs)
    return _DIVISORS

def _get_masks():
    """Precompute bitmask for each (m, r) pair where m divides BASE and m > 1."""
    global _MASKS
    if _MASKS is None:
        _MASKS = {}
        for m in _get_divisors():
            for r in range(m):
                # Build bitmask: bit i is 1 if i % m == r
                mask = 0
                for i in range(r, BASE, m):
                    mask |= (1 << i)
                _MASKS[(m, r)] = mask
    return _MASKS

def _popcount(x):
    """Count set bits in integer x."""
    return bin(x).count('1')

def _local_search(classes, uncovered_mask, deadline):
    """
    Local search: for each class, try changing its residue to maximize coverage
    given all other classes.
    """
    n = len(classes)
    if n == 0:
        return classes, uncovered_mask

    full_mask = (1 << BASE) - 1
    current_uncovered = uncovered_mask
    current_score = _popcount(current_uncovered)
    best_classes = classes[:]
    best_uncovered = current_uncovered
    best_score = current_score

    moduli = [m for m, r in classes]
    residues = [r for m, r in classes]

    max_passes = 10
    for pass_num in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            m_i = moduli[i]
            r_i = residues[i]

            # Compute uncovered mask without class i
            # coverage_i = masks[(m_i, r_i)]
            # other_coverage = full_mask - (uncovered | coverage_i) ... no
            # Actually: uncovered_with_i = current_uncovered
            # coverage_i = masks[(m_i, r_i)]
            # uncovered_without_i = current_uncovered | coverage_i
            # But that's not right either. Let me think...
            # 
            # current_uncovered = full_mask - (OR of all class masks)
            # Let C = OR of all class masks
            # current_uncovered = full_mask ^ C  (since C is subset of full_mask)
            # C_i = mask of class i
            # C_without_i = C with class i removed... but OR doesn't support removal
            # 
            # Alternative: uncovered_without_i = current_uncovered | C_i
            # Because any position covered only by class i would be uncovered without i,
            # and positions covered by other classes remain covered.
            # So: uncovered_without_i = current_uncovered | masks[(m_i, r_i)]
            
            cov_i = _get_masks()[(m_i, r_i)]
            uncovered_without_i = current_uncovered | cov_i

            # Find best residue for modulus m_i given uncovered_without_i
            best_r = -1
            best_gain = -1
            for r in range(m_i):
                cov_r = _get_masks()[(m_i, r)]
                # gain = number of positions in cov_r that are in uncovered_without_i
                gain = _popcount(uncovered_without_i & cov_r)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r

            if best_r != r_i and best_gain > 0:
                # Compute new uncovered if we change class i to (m_i, best_r)
                new_cov_i = _get_masks()[(m_i, best_r)]
                new_uncovered = uncovered_without_i & ~new_cov_i
                new_score = _popcount(new_uncovered)

                if new_score < current_score:
                    # Accept improvement
                    residues[i] = best_r
                    current_uncovered = new_uncovered
                    current_score = new_score
                    improved = True

                    if current_score < best_score:
                        best_score = current_score
                        best_classes = [[moduli[j], residues[j]] for j in range(n)]
                        best_uncovered = current_uncovered

        if not improved:
            break

    return best_classes, best_uncovered

def _try_add_unused_moduli(classes, uncovered_mask, deadline):
    """
    Try adding unused moduli if they improve coverage.
    """
    used_moduli = set(m for m, r in classes)
    all_divisors = _get_divisors()
    current_uncovered = uncovered_mask
    current_score = _popcount(current_uncovered)
    best_classes = classes[:]
    best_uncovered = current_uncovered
    best_score = current_score

    masks = _get_masks()

    for m in all_divisors:
        if time.time() > deadline:
            break
        if m in used_moduli:
            continue

        best_r = -1
        best_gain = -1
        for r in range(m):
            cov_r = masks[(m, r)]
            gain = _popcount(current_uncovered & cov_r)
            if gain > best_gain:
                best_gain = gain
                best_r = r

        if best_gain > 0:
            new_cov = masks[(m, best_r)]
            new_uncovered = current_uncovered & ~new_cov
            new_score = _popcount(new_uncovered)

            if new_score < current_score:
                classes.append([m, best_r])
                current_uncovered = new_uncovered
                current_score = new_score

                if current_score < best_score:
                    best_score = current_score
                    best_classes = classes[:]
                    best_uncovered = current_uncovered

    return best_classes, best_uncovered

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    divisors = _get_divisors()
    masks = _get_masks()

    # Global greedy set cover using bitmasks
    full_mask = (1 << BASE) - 1
    uncovered = full_mask
    classes = []
    remaining_moduli = set(divisors)

    while remaining_moduli:
        if time.time() > deadline:
            break

        if uncovered == 0:
            break

        best_m = None
        best_r = None
        max_gain = -1

        for m in sorted(remaining_moduli):
            for r in range(m):
                cov_r = masks[(m, r)]
                gain = _popcount(uncovered & cov_r)
                if gain > max_gain:
                    max_gain = gain
                    best_m = m
                    best_r = r

        if max_gain <= 0:
            break

        # Add the class
        classes.append([best_m, best_r])
        remaining_moduli.remove(best_m)

        # Update uncovered
        uncovered &= ~masks[(best_m, best_r)]

    # Local search refinement
    classes, uncovered = _local_search(classes, uncovered, deadline)

    # Try adding unused moduli
    classes, uncovered = _try_add_unused_moduli(classes, uncovered, deadline)

    # Another local search pass after adding moduli
    if time.time() < deadline:
        classes, uncovered = _local_search(classes, uncovered, deadline)

    return classes
