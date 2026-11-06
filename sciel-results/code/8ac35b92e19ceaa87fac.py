import time
import numpy as np

BASE = 51975  # 3^3 * 5^2 * 7 * 11

_DIVISORS = None

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

def _coverage_from_classes(classes):
    """Return boolean coverage array for list of [m, r] classes."""
    covered = np.zeros(BASE, dtype=bool)
    for m, r in classes:
        covered[r::m] = True
    return covered

def _best_residue_for_modulus(covered, m):
    """
    Given current covered array, find residue r for modulus m that covers the most
    currently-uncovered residues.
    Returns (best_r, best_gain).
    """
    # We want to maximize number of uncovered positions in residue class r mod m.
    # uncovered = ~covered
    # For each r, count uncovered indices i with i % m == r.
    # Vectorized via bincount over positions that are uncovered.
    uncovered = np.where(~covered)[0]
    if len(uncovered) == 0:
        return 0, 0
    counts = np.bincount(uncovered % m, minlength=m)
    best_r = int(np.argmax(counts))
    best_gain = int(counts[best_r])
    return best_r, best_gain

def _local_search(classes, covered, deadline):
    """
    Local search: for each class, try changing its residue to maximize coverage
    given all other classes. We avoid O(n^2) full recomputation by temporarily
    removing the class's contribution and finding best residue for its modulus.
    """
    n = len(classes)
    if n == 0:
        return classes, covered

    current_score = int(np.sum(~covered))
    best_classes = classes[:]
    best_covered = covered.copy()
    best_score = current_score

    # Precompute a list of (m, r)
    moduli = [m for m, r in classes]
    residues = [r for m, r in classes]

    # We'll do several passes; each pass tries to improve one class at a time.
    # To keep it efficient, we recompute coverage without class i only when needed.
    # But that's O(n * coverage_cost). n is at most number of divisors (~120).
    # Coverage cost is O(BASE) per class addition, so O(n*BASE) per full pass.
    # That might be too slow if we do many passes. We'll limit passes and use time guard.

    max_passes = 8
    for pass_num in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            m_i = moduli[i]
            r_i = residues[i]

            # Build coverage from all classes except i
            # This is expensive, but n is small. We can optimize by precomputing
            # coverage for each class and using XOR? No, coverage is OR, not XOR.
            # So we must recompute.
            cov_other = np.zeros(BASE, dtype=bool)
            for j in range(n):
                if j == i:
                    continue
                mj, rj = moduli[j], residues[j]
                cov_other[rj::mj] = True

            # Find best residue for modulus m_i given cov_other
            best_r, best_gain = _best_residue_for_modulus(cov_other, m_i)

            if best_r != r_i:
                # Check if this actually improves total coverage
                # New coverage = cov_other OR (best_r mod m_i)
                new_covered = cov_other.copy()
                new_covered[best_r::m_i] = True
                new_score = int(np.sum(~new_covered))

                if new_score < current_score:
                    # Accept improvement
                    residues[i] = best_r
                    covered = new_covered
                    current_score = new_score
                    improved = True

                    if current_score < best_score:
                        best_score = current_score
                        best_classes = [[moduli[j], residues[j]] for j in range(n)]
                        best_covered = covered.copy()

        if not improved:
            break

    return best_classes, best_covered

def _try_add_unused_moduli(classes, covered, deadline):
    """
    Try adding unused moduli if they improve coverage.
    """
    used_moduli = set(m for m, r in classes)
    all_divisors = _get_divisors()
    current_score = int(np.sum(~covered))
    best_classes = classes[:]
    best_covered = covered.copy()
    best_score = current_score

    for m in all_divisors:
        if time.time() > deadline:
            break
        if m in used_moduli:
            continue

        best_r, best_gain = _best_residue_for_modulus(covered, m)
        if best_gain > 0:
            new_covered = covered.copy()
            new_covered[best_r::m] = True
            new_score = int(np.sum(~new_covered))

            if new_score < current_score:
                classes.append([m, best_r])
                covered = new_covered
                current_score = new_score

                if current_score < best_score:
                    best_score = current_score
                    best_classes = classes[:]
                    best_covered = covered.copy()

    return best_classes, best_covered

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    divisors = _get_divisors()

    # Global greedy set cover
    covered = np.zeros(BASE, dtype=bool)
    classes = []
    remaining_moduli = set(divisors)

    # Sort moduli for deterministic tie-breaking? We'll just iterate.
    # To speed up, we can process smaller moduli first? They cover more residues.
    # But greedy should pick best gain regardless of order.
    # We'll iterate in sorted order for determinism.
    sorted_moduli = sorted(remaining_moduli)

    while remaining_moduli:
        if time.time() > deadline:
            break

        uncovered = np.where(~covered)[0]
        if len(uncovered) == 0:
            break

        best_m = None
        best_r = None
        max_cov = -1

        # For each remaining modulus, find the residue that covers most uncovered
        # We use bincount on uncovered indices modulo m.
        # This is O(len(uncovered)) per modulus, which is O(BASE) worst-case.
        # With ~120 moduli, this is ~120 * 52000 = 6.2M operations per greedy step.
        # Number of greedy steps is at most number of moduli (~120).
        # Total: ~750M operations, which might be too slow in Python.
        # We need to optimize.

        # Optimization: Instead of iterating all remaining moduli each step,
        # we can precompute for each modulus the best residue given current uncovered.
        # But uncovered changes each step.

        # Alternative: Use the fact that we only need the global best.
        # We can early-stop if we find a modulus that covers all uncovered?
        # But that's unlikely.

        # Let's try to reduce the number of moduli we consider.
        # Smaller moduli have more residue classes, but each class is smaller.
        # Actually, for a given uncovered set, the gain for modulus m is
        # max_r count of uncovered in residue r mod m.
        # This is at most ceil(len(uncovered)/m) * something? Not exactly.

        # We'll just do it and hope it's fast enough. If not, we'll need to optimize.
        # For now, let's try.

        for m in sorted_moduli:
            if m not in remaining_moduli:
                continue
            # Counts of uncovered residues modulo m
            counts = np.bincount(uncovered % m, minlength=m)
            current_max = int(np.max(counts))
            if current_max > max_cov:
                max_cov = current_max
                best_m = m
                best_r = int(np.argmax(counts))

        if max_cov <= 0:
            break

        # Add the class
        classes.append([best_m, best_r])
        remaining_moduli.remove(best_m)

        # Update covered
        covered[best_r::best_m] = True

    # Local search refinement
    classes, covered = _local_search(classes, covered, deadline)

    # Try adding unused moduli
    classes, covered = _try_add_unused_moduli(classes, covered, deadline)

    # Another local search pass after adding moduli
    if time.time() < deadline:
        classes, covered = _local_search(classes, covered, deadline)

    return classes
