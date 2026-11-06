"""Combined program: fast vectorized greedy + local improvement for odd covering systems."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 115.0  # Safety margin before 150s kill
    base = 51975  # 3^3 * 5^2 * 7 * 11, all odd

    # Get all odd divisors > 1 of base, sorted ascending (smallest covers most residues)
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    divisors.sort()

    # Greedy construction: for each m in increasing order, pick r maximizing newly covered
    covered = np.zeros(base, dtype=np.uint8)
    classes = []
    used_mods = set()

    for m in divisors:
        if time.time() > deadline - 20.0:
            break
        # For each residue r in [0, m), count how many positions r, r+m, r+2m, ... are uncovered
        # Vectorized: covered[r::m] gives the slice; count zeros
        best_r = 0
        best_gain = -1
        # Compute gains for all r at once using reshape when possible
        # For speed, loop over r but use vectorized slice
        for r in range(m):
            gain = int(covered[r::m].sum())  # sum of 0/1 = covered count; we want uncovered
            # Actually we want uncovered count = len - covered_count
            # len of slice r::m is (base - r) // m + 1 if r < base else 0
            # Since m divides base and r < m, len = base // m exactly
            uncovered_gain = (base // m) - gain
            if uncovered_gain > best_gain:
                best_gain = uncovered_gain
                best_r = r

        if best_gain > 0:
            classes.append([m, best_r])
            covered[best_r::m] = 1
            used_mods.add(m)
        # If best_gain == 0, this modulus adds nothing; skip it

    current_uncovered = int(covered.sum())
    best_uncovered = current_uncovered
    best_classes = [c[:] for c in classes]

    # Local improvement: try changing residues of existing classes
    # For each class i, temporarily consider alternative residues
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        for i in range(len(classes)):
            if time.time() > deadline - 10.0:
                break
            m, r = classes[i]
            # For large m, sampling is faster; for small m, try all
            if m <= 50:
                candidates = list(range(m))
            else:
                # Sample a subset of residues for large moduli
                candidates = list(range(0, m, max(1, m // 50)))

            best_r_new = r
            best_gain_new = -1
            # Remove current class to see what it uniquely covers
            # Save current covered state
            orig_covered = covered.copy()
            # Remove class i
            covered[r::m] = 0
            # For each candidate r', compute how many new positions it covers
            for r_new in candidates:
                if r_new == r:
                    continue
                # Gain from adding (m, r_new) on top of covered without class i
                gain = int((covered[r_new::m] == 0).sum())
                if gain > best_gain_new:
                    best_gain_new = gain
                    best_r_new = r_new

            # Restore
            covered = orig_covered
            if best_r_new != r and best_gain_new > 0:
                # Check if this actually improves total coverage
                # Compute new uncovered after changing class i to (m, best_r_new)
                new_covered = covered.copy()
                new_covered[r::m] = 0
                new_covered[best_r_new::m] = 1
                new_uncovered = int(new_covered.sum())
                if new_uncovered < current_uncovered:
                    classes[i] = [m, best_r_new]
                    covered = new_covered
                    current_uncovered = new_uncovered
                    if current_uncovered < best_uncovered:
                        best_uncovered = current_uncovered
                        best_classes = [c[:] for c in classes]
                    improved = True

    # Final pass: try adding unused small moduli that might fill gaps
    if time.time() < deadline - 5.0:
        for m in divisors:
            if time.time() > deadline - 5.0:
                break
            if m in used_mods:
                continue
            # Only consider if it could add coverage
            if base // m < 1:
                continue
            best_r = 0
            best_gain = -1
            for r in range(min(m, 50)):  # Sample for large m
                gain = int((covered[r::m] == 0).sum())
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            if best_gain > 0:
                classes.append([m, best_r])
                covered[best_r::m] = 1
                used_mods.add(m)
                current_uncovered = int(covered.sum())
                if current_uncovered < best_uncovered:
                    best_uncovered = current_uncovered
                    best_classes = [c[:] for c in classes]

    return best_classes
# EVOLVE-BLOCK-END
