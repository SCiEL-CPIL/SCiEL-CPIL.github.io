"""Improved greedy set-cover for odd covering systems with moduli dividing 51975."""
import time
import math

_BASE = 51975
# divisors of 51975 = 3^3 * 5^2 * 7 * 11, m > 1, odd
_DIVS = []
for a in range(4):
    for b in range(3):
        for c in range(2):
            for d in range(2):
                m = (3 ** a) * (5 ** b) * (7 ** c) * (11 ** d)
                if m > 1:
                    _DIVS.append(m)
_DIVS.sort()

# Precompute, for each divisor m, the set of residue classes r in [0, m)
# as a list of indices. We'll work with a coverage bitset (bytearray of length _BASE).


def _precompute_classes():
    """For each divisor m, list the residue positions (indices) for each r."""
    data = {}
    for m in _DIVS:
        classes = []
        for r in range(m):
            # positions r, r+m, r+2m, ... < _BASE
            idx = list(range(r, _BASE, m))
            classes.append(idx)
        data[m] = classes
    return data


def _count_uncovered(covered, positions):
    c = 0
    for i in positions:
        if covered[i] == 0:
            c += 1
    return c


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill

    covered = bytearray(_BASE)  # 0 = uncovered
    classes = []
    used_moduli = set()

    # Precompute class index lists
    class_data = _precompute_classes()

    # Cost weight per modulus: larger moduli cover fewer residues per class,
    # so weight = 1 (fixed) but we consider ratio uncovered/size as impact.
    # We'll pick the class (m, r) maximizing newly-covered / cost(m).
    # cost(m) = 1 for all (set cover), but to prioritize small m early (which
    # cover more residues), we can use cost = log(m) or just count.
    # Actually for set cover, standard greedy maximizes newly-covered count.
    # But to be smarter, we can weight by 1/m so that smaller moduli (which
    # cover more residues) are favored. Let's use cost = 1.0 (pure count)
    # but process in a way that naturally favors small m.

    # Better: use ratio = newly_covered / (1.0 + 0.001 * m) to slightly
    # penalize large moduli, encouraging small-m selection first.

    # Sort moduli: we'll iterate, but pick globally best each step.
    # To keep it efficient, we evaluate all (m, r) pairs each step.
    # Number of pairs = sum of m over divisors = manageable.

    total_pairs = sum(len(class_data[m]) for m in _DIVS)

    while time.time() < deadline:
        best_gain = 0
        best_m = None
        best_r = None
        best_cost = 0.0

        for m in _DIVS:
            if m in used_moduli:
                continue
            cls_list = class_data[m]
            # cost: prefer smaller m (they cover more residues per class)
            cost = 1.0 / math.log(m + 1)  # higher cost for small m -> we want max gain/cost
            # Actually we want to maximize gain * priority. Let's use:
            # score = newly_covered * (1.0 / m)  -> favors small m
            # Or simply newly_covered, since small m already cover more.
            # Let's use newly_covered / m as a normalized impact.
            for r in range(len(cls_list)):
                positions = cls_list[r]
                # Quick skip: if already covered count is low, skip
                gain = 0
                for i in positions:
                    if covered[i] == 0:
                        gain += 1
                if gain == 0:
                    continue
                # Normalize by modulus size: smaller m gives higher ratio
                ratio = gain / m
                if ratio > best_gain or (ratio == best_gain and best_m is not None and m < best_m):
                    best_gain = ratio
                    best_m = m
                    best_r = r
                    best_cost = cost

        if best_m is None:
            break

        # Commit
        used_moduli.add(best_m)
        classes.append([best_m, best_r])
        positions = class_data[best_m][best_r]
        for i in positions:
            covered[i] = 1

        # Check if fully covered
        if all(covered):
            break

    # Return the classes
    return classes
