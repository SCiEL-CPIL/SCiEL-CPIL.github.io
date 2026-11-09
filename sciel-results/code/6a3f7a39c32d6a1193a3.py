"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    moduli = list(range(20, 201))
    n_mod = len(moduli)
    mod_idx = {m: i for i, m in enumerate(moduli)}

    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i, n_mod):
            g = gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def compute_score(classes):
        total = Fraction(0, 1)
        for (m, r) in classes:
            total += Fraction(1, m)
        return total

    def is_valid_residue(m, r, classes):
        for (m2, r2) in classes:
            g = gcd(m, m2)
            if (r - r2) % g == 0:
                return False
        return True

    def find_valid_residues(m, classes):
        valid = []
        for r in range(m):
            if is_valid_residue(m, r, classes):
                valid.append(r)
        return valid

    def flexibility_score(m, r, classes, used_moduli):
        """Count how many remaining moduli would be blocked by this residue choice."""
        score = 0
        for m2 in moduli:
            if m2 in used_moduli or m2 == m:
                continue
            g = gcd(m, m2)
            if g == 1:
                continue
            # If we pick residue r for m, it blocks r mod g for m2
            # Count how many residues of m2 are blocked
            score += g  # g residues of m2 are blocked
        return score

    def greedy_construct(order, used_moduli=None):
        if used_moduli is None:
            used_moduli = set()
        classes = []
        for m in order:
            if m in used_moduli:
                continue
            valid_residues = find_valid_residues(m, classes)
            if not valid_residues:
                continue
            # Pick residue that minimizes future conflicts
            best_r = min(valid_residues, key=lambda r: flexibility_score(m, r, classes, used_moduli))
            classes.append([m, best_r])
            used_moduli.add(m)
        return classes

    def local_search(classes, max_time=10.0):
        classes = [list(c) for c in classes]
        current_score = compute_score(classes)
        used_moduli = set(m for m, r in classes)
        start = time.time()

        improved = True
        while improved and (time.time() - start) < max_time:
            improved = False

            # Try adding any missing modulus
            for m in sorted([m for m in moduli if m not in used_moduli]):
                valid_residues = find_valid_residues(m, classes)
                if valid_residues:
                    best_r = min(valid_residues, key=lambda r: flexibility_score(m, r, classes, used_moduli))
                    new_score = current_score + Fraction(1, m)
                    if new_score > current_score:
                        classes.append([m, best_r])
                        current_score = new_score
                        used_moduli.add(m)
                        improved = True

            # Try 1-swap: remove one, add another
            for i in range(len(classes)):
                m_remove = classes[i][0]
                remaining = [c for j, c in enumerate(classes) if j != i]
                remaining_score = current_score - Fraction(1, m_remove)

                for m_add in sorted([m for m in moduli if m != m_remove and m not in used_moduli]):
                    valid_residues = find_valid_residues(m_add, remaining)
                    if valid_residues:
                        best_r = min(valid_residues, key=lambda r: flexibility_score(m_add, r, remaining, used_moduli - {m_remove}))
                        new_score = remaining_score + Fraction(1, m_add)
                        if new_score > current_score:
                            # Replace
                            classes = remaining + [[m_add, best_r]]
                            current_score = new_score
                            used_moduli.discard(m_remove)
                            used_moduli.add(m_add)
                            improved = True
                            break

            # Try changing residue of an existing class
            for i in range(len(classes)):
                m_i = classes[i][0]
                others = [c for j, c in enumerate(classes) if j != i]
                valid_residues = find_valid_residues(m_i, others)
                if len(valid_residues) > 1:
                    best_r = min(valid_residues, key=lambda r: flexibility_score(m_i, r, others, used_moduli))
                    if best_r != classes[i][1]:
                        classes[i][1] = best_r
                        improved = True

        return classes, current_score

    # Initial greedy: always process in ascending order (smallest moduli first)
    best_classes = greedy_construct(sorted(moduli))
    best_score = compute_score(best_classes)
    best_classes, best_score = local_search(best_classes, max_time=15.0)

    # Random restarts with different orderings
    random.seed(42)
    restart = 0
    while time.time() < deadline - 10:
        if restart % 3 == 0:
            # Ascending with small random perturbation
            order = sorted(moduli, key=lambda m: m + random.random() * 5)
        elif restart % 3 == 1:
            # Mostly ascending but with some shuffling among close values
            order = moduli.copy()
            random.shuffle(order)
            order.sort(key=lambda m: m / 20 + random.random() * 10)
        else:
            # Random
            order = moduli.copy()
            random.shuffle(order)
        
        classes = greedy_construct(order)
        classes, sc = local_search(classes, max_time=5.0)
        if sc > best_score:
            best_score = sc
            best_classes = [list(c) for c in classes]
        restart += 1

    # Final polish
    if time.time() < deadline - 5:
        remaining_time = deadline - 5 - time.time()
        best_classes, best_score = local_search(best_classes, max_time=min(remaining_time, 20.0))

    return best_classes
# EVOLVE-BLOCK-END
