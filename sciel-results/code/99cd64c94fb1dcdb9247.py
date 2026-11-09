import time
import math
import sys
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # All candidate moduli 20..200, sorted ascending (prefer smaller m for higher 1/m)
    candidates = list(range(20, 201))
    n_cands = len(candidates)

    def time_up():
        return time.time() > deadline

    def is_compatible(placed, m, r):
        """Check if (m, r) is compatible with all placed classes."""
        for m2, r2 in placed:
            g = math.gcd(m, m2)
            # If g == 1, they are always compatible (any residues are distinct mod 1)
            if g > 1 and r % g == r2 % g:
                return False
        return True

    def find_residue(placed, m):
        """Find a compatible residue for modulus m."""
        for r in range(m):
            if is_compatible(placed, m, r):
                return r
        return None

    # Greedy initial construction
    placed = []
    for m in candidates:
        if time_up():
            break
        r = find_residue(placed, m)
        if r is not None:
            placed.append((m, r))

    best_classes = list(placed)
    best_score = sum(1.0 / m for m, r in best_classes)

    # Local search: try adding, swapping, and 2-opt moves
    def score_of(classes):
        return sum(1.0 / m for m, r in classes)

    def local_search(classes, max_iters=50000):
        nonlocal best_score, best_classes
        current = list(classes)
        current_score = score_of(current)
        improved = True
        iters = 0

        while improved and iters < max_iters and not time_up():
            improved = False
            iters += 1

            # 1. Try adding a new modulus
            for m_new in candidates:
                if any(m == m_new for m, r in current):
                    continue
                r_new = find_residue(current, m_new)
                if r_new is not None:
                    new_score = current_score + 1.0 / m_new
                    if new_score > current_score + 1e-12:
                        current = current + [(m_new, r_new)]
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_classes = list(current)
                        break

            if improved:
                continue

            # 2. Try swapping one modulus for another
            for i in range(len(current)):
                if time_up():
                    break
                m_i, r_i = current[i]
                without_i = current[:i] + current[i+1:]
                base_score = current_score - 1.0 / m_i

                for m_new in candidates:
                    if m_new == m_i:
                        continue
                    if any(m == m_new for m, r in current):
                        continue
                    r_new = find_residue(without_i, m_new)
                    if r_new is not None:
                        new_score = base_score + 1.0 / m_new
                        if new_score > current_score + 1e-12:
                            current = without_i + [(m_new, r_new)]
                            current_score = new_score
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_classes = list(current)
                            break
                if improved:
                    break

            if improved:
                continue

            # 3. Try 2-opt: remove one, add two
            for i in range(len(current)):
                if time_up():
                    break
                m_i, r_i = current[i]
                without_i = current[:i] + current[i+1:]
                base_score = current_score - 1.0 / m_i

                # Try adding two new moduli
                for m_a in candidates:
                    if m_a == m_i:
                        continue
                    if any(m == m_a for m, r in current):
                        continue
                    r_a = find_residue(without_i, m_a)
                    if r_a is None:
                        continue
                    temp1 = without_i + [(m_a, r_a)]
                    for m_b in candidates:
                        if m_b == m_i or m_b == m_a:
                            continue
                        if any(m == m_b for m, r in current):
                            continue
                        r_b = find_residue(temp1, m_b)
                        if r_b is not None:
                            new_score = base_score + 1.0 / m_a + 1.0 / m_b
                            if new_score > current_score + 1e-12:
                                current = temp1 + [(m_b, r_b)]
                                current_score = new_score
                                improved = True
                                if current_score > best_score:
                                    best_score = current_score
                                    best_classes = list(current)
                                break
                    if improved:
                        break
                if improved:
                    break

        return current, current_score

    if not time_up():
        best_classes, best_score = local_search(best_classes)

    # Final polish: try to squeeze in more small moduli
    if not time_up():
        for m in candidates:
            if time_up():
                break
            if any(mod == m for mod, r in best_classes):
                continue
            r = find_residue(best_classes, m)
            if r is not None:
                best_classes.append((m, r))
                best_score = score_of(best_classes)

    result = [[m, r] for m, r in best_classes]
    return result
