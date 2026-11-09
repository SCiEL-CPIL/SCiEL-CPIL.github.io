"""Odd covering systems with moduli dividing 51975; NumPy-accelerated local search."""
import time
import random
import numpy as np

BASE = 51975
# all odd divisors > 1 of 51975
_DIVS = []
for _d in range(3, BASE + 1, 2):
    if BASE % _d == 0:
        _DIVS.append(_d)
_DIVS.sort()

def _compute_coverage(classes):
    """Compute coverage array using NumPy for speed."""
    cov = np.zeros(BASE, dtype=np.uint8)
    for m, r in classes:
        cov[r::m] = 1
    return cov

def _score(covered):
    """Count uncovered residues."""
    return int(np.sum(covered == 0))

def _build_greedy(order):
    """Greedy construction using NumPy."""
    covered = np.zeros(BASE, dtype=np.uint8)
    classes = []
    for m in order:
        best_r = 0
        best_c = -1
        for r in range(m):
            c = int(np.sum(covered[r::m] == 0))
            if c > best_c:
                best_c = c
                best_r = r
            if best_c == BASE // m:
                break
        classes.append([m, best_r])
        covered[best_r::m] = 1
    return classes, covered

def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin under 120s

    rng = random.Random(12345)

    # Initial greedy in increasing modulus order
    classes, covered = _build_greedy(_DIVS)
    best_classes = [list(c) for c in classes]
    best_score = _score(covered)

    cur_classes = [list(c) for c in classes]
    cur_score = best_score
    n = len(cur_classes)

    # Precompute residue ranges for each class index for faster access
    def get_residues(m):
        return list(range(m))

    # Local search with simulated annealing
    temperature = 10.0
    cooling = 0.95
    min_temp = 0.1

    improved = True
    while improved and time.time() < deadline:
        improved = False
        idx_order = list(range(n))
        rng.shuffle(idx_order)

        # Move 1: change residue of one class
        for i in idx_order:
            if time.time() > deadline:
                break
            m_i, r_i = cur_classes[i]
            best_r = r_i
            best_sc = cur_score
            # Try all residues for this class
            for r in range(m_i):
                if r == r_i:
                    continue
                # Compute new coverage
                old_val = cur_classes[i][1]
                cur_classes[i][1] = r
                cov = _compute_coverage(cur_classes)
                sc = _score(cov)
                cur_classes[i][1] = old_val

                if sc < best_sc:
                    best_sc = sc
                    best_r = r

            if best_r != r_i:
                cur_classes[i][1] = best_r
                cur_score = best_sc
                improved = True
                if cur_score < best_score:
                    best_score = cur_score
                    best_classes = [list(c) for c in cur_classes]

            if cur_score == 0:
                break

        # Move 2: swap residues between two classes
        if time.time() < deadline and cur_score > 0:
            for i in idx_order:
                if time.time() > deadline:
                    break
                for j in idx_order:
                    if j <= i:
                        continue
                    if time.time() > deadline:
                        break
                    m_i, r_i = cur_classes[i]
                    m_j, r_j = cur_classes[j]
                    # Check if swap is valid
                    if r_i < m_j and r_j < m_i:
                        old_r_i = cur_classes[i][1]
                        old_r_j = cur_classes[j][1]
                        cur_classes[i][1] = r_j
                        cur_classes[j][1] = r_i
                        cov = _compute_coverage(cur_classes)
                        sc = _score(cov)

                        if sc < cur_score:
                            cur_score = sc
                            improved = True
                            if cur_score < best_score:
                                best_score = cur_score
                                best_classes = [list(c) for c in cur_classes]
                        else:
                            cur_classes[i][1] = old_r_i
                            cur_classes[j][1] = old_r_j

        # Move 3: Try changing which moduli are used (swap modulus with another)
        if time.time() < deadline and cur_score > 0 and temperature > min_temp:
            # Pick two random classes and try swapping their moduli
            for _ in range(50):  # limited attempts per temperature
                if time.time() > deadline:
                    break
                i, j = rng.sample(range(n), 2)
                m_i, r_i = cur_classes[i]
                m_j, r_j = cur_classes[j]
                # Swap moduli, keeping residues if valid
                if r_i < m_j and r_j < m_i:
                    cur_classes[i][0] = m_j
                    cur_classes[j][0] = m_i
                    cov = _compute_coverage(cur_classes)
                    sc = _score(cov)
                    if sc < cur_score or (temperature > 0 and rng.random() < np.exp(-(sc - cur_score) / temperature)):
                        cur_score = sc
                        improved = True
                        if cur_score < best_score:
                            best_score = cur_score
                            best_classes = [list(c) for c in cur_classes]
                    else:
                        cur_classes[i][0] = m_i
                        cur_classes[j][0] = m_j
            temperature *= cooling

        # Cooling and early termination
        if cur_score == 0:
            break
        if not improved:
            # No improvement in this pass, cool down more aggressively
            temperature *= 0.9

    return best_classes
