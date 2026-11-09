"""Odd covering systems with moduli dividing 51975; local-search over greedy seed."""
import time
import random

BASE = 51975
# all odd divisors > 1 of 51975
_DIVS = []
for _d in range(3, BASE + 1, 2):
    if BASE % _d == 0:
        _DIVS.append(_d)
_DIVS.sort()

def _uncovered_count(covered):
    return covered.count(0)

def _build_greedy(order, rng=None):
    covered = bytearray(BASE)
    classes = []
    for m in order:
        best_r = 0
        best_c = -1
        # choose residue maximizing newly covered
        for r in range(m):
            c = covered[r::m].count(0)
            if c > best_c:
                best_c = c
                best_r = r
            if best_c == BASE // m:
                break
        classes.append([m, best_r])
        covered[best_r::m] = b'\x01' * len(range(best_r, BASE, m))
    return classes, covered

def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin under 120s

    rng = random.Random(12345)

    # Initial greedy in increasing modulus order
    classes, covered = _build_greedy(_DIVS)
    best_classes = [list(c) for c in classes]
    best_score = _uncovered_count(covered)

    # Local search: try swapping residues or swapping which modulus is used
    # Represent state as list of [m, r].
    def evaluate(classes):
        cov = bytearray(BASE)
        for m, r in classes:
            cov[r::m] = b'\x01' * len(range(r, BASE, m))
        return cov

    cur_classes = [list(c) for c in classes]
    cur_cov = covered
    cur_score = best_score

    # Precompute for speed: for each modulus m, the list of residues is just range(m)
    # Try local moves: change residue of a class, or swap two classes' moduli.
    n = len(cur_classes)

    # Move 1: change residue of one class
    # Move 2: swap residues between two classes
    # Move 3: swap moduli between two classes (keeping residues valid)

    # We'll do iterative improvement with random restarts of the neighborhood order.
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # shuffle class order to vary neighborhood
        idx_order = list(range(n))
        rng.shuffle(idx_order)
        for i in idx_order:
            if time.time() > deadline:
                break
            m_i, r_i = cur_classes[i]
            # try all residues for class i
            # compute current covered excluding class i quickly? too slow.
            # Instead, evaluate full coverage for candidate.
            best_r = r_i
            best_sc = cur_score
            # only try a subset if too slow
            for r in range(m_i):
                if r == r_i:
                    continue
                # build coverage
                cov = bytearray(BASE)
                for j in range(n):
                    m_j, r_j = cur_classes[j]
                    if j == i:
                        rr = r
                    else:
                        rr = r_j
                    cov[rr::m_j] = b'\x01' * len(range(rr, BASE, m_j))
                sc = cov.count(0)
                if sc < best_sc:
                    best_sc = sc
                    best_r = r
            if best_r != r_i:
                cur_classes[i][1] = best_r
                cur_score = best_sc
                cur_cov = evaluate(cur_classes)
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
                    # swap residues
                    if r_i < m_j and r_j < m_i:
                        cur_classes[i][1] = r_j
                        cur_classes[j][1] = r_i
                        cov = evaluate(cur_classes)
                        sc = cov.count(0)
                        if sc < cur_score:
                            cur_score = sc
                            cur_cov = cov
                            improved = True
                            if cur_score < best_score:
                                best_score = cur_score
                                best_classes = [list(c) for c in cur_classes]
                        else:
                            cur_classes[i][1] = r_i
                            cur_classes[j][1] = r_j

    return best_classes
