"""Rewritten from scratch: multi-start greedy + simulated annealing on a coverage-count array."""
import time
import random
import numpy as np

BASE = 51975


def _divisors():
    divs = []
    for d in range(3, BASE + 1, 2):
        if BASE % d == 0:
            divs.append(d)
    return divs


def _uncovered_count(cov):
    return int((cov == 0).sum())


def _greedy(order):
    cov = np.zeros(BASE, dtype=np.int8)
    classes = []
    for m in order:
        best_r = 0
        best_gain = -1
        # try residues; for large m sample to keep time bounded, but exact for small m
        if m <= 400:
            for r in range(m):
                g = int((cov[r::m] == 0).sum())
                if g > best_gain:
                    best_gain = g
                    best_r = r
        else:
            # sample a subset of residues for large moduli
            step = max(1, m // 400)
            for r in range(0, m, step):
                g = int((cov[r::m] == 0).sum())
                if g > best_gain:
                    best_gain = g
                    best_r = r
        cov[best_r::m] += 1
        classes.append([m, best_r])
    return classes, cov


def _local_refine(classes, cov, deadline):
    # simple hill-climbing: try changing residues of a few random classes
    rng = random.Random(12345)
    n = len(classes)
    for _ in range(20000):
        if time.time() >= deadline:
            break
        i = rng.randrange(n)
        m, r = classes[i]
        if m > 300:
            continue
        # try a random new residue
        new_r = rng.randrange(m)
        if new_r == r:
            continue
        before = _uncovered_count(cov)
        cov[r::m] -= 1
        cov[new_r::m] += 1
        after = _uncovered_count(cov)
        if after < before:
            classes[i][1] = new_r
        else:
            cov[r::m] += 1
            cov[new_r::m] -= 1
    return classes, cov


def construct():
    deadline = time.time() + 140.0
    divs = _divisors()
    rng = random.Random(7)

    # Multi-start greedy
    best_classes = None
    best_cov = None
    best_score = None
    for order in (divs, list(reversed(divs))):
        classes, cov = _greedy(order)
        sc = _uncovered_count(cov)
        if best_score is None or sc < best_score:
            best_score = sc
            best_classes = classes
            best_cov = cov
        if time.time() >= deadline:
            break

    # Local refine
    best_classes, best_cov = _local_refine(best_classes, best_cov, deadline)
    best_score = _uncovered_count(best_cov)

    # Simulated annealing
    cov = best_cov.copy()
    classes = [list(c) for c in best_classes]
    n = len(classes)
    # build index of modulus positions
    temp = 3.0
    cooling = 0.999
    iters = 0
    while time.time() < deadline and temp > 0.05:
        iters += 1
        move = rng.random()
        if move < 0.5:
            # change residue of a class
            i = rng.randrange(n)
            m, r = classes[i]
            new_r = rng.randrange(m)
            if new_r == r:
                continue
            before = _uncovered_count(cov)
            cov[r::m] -= 1
            cov[new_r::m] += 1
            after = _uncovered_count(cov)
            delta = after - before
            if delta <= 0 or rng.random() < np.exp(-delta / temp):
                classes[i][1] = new_r
            else:
                cov[r::m] += 1
                cov[new_r::m] -= 1
            if after < best_score:
                best_score = after
                best_classes = [list(c) for c in classes]
                best_cov = cov.copy()
        else:
            # swap moduli between two classes (distinct moduli preserved)
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                continue
            mi, ri = classes[i]
            mj, rj = classes[j]
            before = _uncovered_count(cov)
            cov[ri::mi] -= 1
            cov[rj::mj] -= 1
            cov[ri::mj] += 1
            cov[rj::mi] += 1
            after = _uncovered_count(cov)
            delta = after - before
            if delta <= 0 or rng.random() < np.exp(-delta / temp):
                classes[i] = [mj, ri]
                classes[j] = [mi, rj]
            else:
                cov[ri::mi] += 1
                cov[rj::mj] += 1
                cov[ri::mj] -= 1
                cov[rj::mi] -= 1
            if after < best_score:
                best_score = after
                best_classes = [list(c) for c in classes]
                best_cov = cov.copy()
        temp *= cooling

    # final safety: recompute score from best_classes to ensure validity
    if best_classes is None:
        best_classes, best_cov = _greedy(divs)
    # verify distinct moduli
    mods = [c[0] for c in best_classes]
    if len(set(mods)) != len(mods):
        # fallback to greedy
        best_classes, best_cov = _greedy(divs)
    return best_classes
