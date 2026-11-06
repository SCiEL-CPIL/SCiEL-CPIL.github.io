"""Odd covering system with moduli dividing 51975; minimize uncovered residues."""
import time

BASE = 51975
DEADLINE = 130.0  # safety margin under the 150 s wall-clock limit

# All odd divisors of 51975 = 3^3 * 5^2 * 7 * 11, excluding 1.
def _divisors():
    divs = []
    for a in range(4):
        for b in range(3):
            for c in range(2):
                for d in range(2):
                    m = (3 ** a) * (5 ** b) * (7 ** c) * (11 ** d)
                    if m > 1:
                        divs.append(m)
    return sorted(divs)

DIVS = _divisors()
N = len(DIVS)


def _score(covered):
    return BASE - int(covered.count(1))


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE

    # Descending moduli: sparse baseline first, small (high-leverage) moduli saved.
    order = sorted(range(N), key=lambda i: -DIVS[i])

    covered = bytearray(BASE)
    chosen = [None] * N
    for i in order:
        m = DIVS[i]
        best_r, best_g = 0, -1
        for r in range(m):
            g = covered[r::m].count(0)
            if g > best_g:
                best_g = g
                best_r = r
        chosen[i] = best_r
        covered[best_r::m] = b'\x01' * len(range(best_r, BASE, m))

    best_score = _score(covered)
    best_chosen = list(chosen)

    # Local search: reassign residues to maximize marginal gain (or minimize loss).
    improved = True
    while improved and (time.time() - t0) < deadline:
        improved = False
        for i in range(N):
            m = DIVS[i]
            cur = chosen[i]
            best_r, best_g = cur, -1
            for r in range(m):
                if r == cur:
                    continue
                # marginal gain of switching i from cur to r
                g = covered[r::m].count(0) - covered[cur::m].count(1)
                if g > best_g:
                    best_g = g
                    best_r = r
            if best_r != cur:
                covered[cur::m] = b'\x00' * len(range(cur, BASE, m))
                covered[best_r::m] = b'\x01' * len(range(best_r, BASE, m))
                chosen[i] = best_r
                sc = _score(covered)
                if sc < best_score:
                    best_score = sc
                    best_chosen = list(chosen)
                improved = True

    # Safety margin: if we are close to the deadline, return the best snapshot.
    if (time.time() - t0) > deadline - 5:
        return [[DIVS[i], best_chosen[i]] for i in range(N)]

    # One more pass to confirm optimality of the current assignment.
    for i in range(N):
        m = DIVS[i]
        cur = chosen[i]
        for r in range(m):
            if r == cur:
                continue
            g = covered[r::m].count(0) - covered[cur::m].count(1)
            if g > 0:
                covered[cur::m] = b'\x00' * len(range(cur, BASE, m))
                covered[r::m] = b'\x01' * len(range(r, BASE, m))
                chosen[i] = r
                sc = _score(covered)
                if sc < best_score:
                    best_score = sc
                    best_chosen = list(chosen)

    return [[DIVS[i], best_chosen[i]] for i in range(N)]
