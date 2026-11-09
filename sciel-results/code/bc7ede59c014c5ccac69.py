"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import random
import math

DENOMS = list(range(2, 49))
N = len(DENOMS)
RECS = [Fraction(1, d) for d in DENOMS]

def _sum_recips(idx):
    s = Fraction(0, 1)
    for i in idx:
        s += RECS[i]
    return s

def construct():
    t0 = time.time()
    DEADLINE = t0 + 85.0
    best_deficit = None
    best_indices = None

    def consider(indices):
        nonlocal best_deficit, best_indices
        if not indices:
            return
        s = _sum_recips(indices)
        if s < 1:
            d = 1 - s
            if best_deficit is None or d < best_deficit:
                best_deficit = d
                best_indices = list(indices)
                return True
        return False

    # Greedy baseline
    greedy = []
    tot = Fraction(0, 1)
    for i in range(N):
        if tot + RECS[i] < 1:
            greedy.append(i)
            tot += RECS[i]
    consider(greedy)

    # Local search
    if best_indices is not None:
        sel = set(best_indices)
        cur = _sum_recips(best_indices)
        improved = True
        while improved and time.time() < DEADLINE - 3:
            improved = False
            unsel = sorted(set(range(N)) - sel)
            for i in unsel:
                if time.time() > DEADLINE - 3:
                    break
                ns = cur + RECS[i]
                if ns < 1:
                    nd = 1 - ns
                    if nd < best_deficit:
                        best_indices.append(i)
                        best_deficit = nd
                        sel.add(i)
                        cur = ns
                        improved = True
                        break
            if not improved:
                unsel = sorted(set(range(N)) - sel)
                for out in sorted(sel):
                    if time.time() > DEADLINE - 3:
                        break
                    for inn in unsel:
                        if time.time() > DEADLINE - 3:
                            break
                        ns = cur - RECS[out] + RECS[inn]
                        if ns < 1:
                            nd = 1 - ns
                            if nd < best_deficit:
                                best_indices = [x for x in best_indices if x != out] + [inn]
                                best_deficit = nd
                                sel.discard(out)
                                sel.add(inn)
                                cur = ns
                                improved = True
                                break
                    if improved:
                        break

    # Pair-add local search
    if best_indices is not None and time.time() < DEADLINE - 3:
        sel = set(best_indices)
        cur = _sum_recips(best_indices)
        unsel = sorted(set(range(N)) - sel)
        for a in range(len(unsel)):
            if time.time() > DEADLINE - 3:
                break
            for b in range(a + 1, len(unsel)):
                if time.time() > DEADLINE - 3:
                    break
                i, j = unsel[a], unsel[b]
                ns = cur + RECS[i] + RECS[j]
                if ns < 1:
                    nd = 1 - ns
                    if nd < best_deficit:
                        best_indices.extend([i, j])
                        best_deficit = nd
                        sel.add(i)
                        sel.add(j)
                        cur = ns

    # Exhaustive small subsets (size 2..7)
    from itertools import combinations
    for sz in range(2, 8):
        if time.time() > DEADLINE - 5:
            break
        for combo in combinations(range(N), sz):
            if time.time() > DEADLINE - 5:
                break
            s = sum(RECS[i] for i in combo)
            if s < 1:
                d = 1 - s
                if best_deficit is None or d < best_deficit:
                    best_deficit = d
                    best_indices = list(combo)

    # Randomized trials
    rng = random.Random(2024)
    def random_greedy():
        order = list(range(N))
        rng.shuffle(order)
        idx = []
        tot = Fraction(0, 1)
        for i in order:
            if tot + RECS[i] < 1:
                idx.append(i)
                tot += RECS[i]
        consider(idx)

    def random_subset():
        k = rng.randint(2, 18)
        idx = rng.sample(range(N), k)
        consider(idx)

    def perturbed_greedy():
        order = list(range(N))
        rng.shuffle(order)
        idx = []
        tot = Fraction(0, 1)
        for i in order:
            if rng.random() < 0.15:
                continue
            if tot + RECS[i] < 1:
                idx.append(i)
                tot += RECS[i]
        consider(idx)

    while time.time() < DEADLINE - 2:
        r = rng.random()
        if r < 0.35:
            random_greedy()
        elif r < 0.7:
            random_subset()
        else:
            perturbed_greedy()

    if best_indices is None:
        best_indices = greedy
    return best_indices
# EVOLVE-BLOCK-END
