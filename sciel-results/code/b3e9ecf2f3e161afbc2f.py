"""HP lattice protein folding: fast greedy growth + deadline-guarded local search."""
# EVOLVE-BLOCK-START
import random
import time
import math

_START_TIME = time.time()
_DEADLINE = 200.0  # Safety margin before the 210s kill


def _remaining():
    return max(0.0, _DEADLINE - (time.time() - _START_TIME))


def _neighbours(point):
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _grow(sequence, dimension, rng):
    """Randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    n = len(sequence)
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _energy(sequence, chain):
    """Energy = minus the number of H-H contacts (each counted once)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _local_search(sequence, chain, budget):
    """Improve a conformation via single-residue moves (incremental energy) + reversals."""
    n = len(sequence)
    cur = list(chain)
    occupied = {p: i for i, p in enumerate(cur)}
    best = list(cur)
    best_e = _energy(sequence, cur)
    cur_e = best_e
    T = 1.5
    t_end = time.time() + budget
    while time.time() < t_end:
        i = random.randrange(n)
        p = cur[i]
        moved = False
        for np in _neighbours(p):
            if np in occupied:
                continue
            if i > 0 and np == cur[i - 1]:
                continue
            if i < n - 1 and np == cur[i + 1]:
                continue
            # compute energy delta for moving i from p to np
            delta = 0
            if sequence[i] == "H":
                # contacts i had with p's neighbours (j != i-1, i+1)
                for q in _neighbours(p):
                    j = occupied.get(q)
                    if j is not None and j > i + 1 and sequence[j] == "H":
                        delta -= 1
                # new contacts at np
                for q in _neighbours(np):
                    j = occupied.get(q)
                    if j is not None and j > i + 1 and sequence[j] == "H":
                        delta += 1
            else:
                # i is P: moving it breaks no H-H contact, but it may have been a contact partner
                # actually P never participates, so delta from i's own contacts is 0.
                # But other H residues contacting i? No, i is P. So delta = 0.
                pass
            # apply
            del occupied[p]
            occupied[np] = i
            cur[i] = np
            new_e = cur_e + delta
            if new_e < cur_e or random.random() < math.exp(-max(0, new_e - cur_e) / T):
                cur_e = new_e
                if cur_e < best_e:
                    best_e = cur_e
                    best = list(cur)
                moved = True
                break
            else:
                # undo
                del occupied[np]
                occupied[p] = i
                cur[i] = p
        if not moved and n > 2:
            # try a reversal of a random segment
            a = random.randrange(n)
            b = random.randrange(n)
            if a > b:
                a, b = b, a
            if b - a >= 1:
                cur[a:b + 1] = cur[a:b + 1][::-1]
                occupied = {p: i for i, p in enumerate(cur)}
                e_r = _energy(sequence, cur)
                if e_r < cur_e or random.random() < math.exp(-max(0, e_r - cur_e) / T):
                    cur_e = e_r
                    if cur_e < best_e:
                        best_e = cur_e
                        best = list(cur)
                else:
                    cur[a:b + 1] = cur[a:b + 1][::-1]
                    occupied = {p: i for i, p in enumerate(cur)}
        T *= 0.9995
        if T < 0.05:
            T = 0.05
    return best, best_e


def fold(sequence, dimension):
    """Best conformation via many greedy growths then local search on the best."""
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 1

    # Phase 1: greedy growths until little time remains
    while _remaining() > 5.0:
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        e = _energy(sequence, chain)
        if e < best_energy:
            best, best_energy = chain, e
    if best is None:
        best = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best)

    # Phase 2: local search on the best
    remaining = _remaining()
    if remaining > 1.0:
        best, best_energy = _local_search(sequence, best, remaining - 1.0)

    return [list(p) for p in best]
# EVOLVE-BLOCK-END
