"""HP lattice protein folding: randomized growth + local-search polish, deadline-guarded."""
# EVOLVE-BLOCK-START
import random
import time

_START = time.time()
_DEADLINE = _START + 205.0  # safety margin before the 210 s wall-clock kill


def _neighbours(point):
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _grow(sequence, hset, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain = [start]
    occupied = {start}
    n = len(sequence)
    for i in range(1, n):
        best_score = None
        best_site = None
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if i in hset:
                for q in _neighbours(site):
                    if q in occupied:
                        gain += 1
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            score = gain * 3 + free + rng.random()
            if best_score is None or score > best_score:
                best_score = score
                best_site = site
        if best_site is None:
            return None
        occupied.add(best_site)
        chain.append(best_site)
    return chain


def _local_search(sequence, chain, hset, budget, rng):
    """Incremental single-residue moves + 2-opt reversals to lower the energy."""
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if i in hset:
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and j in hset:
                    contacts += 1
    energy = -contacts

    def _move_delta(idx, new):
        old = chain[idx]
        if new in where:
            return 0
        d = 0
        if idx in hset:
            for q in _neighbours(old):
                j = where.get(q)
                if j is not None and abs(j - idx) > 1 and j in hset:
                    d -= 1
            for q in _neighbours(new):
                j = where.get(q)
                if j is not None and abs(j - idx) > 1 and j in hset:
                    d += 1
        return d

    def _apply(idx, new):
        old = chain[idx]
        del where[old]
        chain[idx] = new
        where[new] = idx

    end = time.time() + budget
    order = list(range(n))
    while time.time() < end:
        improved = False
        rng.shuffle(order)
        for idx in order:
            for cand in _neighbours(chain[idx]):
                d = _move_delta(idx, cand)
                if d < 0:
                    _apply(idx, cand)
                    energy += d
                    improved = True
                    break
        # 2-opt reversal pass
        for a in range(1, n - 2):
            for b in range(a + 2, n):
                pa, pb = chain[a], chain[b]
                if pa == pb:
                    continue
                da = _neighbours(pa)
                db = _neighbours(pb)
                if da[0] == pb or da[1] == pb or da[2] == pb or da[3] == pb:
                    continue
                if db[0] == pa or db[1] == pa or db[2] == pa or db[3] == pa:
                    continue
                d = 0
                if a in hset:
                    for q in da:
                        j = where.get(q)
                        if j is not None and abs(j - a) > 1 and j in hset:
                            d -= 1
                    for q in db:
                        j = where.get(q)
                        if j is not None and abs(j - a) > 1 and j in hset:
                            d += 1
                if b in hset:
                    for q in db:
                        j = where.get(q)
                        if j is not None and abs(j - b) > 1 and j in hset:
                            d -= 1
                    for q in da:
                        j = where.get(q)
                        if j is not None and abs(j - b) > 1 and j in hset:
                            d += 1
                if d < 0:
                    for k in range(a, b + 1):
                        del where[chain[k]]
                    seg = chain[a:b + 1]
                    seg.reverse()
                    for k, p in enumerate(seg):
                        chain[a + k] = p
                        where[p] = a + k
                    energy += d
                    improved = True
        if not improved:
            break
    return chain, energy


def fold(sequence, dimension):
    """Best conformation from time-aware randomized growth + local search."""
    n = len(sequence)
    hset = {i for i, c in enumerate(sequence) if c == "H"}
    seed = 12345 + n * 7919
    rng = random.Random(seed)

    best = None
    best_energy = 0  # energy is <= 0; 0 means no valid fold yet

    def _consider(chain, energy):
        nonlocal best, best_energy
        if best is None or energy < best_energy:
            best = chain
            best_energy = energy

    # A straight line is always a valid self-avoiding conformation.
    straight = [(i, 0) for i in range(n)]
    _consider(straight, _energy(sequence, straight))

    if n <= 1:
        return [list(p) for p in best]

    # Time-aware restarts: more local-search budget per restart as time goes on.
    base_ls = 0.02 if n <= 30 else (0.06 if n <= 50 else (0.15 if n <= 70 else 0.30))
    max_ls = 1.5 if n <= 30 else (3.0 if n <= 50 else (6.0 if n <= 70 else 9.0))
    growths_per_round = 40 if n <= 30 else (20 if n <= 50 else 10)
    rounds = 0

    while time.time() < _DEADLINE - 3.0:
        rounds += 1
        ls_budget = min(base_ls * (1 + 0.08 * rounds), max_ls)
        # Stop early if we are close to the deadline and the budget would overrun.
        if time.time() + ls_budget > _DEADLINE - 2.0:
            ls_budget = max(0.0, _DEADLINE - 2.0 - time.time())
            if ls_budget < 0.01:
                break
        for _ in range(growths_per_round):
            if time.time() > _DEADLINE - 2.0:
                break
            chain = _grow(sequence, hset, dimension, rng)
            if chain is None:
                continue
            chain, energy = _local_search(sequence, chain, hset, ls_budget, rng)
            _consider(chain, energy)

    # Final high-precision polish pass on the best conformation.
    if best is not None and time.time() < _DEADLINE - 1.5:
        polish = max(0.0, _DEADLINE - 1.5 - time.time())
        if polish > 0.05:
            best, best_energy = _local_search(sequence, list(best), hset, polish, rng)

    return [list(p) for p in best]
# EVOLVE-BLOCK-END
