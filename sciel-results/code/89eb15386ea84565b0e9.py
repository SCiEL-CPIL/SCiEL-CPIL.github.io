"""HP lattice protein folding: contact-aware grow + SA with segment-reversal moves."""
import random
import time

_T0 = time.time()
_LIMIT = 320.0          # safety margin under the 330 s kill
_MARGIN = 2.0           # stop new work this many seconds before the limit


def _time_left():
    return _LIMIT - (time.time() - _T0)


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Return (-contacts, contacts) for the chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts, contacts


def _grow(sequence, dimension, rng, contact_w=1.0, free_w=0.15, noise=0.15):
    """Greedy contact-aware growth; returns chain or None."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    n = len(sequence)
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((contact_w * gain + free_w * free + rng.random() * noise, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _sa(sequence, chain, dimension, rng, time_budget):
    """Simulated annealing with segment-reversal + local moves. Returns best chain."""
    n = len(chain)
    if n < 5:
        return chain
    best = list(chain)
    best_e, _ = _energy(sequence, best)
    cur = list(chain)
    cur_e, _ = _energy(sequence, cur)

    # temperature schedule
    t_start = 1.0
    t_end = 0.02
    # segment lengths to try for reversal
    seg_pool = [2, 3, 4, 5, max(2, n // 8), max(3, n // 5), max(4, n // 3)]
    seg_pool = sorted(set(s for s in seg_pool if 2 <= s <= n - 2))

    t0 = time.time()
    deadline = t0 + time_budget
    iters = 0
    while time.time() < deadline:
        # temperature
        frac = max(0.0, 1.0 - (time.time() - t0) / max(1e-9, time_budget))
        temp = t_end + (t_start - t_end) * frac
        iters += 1

        move = rng.random()
        if move < 0.6:
            # segment reversal
            L = seg_pool[rng.randrange(len(seg_pool))]
            i = rng.randrange(0, n - L)
            # reverse positions i..i+L
            new = cur[:i] + cur[i:i + L][::-1] + cur[i + L:]
        else:
            # local single-residue move: pick a residue, move it to a free neighbour
            i = rng.randrange(1, n - 1)
            cand = [q for q in _neighbours(cur[i]) if q not in set(cur)]
            if not cand:
                continue
            site = cand[rng.randrange(len(cand))]
            new = cur[:i] + [site] + cur[i + 1:]

        # quick validity: no duplicates (chain-adjacency still unit by construction)
        if len(set(new)) != n:
            continue

        new_e, _ = _energy(sequence, new)
        d = new_e - cur_e  # energies negative; lower is better
        if d <= 0 or rng.random() < (2.718281828459045 ** (-d / max(temp, 1e-9))):
            cur = new
            cur_e = new_e
            if cur_e < best_e:
                best = list(cur)
                best_e = cur_e
        # occasional long jump: large reversal
        if iters % 40 == 0 and n >= 10:
            L = rng.randrange(n // 4, n // 2 + 1)
            i = rng.randrange(0, n - L)
            new = cur[:i] + cur[i:i + L][::-1] + cur[i + L:]
            if len(set(new)) == n:
                new_e, _ = _energy(sequence, new)
                d = new_e - cur_e
                if d <= 0 or rng.random() < (2.718281828459045 ** (-d / max(temp, 1e-9))):
                    cur = new
                    cur_e = new_e
                    if cur_e < best_e:
                        best = list(cur)
                        best_e = cur_e
    return best


def fold(sequence, dimension):
    n = len(sequence)
    rng = random.Random(12345 + n)

    # fallback straight line
    straight = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]

    # 1) contact-aware greedy restarts to seed
    best, best_e = None, 0
    t0 = time.time()
    seed_budget = min(20.0, max(2.0, _time_left() * 0.15))
    while time.time() - t0 < seed_budget:
        if _time_left() < _MARGIN:
            break
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        e, _ = _energy(sequence, chain)
        if best is None or e < best_e:
            best, best_e = chain, e

    if best is None:
        best, best_e = straight, 0

    # 2) simulated annealing with segment reversals on the best seed
    if _time_left() > _MARGIN:
        sa_budget = max(1.0, _time_left() - _MARGIN - 1.0)
        improved = _sa(sequence, best, dimension, rng, sa_budget)
        e, _ = _energy(sequence, improved)
        if e < best_e:
            best, best_e = improved, e

    # 3) a couple more greedy+SA cycles if time remains
    while _time_left() > _MARGIN + 5.0:
        chain = _grow(sequence, dimension, rng, contact_w=1.2, free_w=0.1, noise=0.2)
        if chain is None:
            continue
        e, _ = _energy(sequence, chain)
        if e < best_e:
            best, best_e = chain, e
        if _time_left() > _MARGIN + 4.0:
            sa_budget = max(0.5, _time_left() - _MARGIN - 2.0)
            improved = _sa(sequence, best, dimension, rng, sa_budget)
            e2, _ = _energy(sequence, improved)
            if e2 < best_e:
                best, best_e = improved, e2

    return [list(p) for p in best]
