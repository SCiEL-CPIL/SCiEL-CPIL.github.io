"""HP lattice protein folding: greedy growth + local search (2-opt/3-opt + SA) with deadline guard."""
# EVOLVE-BLOCK-START
import random
import time

_DEADLINE = 205.0  # wall-clock safety margin under the 210s kill
_START = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _neighbours(point):
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _energy(sequence, chain):
    """Minus the number of non-adjacent H-H contacts on neighbouring lattice sites."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _is_valid(chain):
    if len(chain) < 2:
        return True
    seen = set(chain)
    if len(seen) != len(chain):
        return False
    for i in range(len(chain) - 1):
        dx = abs(chain[i][0] - chain[i + 1][0])
        dy = abs(chain[i][1] - chain[i + 1][1])
        if dx + dy != 1:
            return False
    return True


def _grow(sequence, rng):
    """Randomized greedy chain growth: prefer sites creating new H-H contacts, then free space."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    for i in range(1, len(sequence)):
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
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _try_move(chain, i, j, new_seg):
    """Try replacing chain[i..j] with new_seg (same length). Return new chain or None."""
    n = len(chain)
    # check self-avoidance of the whole thing
    prefix = chain[:i]
    suffix = chain[j + 1:]
    candidate = prefix + new_seg + suffix
    # unit steps at boundaries
    if i > 0:
        dx = abs(prefix[-1][0] - new_seg[0][0])
        dy = abs(prefix[-1][1] - new_seg[0][1])
        if dx + dy != 1:
            return None
    if j + 1 < n:
        dx = abs(new_seg[-1][0] - suffix[0][0])
        dy = abs(new_seg[-1][1] - suffix[0][1])
        if dx + dy != 1:
            return None
    # internal unit steps
    for k in range(len(new_seg) - 1):
        dx = abs(new_seg[k][0] - new_seg[k + 1][0])
        dy = abs(new_seg[k][1] - new_seg[k + 1][1])
        if dx + dy != 1:
            return None
    # self-avoidance
    if len(set(candidate)) != len(candidate):
        return None
    return candidate


def _local_search(sequence, chain, budget):
    """Improve chain via 2-opt / 3-opt segment reversals + simulated annealing."""
    n = len(chain)
    best = list(chain)
    best_e = _energy(sequence, best)
    current = list(chain)
    current_e = best_e
    where = {p: i for i, p in enumerate(current)}
    temp = 3.0
    cooling = 0.999
    t0 = time.time()

    def _rebuild_where(c):
        return {p: i for i, p in enumerate(c)}

    while _time_left() > 0.2 and (time.time() - t0) < budget:
        # pick a segment length
        L = random.choice([2, 3, 3, 4, 5])
        if L >= n:
            L = max(2, n - 1)
        i = random.randint(0, n - L - 1)
        j = i + L
        seg = current[i:j + 1]
        # try reversing the segment
        rev = seg[::-1]
        # quick boundary unit-step check
        ok = True
        if i > 0:
            dx = abs(current[i - 1][0] - rev[0][0])
            dy = abs(current[i - 1][1] - rev[0][1])
            if dx + dy != 1:
                ok = False
        if ok and j + 1 < n:
            dx = abs(rev[-1][0] - current[j + 1][0])
            dy = abs(rev[-1][1] - current[j + 1][1])
            if dx + dy != 1:
                ok = False
        if not ok:
            continue
        candidate = current[:i] + rev + current[j + 1:]
        if len(set(candidate)) != len(candidate):
            continue
        new_e = _energy(sequence, candidate)
        delta = new_e - current_e
        if delta < 0 or random.random() < pow(2.718281828, -(-delta) / max(temp, 1e-6)):
            current = candidate
            current_e = new_e
            if new_e < best_e:
                best = list(current)
                best_e = new_e
        temp *= cooling
        if temp < 0.05:
            temp = 0.05
    return best, best_e


def fold(sequence, dimension):
    """Best conformation from multiple greedy growths refined by local search, with deadline guard."""
    rng = random.Random(12345 + len(sequence))
    best = None
    best_energy = 1  # energies are negative or zero; 1 is a safe "worse" sentinel

    n = len(sequence)
    # Phase 1: generate a pool of diverse starting conformations via greedy growth
    starts = []
    for _ in range(60):
        if _time_left() < 2.0:
            break
        c = _grow(sequence, rng)
        if c is None or len(c) != n:
            continue
        if not _is_valid(c):
            continue
        starts.append(c)
        if len(starts) >= 25:
            break

    if not starts:
        # fallback: a spiral / straight line that is always self-avoiding
        starts.append([(i, 0) for i in range(n)])

    # Phase 2: refine each start with local search, keeping the global best
    # allocate per-start budget so total stays within deadline
    per_budget = max(0.5, min(8.0, (_time_left() - 1.0) / max(1, len(starts))))
    for s in starts:
        if _time_left() < 1.0:
            break
        refined, e = _local_search(sequence, s, per_budget)
        if e < best_energy:
            best_energy = e
            best = refined

    if best is None:
        best = starts[0]

    # final safety: ensure validity
    if not _is_valid(best):
        best = [(i, 0) for i in range(n)]

    return [list(p) for p in best]
# EVOLVE-BLOCK-END
