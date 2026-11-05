"""HP lattice protein folding: Hamiltonian-path skeleton + fast local search."""
# EVOLVE-BLOCK-START
import random
import time

_DEADLINE = 205.0  # safety margin before 210s wall-clock kill
_START = time.time()


def _remaining():
    return _DEADLINE - time.time()


def _neighbours(p):
    x, y = p
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _snake_rows(rows, cols):
    """Hamiltonian path over a rows x cols grid via row-by-row snake."""
    path = []
    for r in range(rows):
        xs = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
        for c in xs:
            path.append((c, r))
    return path


def _generate_skeletons(n):
    """All snake Hamiltonian paths that fit exactly n cells, sorted by contact density."""
    cands = []
    for rows in range(1, n + 1):
        if n % rows == 0:
            cols = n // rows
            path = _snake_rows(rows, cols)
            # Count non-consecutive adjacent pairs (potential contacts)
            pos = {p: i for i, p in enumerate(path)}
            contacts = 0
            for p, i in pos.items():
                for q in _neighbours(p):
                    j = pos.get(q)
                    if j is not None and j > i + 1:
                        contacts += 1
            cands.append((contacts, path))
    cands.sort(key=lambda t: -t[0])
    return [p for _, p in cands]


def _build_maps(chain, seq):
    where = {p: i for i, p in enumerate(chain)}
    occ = set(chain)
    hh = [seq[i] == "H" for i in range(len(chain))]
    return where, occ, hh


def _energy_from_map(where, seq, hh):
    contacts = 0
    for i, p in enumerate(where):
        if hh[i]:
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and hh[j]:
                    contacts += 1
    return -contacts


def _move_gain(i, seq, hh, where, occ, chain):
    """Return (delta_energy, new_site) for moving residue i to a free neighbour, or None."""
    p = chain[i]
    best = None
    for q in _neighbours(p):
        if q in occ:
            continue
        # delta = -contacts_after + contacts_before for residue i
        before = 0
        after = 0
        if hh[i]:
            for r in _neighbours(p):
                j = where.get(r)
                if j is not None and j != i - 1 and j != i + 1 and hh[j]:
                    before += 1
            for r in _neighbours(q):
                j = where.get(r)
                if j is not None and j != i - 1 and j != i + 1 and hh[j]:
                    after += 1
        delta = -(after) + before
        if best is None or delta < best[0]:
            best = (delta, q)
    return best


def _local_search(sequence, chain, hh, where, occ, max_iters):
    """First-improvement single-residue moves, keeping the best conformation."""
    n = len(chain)
    best_energy = _energy_from_map(where, sequence, hh)
    best_chain = list(chain)
    it = 0
    while it < max_iters and _remaining() > 0.5:
        improved = False
        rng = random.Random(it)
        order = list(range(n))
        rng.shuffle(order)
        for i in order:
            if i == 0 or i == n - 1:
                continue
            mv = _move_gain(i, sequence, hh, where, occ, chain)
            if mv is None:
                continue
            delta, q = mv
            if delta >= 0:
                continue
            p = chain[i]
            # apply move
            occ.remove(p)
            occ.add(q)
            del where[p]
            where[q] = i
            chain[i] = q
            best_energy += delta
            improved = True
        it += 1
        if not improved:
            break
    # record best
    if best_energy < 0:
        return best_chain, best_energy
    return best_chain, best_energy


def _refine(sequence, chain, hh, where, occ, iters):
    out_chain, out_e = _local_search(sequence, chain, hh, where, occ, iters)
    return out_chain, out_e


def fold(sequence, dimension):
    n = len(sequence)
    if n <= 2:
        return [[i, 0] for i in range(n)]

    # fallback straight line
    straight = [[i, 0] for i in range(n)]

    skeletons = _generate_skeletons(n)
    if not skeletons:
        return straight

    hh = [sequence[i] == "H" for i in range(n)]

    best_chain = straight
    best_energy = 0
    # energy of straight line (no contacts) = 0

    # budget per skeleton
    total_budget = max(1, int(_remaining() * 0.9))
    per_skel = max(30, total_budget // max(1, len(skeletons)))

    for path in skeletons:
        if _remaining() < 1.0:
            break
        chain = list(path)
        # normalize: shift so min coords are 0 (already non-negative)
        where, occ, _ = _build_maps(chain, sequence)
        e = _energy_from_map(where, sequence, hh)
        if e < best_energy:
            best_energy = e
            best_chain = list(chain)
        # refine
        chain2, e2 = _refine(sequence, chain, hh, where, occ, per_skel)
        if e2 < best_energy:
            best_energy = e2
            best_chain = list(chain2)

    if best_energy >= 0:
        return straight
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
