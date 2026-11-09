"""HP lattice protein folding: beam-search init + contact-map incremental polish + SA."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_DEADLINE = _GLOBAL_START + 205.0


def _remaining_time():
    return max(0.0, _GLOBAL_DEADLINE - time.time())


_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbours(point):
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _is_valid_chain(chain):
    seen = set()
    for i, p in enumerate(chain):
        if p in seen:
            return False
        seen.add(p)
        if i > 0:
            if abs(chain[i][0] - chain[i - 1][0]) + abs(chain[i][1] - chain[i - 1][1]) != 1:
                return False
    return True


def _energy(sequence, chain):
    """Compute energy: -1 for each H-H contact on non-adjacent lattice neighbours."""
    n = len(chain)
    if n < 2:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts // 2


# ---------------- Contact-map incremental energy ----------------

def _build_contact(sequence, chain):
    """Return (contacts, hmap, cmap).
    hmap: pos -> i for H residues.
    cmap: frozenset({i,j}) -> 1 for each H-H contact (i<j).
    energy = -len(cmap)."""
    n = len(chain)
    hmap = {}
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            hmap[p] = i
    cmap = {}
    for p, i in hmap.items():
        for q in _neighbours(p):
            j = hmap.get(q)
            if j is not None and abs(i - j) > 1:
                a, b = (i, j) if i < j else (j, i)
                cmap[(a, b)] = 1
    return len(cmap), hmap, cmap


def _move_energy_delta(sequence, chain, a, b, hmap, cmap):
    """Energy delta for reversing segment chain[a..b] inclusive.
    Returns (delta, new_cmap) or None if the move is invalid (self-avoid / junction)."""
    n = len(chain)
    if a >= b:
        return None
    # junction validity
    if a > 0:
        pp, pe = chain[a - 1], chain[b]
        if abs(pp[0] - pe[0]) + abs(pp[1] - pe[1]) != 1:
            return None
    if b < n - 1:
        ps, pn = chain[a], chain[b + 1]
        if abs(ps[0] - pn[0]) + abs(ps[1] - pn[1]) != 1:
            return None
    # self-avoidance of reversed segment vs prefix/suffix
    prefix = set(chain[:a])
    suffix = set(chain[b + 1:])
    seg = chain[a:b + 1]
    for p in seg:
        if p in prefix or p in suffix:
            return None

    # old contacts involving any index in [a, b]
    old = {}
    for (i, j) in list(cmap.keys()):
        if a <= i <= b or a <= j <= b:
            old[(i, j)] = 1
    # new contacts: recompute for H in segment and H in full chain
    seg_positions = {}
    for k in range(a, b + 1):
        new_idx = a + (b - k)
        seg_positions[chain[k]] = new_idx
    # build temporary hmap for new positions of segment
    new_hmap = dict(hmap)
    for p, idx in seg_positions.items():
        new_hmap[p] = idx
    new = {}
    for p, idx in seg_positions.items():
        if sequence[idx] != "H":
            continue
        for q in _neighbours(p):
            j = new_hmap.get(q)
            if j is not None and abs(idx - j) > 1 and sequence[j] == "H":
                x, y = (idx, j) if idx < j else (j, idx)
                new[(x, y)] = 1
    delta = len(new) - len(old)
    new_cmap = {k: 1 for k in cmap}
    for k in old:
        new_cmap.pop(k, None)
    for k, v in new.items():
        new_cmap[k] = v
    return delta, new_cmap


def _pivot_energy_delta(sequence, chain, i, angle, hmap, cmap):
    """Energy delta for rotating tail chain[i+1:] around chain[i] by `angle` quarter turns.
    Returns (delta, new_hmap, new_cmap, new_tail) or None if invalid."""
    n = len(chain)
    if i <= 0 or i >= n - 1:
        return None
    pivot = chain[i]
    tail = chain[i + 1:]
    prefix_set = set(chain[:i + 1])
    rotated = []
    for p in tail:
        x, y = p[0] - pivot[0], p[1] - pivot[1]
        for _ in range(angle):
            x, y = -y, x
        np_ = (pivot[0] + x, pivot[1] + y)
        rotated.append(np_)
    if rotated[0] == chain[i - 1]:
        return None
    for np_ in rotated:
        if np_ in prefix_set:
            return None

    old = {}
    for (a, b) in list(cmap.keys()):
        if a > i or b > i:
            old[(a, b)] = 1
    new_hmap = dict(hmap)
    for k, np_ in enumerate(rotated):
        idx = i + 1 + k
        if sequence[idx] == "H":
            new_hmap[np_] = idx
        else:
            new_hmap.pop(chain[i + 1 + k], None)
    new = {}
    for k, np_ in enumerate(rotated):
        idx = i + 1 + k
        if sequence[idx] != "H":
            continue
        for q in _neighbours(np_):
            j = new_hmap.get(q)
            if j is not None and abs(idx - j) > 1 and sequence[j] == "H":
                x, y = (idx, j) if idx < j else (j, idx)
                new[(x, y)] = 1
    delta = len(new) - len(old)
    new_cmap = {k: 1 for k in cmap}
    for k in old:
        new_cmap.pop(k, None)
    for k, v in new.items():
        new_cmap[k] = v
    return delta, new_hmap, new_cmap, rotated


# ---------------- Beam search initialization ----------------

def _beam_search(sequence, beam_width=60):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]

    def _score(chain, occupied):
        contacts = 0
        where = {}
        for i, p in enumerate(chain):
            if sequence[i] == "H":
                where[p] = i
        for p, i in where.items():
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(i - j) > 1:
                    contacts += 1
        contacts //= 2
        free_h = 0
        for p, i in where.items():
            for q in _neighbours(p):
                if q not in occupied:
                    free_h += 1
        return (contacts, -free_h)

    beams = []
    for d in _DELTAS:
        p0 = (0, 0)
        p1 = (d[0], d[1])
        chain = [p0, p1]
        occ = {p0, p1}
        beams.append((_score(chain, occ), chain, occ, d))
    beams.sort(key=lambda x: (x[0][0], x[0][1]), reverse=True)
    beam = beams[:beam_width]

    for step in range(2, n):
        if _remaining_time() < 2.0:
            break
        nxt = []
        for score, chain, occ, last_dir in beam:
            cur = chain[-1]
            opp = (-last_dir[0], -last_dir[1])
            for d in _DELTAS:
                if d == opp:
                    continue
                np_ = (cur[0] + d[0], cur[1] + d[1])
                if np_ in occ:
                    continue
                nchain = chain + [np_]
                nocc = occ | {np_}
                nxt.append((_score(nchain, nocc), nchain, nocc, d))
        if not nxt:
            break
        nxt.sort(key=lambda x: (x[0][0], x[0][1]), reverse=True)
        beam = nxt[:beam_width]
    if not beam:
        return None
    return beam[0][1]


# ---------------- Polish (systematic) + SA, using incremental energy ----------------

def _polish(sequence, chain, time_limit, rng):
    n = len(chain)
    if n < 4:
        return chain
    _, hmap, cmap = _build_contact(sequence, chain)
    best = -len(cmap)
    deadline = time.time() + time_limit
    ops = 0

    def try_reversal(a, b):
        nonlocal hmap, cmap, best, ops
        r = _move_energy_delta(sequence, chain, a, b, hmap, cmap)
        if r is None:
            return False
        delta, new_cmap = r
        # apply reversal to chain
        seg = chain[a:b + 1]
        seg.reverse()
        chain[a:b + 1] = seg
        # update hmap for moved H positions
        for k in range(a, b + 1):
            idx = a + (b - k)
            if sequence[idx] == "H":
                hmap[chain[idx]] = idx
        cmap = new_cmap
        ops += 1
        if -len(cmap) < best:
            best = -len(cmap)
            return True
        # revert
        seg.reverse()
        chain[a:b + 1] = seg
        for k in range(a, b + 1):
            idx = k
            if sequence[idx] == "H":
                hmap[chain[idx]] = idx
        # revert cmap
        r2 = _move_energy_delta(sequence, chain, a, b, hmap, cmap)
        if r2 is not None:
            cmap = r2[1]
        return False

    # systematic short reversals
    max_seg = max(2, int(n * 0.3))
    while time.time() < deadline:
        improved = False
        for a in range(0, n - 1):
            if time.time() > deadline:
                break
            for b in range(a + 1, min(n - 1, a + max_seg) + 1):
                if try_reversal(a, b):
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    # SA with incremental energy
    T = 1.0
    cooling = 0.997
    min_T = 0.05
    stall = 0
    while time.time() < deadline and T > min_T:
        if _remaining_time() < 1.0:
            break
        r = rng.random()
        if r < 0.7:
            a = rng.randint(0, n - 2)
            b = rng.randint(a + 1, min(n - 1, a + max_seg))
            res = _move_energy_delta(sequence, chain, a, b, hmap, cmap)
            if res is None:
                continue
            delta, new_cmap = res
            if delta < 0 or rng.random() < math.exp(-delta / max(T, 0.01)):
                seg = chain[a:b + 1]
                seg.reverse()
                chain[a:b + 1] = seg
                for k in range(a, b + 1):
                    idx = a + (b - k)
                    if sequence[idx] == "H":
                        hmap[chain[idx]] = idx
                cmap = new_cmap
                stall = 0
                if -len(cmap) < best:
                    best = -len(cmap)
            else:
                stall += 1
        else:
            i = rng.randint(1, n - 2)
            angle = rng.choice([1, 2, 3])
            res = _pivot_energy_delta(sequence, chain, i, angle, hmap, cmap)
            if res is None:
                continue
            delta, new_hmap, new_cmap, rotated = res
            if delta < 0 or rng.random() < math.exp(-delta / max(T, 0.01)):
                chain[i + 1:] = rotated
                hmap = new_hmap
                cmap = new_cmap
                stall = 0
                if -len(cmap) < best:
                    best = -len(cmap)
            else:
                stall += 1
        T *= cooling
        if stall > 60:
            # kick
            a = rng.randint(0, n - 2)
            b = rng.randint(a + 1, n - 1)
            res = _move_energy_delta(sequence, chain, a, b, hmap, cmap)
            if res is not None:
                delta, new_cmap = res
                seg = chain[a:b + 1]
                seg.reverse()
                chain[a:b + 1] = seg
                for k in range(a, b + 1):
                    idx = a + (b - k)
                    if sequence[idx] == "H":
                        hmap[chain[idx]] = idx
                cmap = new_cmap
                if -len(cmap) < best:
                    best = -len(cmap)
            stall = 0
            T = max(T, 0.5)
    return chain


def fold(sequence, dimension):
    rng = random.Random(12345 + len(sequence))
    n = len(sequence)
    best, best_energy = None, 0

    # Phase 1: beam search for a strong initial conformation
    t0 = time.time()
    chain = _beam_search(sequence, beam_width=60)
    if chain is None:
        chain = _beam_search(sequence, beam_width=30)
    if chain is None:
        chain = [(i, 0) for i in range(n)]
    best = chain[:]
    best_energy = _energy(sequence, best)

    # Phase 2: polish
    if _remaining_time() > 1.0:
        ls_time = min(_remaining_time() * 0.9, 150.0)
        _polish(sequence, best, ls_time, rng)
        best_energy = _energy(sequence, best)

    # Phase 3: a couple of beam restarts if time remains, polish each
    while _remaining_time() > 20.0:
        cand = _beam_search(sequence, beam_width=40)
        if cand is None:
            break
        _polish(sequence, cand, min(_remaining_time() * 0.5, 60.0), rng)
        ce = _energy(sequence, cand)
        if ce < best_energy:
            best, best_energy = cand, ce
        if _remaining_time() < 20.0:
            break

    if best is None:
        best = [(i, 0) for i in range(n)]
    if not _is_valid_chain(best):
        best = [(i, 0) for i in range(n)]
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
