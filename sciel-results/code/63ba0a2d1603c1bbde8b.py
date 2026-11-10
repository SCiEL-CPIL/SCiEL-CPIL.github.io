"""HP lattice protein folding: simulated annealing with chain-reversal moves."""
import random
import time


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _initial_spiral(n):
    """Deterministic self-avoiding serpentine spiral of length n (2D)."""
    if n <= 0:
        return []
    if n == 1:
        return [(0, 0)]
    # Build a serpentine grid that is self-avoiding and compact.
    # Grow a rectangular spiral; fall back to a simple zig-zag if needed.
    chain = [(0, 0)]
    occupied = {(0, 0)}
    # Direction cycle for a spiral: right, up, left, down, right...
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    di = 0
    step_len = 1
    steps_in_leg = 0
    guard = 0
    max_guard = n * 12 + 100
    while len(chain) < n and guard < max_guard:
        guard += 1
        d = dirs[di % 4]
        nx = chain[-1][0] + d[0]
        ny = chain[-1][1] + d[1]
        if (nx, ny) not in occupied:
            chain.append((nx, ny))
            occupied.add((nx, ny))
            steps_in_leg += 1
            if steps_in_leg == step_len:
                steps_in_leg = 0
                di += 1
                if di % 2 == 0:  # after a left or right leg, extend length
                    step_len += 1
        else:
            # blocked: turn
            di += 1
            step_len = 1
            steps_in_leg = 0
    # Fallback: if spiral got stuck before n, append a zig-zag tail.
    if len(chain) < n:
        last = chain[-1]
        # extend downward in a zig-zag that avoids occupied cells
        x, y = last
        for _ in range(n - len(chain)):
            placed = False
            for cand in [(x, y - 1), (x + 1, y - 1), (x - 1, y - 1),
                         (x, y + 1), (x + 1, y + 1), (x - 1, y + 1),
                         (x + 1, y), (x - 1, y)]:
                if cand not in occupied:
                    chain.append(cand)
                    occupied.add(cand)
                    x, y = cand
                    placed = True
                    break
            if not placed:
                # absolute fallback: keep moving down
                y -= 1
                chain.append((x, y))
                occupied.add((x, y))
    return chain


def _compute_energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _is_valid_walk(chain):
    n = len(chain)
    if n == 0:
        return True
    occ = set()
    for i, p in enumerate(chain):
        if p in occ:
            return False
        occ.add(p)
        if i > 0:
            dx = abs(chain[i][0] - chain[i - 1][0])
            dy = abs(chain[i][1] - chain[i - 1][1])
            if dx + dy != 1:
                return False
    return True


def _reverse_move(chain, i, j):
    """Reverse the sub-chain from i to j (inclusive). Returns new chain or None."""
    if j - i < 2:
        return None
    a, b = chain[i - 1], chain[j + 1]
    # For a valid SAW reversal, the endpoints of the reversed segment must be
    # adjacent to the new neighbours (a to old chain[j], b to old chain[i]).
    if _adjacent(a, chain[j]) is False:
        return None
    if _adjacent(b, chain[i]) is False:
        return None
    # The interior of the reversed segment must not touch a or b.
    for k in range(i, j + 1):
        if k == i or k == j:
            continue
        p = chain[k]
        if _adjacent(p, a) or _adjacent(p, b):
            return None
    new_chain = chain[:i] + chain[i:j + 1][::-1] + chain[j + 1:]
    return new_chain


def _adjacent(p, q):
    return abs(p[0] - q[0]) + abs(p[1] - q[1]) == 1


def fold(sequence, dimension):
    """Best conformation via simulated annealing with chain-reversal moves."""
    n = len(sequence)
    if n == 0:
        return []
    deadline = time.time() + 110.0  # safety margin well under the 180s limit

    # Precompute which residues are hydrophobic.
    isH = [c == "H" for c in sequence]

    def energy_of(chain):
        return _compute_energy(sequence, chain)

    # Starting conformation: deterministic spiral.
    start = _initial_spiral(n)
    if not _is_valid_walk(start):
        start = [(i, 0) for i in range(n)]
    best_chain = list(start)
    best_energy = energy_of(best_chain)

    rng = random.Random(20240517 + n)

    def anneal(chain, t0, t1, steps):
        nonlocal best_chain, best_energy
        cur = list(chain)
        cur_e = energy_of(cur)
        t = t0
        cool = (t1 / t0) ** (1.0 / max(1, steps))
        for s in range(steps):
            if time.time() > deadline - 2.0:
                break
            # Choose a move: mostly chain-reversal, occasionally a local nudge.
            if rng.random() < 0.8:
                # Chain reversal on a random sub-interval.
                span = rng.randint(2, n - 2)
                i = rng.randint(0, n - 1 - span)
                j = i + span
                new_chain = _reverse_move(cur, i, j)
                if new_chain is None:
                    continue
                new_e = energy_of(new_chain)
                delta = new_e - cur_e
                if delta <= 0 or rng.random() < _boltzmann(-delta, t):
                    cur, cur_e = new_chain, new_e
                    if cur_e < best_energy:
                        best_energy = cur_e
                        best_chain = list(cur)
            else:
                # Local single-residue move: try moving one interior residue.
                k = rng.randint(1, n - 2)
                base = cur[k]
                for cand in _neighbours(base):
                    if cand in cur:
                        continue
                    # ensure cand is adjacent to both neighbours (it is by construction)
                    new_chain = list(cur)
                    new_chain[k] = cand
                    if not _is_valid_walk(new_chain):
                        continue
                    new_e = energy_of(new_chain)
                    delta = new_e - cur_e
                    if delta <= 0 or rng.random() < _boltzmann(-delta, t):
                        cur, cur_e = new_chain, new_e
                        if cur_e < best_energy:
                            best_energy = cur_e
                            best_chain = list(cur)
                    break
            t *= cool
        return cur, cur_e

    # Adaptive: use larger sequences' budgets proportionally but within deadline.
    # Decide number of restarts based on sequence length.
    if n <= 30:
        restarts = 12
        steps_per = 4000
        t0 = 1.0
        t1 = 0.02
    elif n <= 60:
        restarts = 8
        steps_per = 3000
        t0 = 1.0
        t1 = 0.02
    else:
        restarts = 6
        steps_per = 2500
        t0 = 1.0
        t1 = 0.02

    for r in range(restarts):
        if time.time() > deadline - 2.0:
            break
        # Restart from a fresh spiral perturbed, or from best.
        if r == 0:
            cur = list(best_chain)
        else:
            cur = list(_initial_spiral(n))
            if not _is_valid_walk(cur):
                cur = list(best_chain)
        # Occasionally restart from a randomised version of the best.
        if r > 1 and rng.random() < 0.5:
            cur = list(best_chain)
        anneal(cur, t0, t1, steps_per)
        # Final low-temperature polish on the best.
        if time.time() < deadline - 2.0:
            anneal(list(best_chain), 0.05, 0.005, 1500)

    # Guarantee validity.
    if not _is_valid_walk(best_chain):
        best_chain = [(i, 0) for i in range(n)]

    return [list(p) for p in best_chain]


def _boltzmann(gain, t):
    """P(accept worse) = exp(-delta/t) where delta = new_e - cur_e (negative gain)."""
    import math
    if t <= 1e-9:
        return 0.0
    return math.exp(gain / t)
