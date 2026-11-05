"""HP lattice protein folding: randomized construction + local-search refinement."""
# EVOLVE-BLOCK-START
import random
import time

_START = time.time()
_TOTAL_BUDGET = 190.0  # wall-clock safety margin (evaluator kills at 210s)

# Precomputed neighbour deltas for 2D
_DX = (1, -1, 0, 0)
_DY = (0, 0, 1, -1)


def _neighbours(x, y):
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _energy_from(chain, is_h):
    """Recompute energy from scratch."""
    where = {}
    for i, (x, y) in enumerate(chain):
        where[(x, y)] = i
    contacts = 0
    for i, (x, y) in enumerate(chain):
        if not is_h[i]:
            continue
        for (nx, ny) in _neighbours(x, y):
            j = where.get((nx, ny))
            if j is not None and j > i + 1 and is_h[j]:
                contacts += 1
    return -contacts


def _greedy_grow(sequence, is_h, rng):
    """Randomized greedy chain growth."""
    n = len(sequence)
    cx, cy = 0, 0
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    for i in range(1, n):
        options = []
        for (nx, ny) in _neighbours(cx, cy):
            if (nx, ny) in occupied:
                continue
            gain = 0
            if is_h[i]:
                for (gx, gy) in _neighbours(nx, ny):
                    j = occupied.get((gx, gy))
                    if j is not None and j < i - 1 and is_h[j]:
                        gain += 1
            free = 0
            for (gx, gy) in _neighbours(nx, ny):
                if (gx, gy) not in occupied:
                    free += 1
            score = gain * 2.0 + 0.1 * free + rng.random()
            options.append((score, nx, ny))
        if not options:
            return None
        options.sort(key=lambda t: -t[0])
        if len(options) > 1 and rng.random() < 0.2:
            _, nx, ny = options[1]
        else:
            _, nx, ny = options[0]
        occupied[(nx, ny)] = i
        chain.append((nx, ny))
        cx, cy = nx, ny
    return chain


def _local_search(sequence, is_h, chain, rng, time_limit):
    """Local search with pivot and single-residue moves."""
    n = len(chain)
    if n < 3:
        return chain, 0

    occupied = {}
    for i, (x, y) in enumerate(chain):
        occupied[(x, y)] = i

    def build_contacts():
        contacts = set()
        for i in range(n):
            if not is_h[i]:
                continue
            x, y = chain[i]
            for (nx, ny) in _neighbours(x, y):
                j = occupied.get((nx, ny))
                if j is not None and j > i + 1 and is_h[j]:
                    contacts.add((i, j))
        return contacts

    contacts = build_contacts()
    energy = -len(contacts)
    best_chain = list(chain)
    best_energy = energy

    T0 = 1.5
    Tend = 0.05
    steps = 60000

    import math

    for step in range(steps):
        if time.time() > time_limit:
            break

        frac = step / steps
        T = T0 * (Tend / T0) ** frac
        if T < 0.01:
            T = 0.01

        r = rng.random()
        if r < 0.55:
            # Pivot move
            i = rng.randrange(n - 1)
            j = rng.randrange(i + 1, n)
            if j - i < 1:
                continue
            ax, ay = chain[i]
            # Save original segment positions
            orig_seg = chain[i + 1:j + 1]
            for rot in range(1, 4):
                new_seg = []
                valid = True
                for k in range(i + 1, j + 1):
                    ox, oy = chain[k]
                    dx, dy = ox - ax, oy - ay
                    if rot == 1:
                        ndx, ndy = -dy, dx
                    elif rot == 2:
                        ndx, ndy = -dx, -dy
                    else:
                        ndx, ndy = dy, -dx
                    np_ = (ax + ndx, ay + ndy)
                    if np_ in occupied:
                        orig_idx = occupied[np_]
                        if orig_idx < i or orig_idx > j:
                            valid = False
                            break
                    new_seg.append(np_)
                if not valid:
                    continue
                # Check consecutive unit steps
                ok = True
                prev = (ax, ay)
                for np_ in new_seg:
                    dx = np_[0] - prev[0]
                    dy = np_[1] - prev[1]
                    if abs(dx) + abs(dy) != 1:
                        ok = False
                        break
                    prev = np_
                if not ok:
                    continue
                # Check junction to chain[j+1]
                if j < n - 1:
                    last_new = new_seg[-1]
                    nx, ny = chain[j + 1]
                    if abs(last_new[0] - nx) + abs(last_new[1] - ny) != 1:
                        continue
                # Apply
                for idx, np_ in zip(range(i + 1, j + 1), new_seg):
                    old = chain[idx]
                    del occupied[old]
                    chain[idx] = np_
                    occupied[np_] = idx
                new_contacts = build_contacts()
                new_energy = -len(new_contacts)
                delta = new_energy - energy
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    energy = new_energy
                    contacts = new_contacts
                    if energy < best_energy:
                        best_energy = energy
                        best_chain = list(chain)
                else:
                    # Revert
                    for idx, orig in zip(range(i + 1, j + 1), orig_seg):
                        del occupied[chain[idx]]
                        chain[idx] = orig
                        occupied[orig] = idx
                break  # Only try one rotation per pivot attempt
        else:
            # Single-residue move
            k = rng.randrange(1, n - 1)
            x, y = chain[k]
            tried = False
            for (nx, ny) in _neighbours(x, y):
                if (nx, ny) in occupied:
                    continue
                px, py = chain[k - 1]
                qx, qy = chain[k + 1]
                if abs(nx - px) + abs(ny - py) != 1:
                    continue
                if abs(nx - qx) + abs(ny - qy) != 1:
                    continue
                # Apply
                del occupied[(x, y)]
                chain[k] = (nx, ny)
                occupied[(nx, ny)] = k
                new_contacts = build_contacts()
                new_energy = -len(new_contacts)
                delta = new_energy - energy
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    energy = new_energy
                    contacts = new_contacts
                    tried = True
                    if energy < best_energy:
                        best_energy = energy
                        best_chain = list(chain)
                else:
                    del occupied[(nx, ny)]
                    chain[k] = (x, y)
                    occupied[(x, y)] = k
                if tried:
                    break
            continue

    return best_chain, best_energy


def fold(sequence, dimension):
    """Fold a sequence into a low-energy conformation."""
    n = len(sequence)
    is_h = [c == 'H' for c in sequence]
    rng = random.Random(42 + n)

    elapsed = time.time() - _START
    remaining = _TOTAL_BUDGET - elapsed
    if remaining < 5:
        return [[i, 0] for i in range(n)]

    # Allocate: 50% construction, 50% local search
    constr_budget = remaining * 0.5
    ls_budget = remaining * 0.5

    best_chain = None
    best_energy = 0

    # Phase 1: Greedy constructions
    constr_deadline = time.time() + constr_budget
    while time.time() < constr_deadline:
        chain = _greedy_grow(sequence, is_h, rng)
        if chain is None:
            continue
        e = _energy_from(chain, is_h)
        if best_chain is None or e < best_energy:
            best_chain = chain
            best_energy = e

    if best_chain is None:
        return [[i, 0] for i in range(n)]

    # Phase 2: Local search
    ls_deadline = time.time() + ls_budget
    refined, e = _local_search(sequence, is_h, list(best_chain), rng, ls_deadline)
    if e < best_energy:
        best_chain = refined
        best_energy = e

    # Second refinement if time allows
    if time.time() < ls_deadline - 3:
        refined2, e2 = _local_search(sequence, is_h, list(best_chain), rng, ls_deadline)
        if e2 < best_energy:
            best_chain = refined2
            best_energy = e2

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
