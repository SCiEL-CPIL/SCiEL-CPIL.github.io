"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

_DEADLINE = time.time() + 170.0


def _neighbours2d(point):
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    n = len(sequence)
    for i in range(1, n):
        options = []
        for site in _neighbours2d(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours2d(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = 0
            for q in _neighbours2d(site):
                if q not in occupied:
                    free += 1
            # small temperature that rises near the end to avoid dead-ends
            t = 0.3 if i < n - 5 else 1.0
            score = gain * 2.0 + 0.1 * free + t * rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            x, y = p
            for q in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _local_optimize(sequence, chain, max_iters=2000):
    """Fast local search: single-residue moves (incremental energy) then 2-opt segment reversals."""
    n = len(chain)
    if n < 3:
        return chain, _energy(sequence, chain)
    occupied = {p: i for i, p in enumerate(chain)}
    best_energy = _energy(sequence, chain)

    # ---- Phase 1: single-residue moves with incremental energy ----
    improved = True
    iters = 0
    while improved and iters < max_iters and time.time() < _DEADLINE:
        improved = False
        iters += 1
        for i in range(1, n - 1):
            if time.time() >= _DEADLINE:
                return chain, best_energy
            x, y = chain[i]
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                site = (nx, ny)
                if site == chain[i]:
                    continue
                if site in occupied and occupied[site] != i:
                    continue
                # consecutive unit-step checks
                px, py = chain[i - 1]
                if abs(nx - px) + abs(ny - py) != 1:
                    continue
                px, py = chain[i + 1]
                if abs(nx - px) + abs(ny - py) != 1:
                    continue
                # incremental energy delta: only residues within 2 of i can change contact status
                delta = 0
                for k in range(max(0, i - 2), min(n, i + 3)):
                    if k == i:
                        continue
                    if sequence[k] != "H":
                        continue
                    # old position of k
                    kx, ky = chain[k]
                    for q in ((kx + 1, ky), (kx - 1, ky), (kx, ky + 1), (kx, ky - 1)):
                        j = occupied.get(q)
                        if j is not None and j > k + 1 and sequence[j] == "H":
                            # contact exists; does it survive the move?
                            # if q is the old site of i and i is moving away, this contact is lost
                            # if q is the new site of i, it's not a contact (i is at new site now)
                            # We need to check if the contact (k, j) is affected by moving i.
                            # A contact is affected if i is one of the endpoints (k or j) — but k != i here.
                            # Actually, moving i only affects contacts that involve i.
                            # So we should only check contacts involving i.
                            pass
                    # Simpler: recompute contacts involving i only.
                # Recompute: contacts involving i before and after
                # Before: i at (x,y)
                old_contacts = 0
                if sequence[i] == "H":
                    for q in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                        j = occupied.get(q)
                        if j is not None and j > i + 1 and sequence[j] == "H":
                            old_contacts += 1
                # After: i at (nx,ny)
                new_contacts = 0
                if sequence[i] == "H":
                    for q in ((nx + 1, ny), (nx - 1, ny), (nx, ny + 1), (nx, ny - 1)):
                        j = occupied.get(q)
                        if j is not None and j > i + 1 and sequence[j] == "H":
                            # but if j == i, skip (shouldn't happen)
                            if j != i:
                                new_contacts += 1
                # Also, moving i may break/form contacts where i was the "j" (higher index) endpoint.
                # For each neighbor k of i (k < i), if sequence[k]=='H', check if k was adjacent to i's old/new site.
                # Before: was k adjacent to (x,y)?
                if sequence[i] == "H":
                    for k in range(max(0, i - 2), i):
                        if sequence[k] == "H":
                            kx, ky = chain[k]
                            # was k adjacent to old site?
                            if abs(kx - x) + abs(ky - y) == 1:
                                old_contacts += 1  # this contact existed
                            # is k adjacent to new site?
                            if abs(kx - nx) + abs(ky - ny) == 1:
                                new_contacts += 1
                delta = new_contacts - old_contacts
                if delta < 0:  # energy decreases
                    occupied[site] = i
                    del occupied[(x, y)]
                    chain[i] = site
                    best_energy += delta
                    improved = True
                    break

    # ---- Phase 2: 2-opt segment reversals ----
    improved = True
    while improved and time.time() < _DEADLINE:
        improved = False
        for i in range(1, n - 2):
            if time.time() >= _DEADLINE:
                break
            for j in range(i + 1, n - 1):
                if time.time() >= _DEADLINE:
                    break
                # Reverse segment [i+1..j]
                # Check unit steps at boundaries: chain[i] to chain[j], and chain[i+1] to chain[j+1]
                cx, cy = chain[i]
                jx, jy = chain[j]
                if abs(jx - cx) + abs(jy - cy) != 1:
                    continue
                if i + 1 < n - 1:
                    ix, iy = chain[i + 1]
                    jpx, jpy = chain[j + 1]
                    if abs(jpx - ix) + abs(jpy - iy) != 1:
                        continue
                # Check self-avoidance of reversed segment
                # The reversed segment occupies the same set of sites, so internal self-avoidance is preserved.
                # But we must ensure no new collisions: the set of sites is identical, so no collision.
                # Apply reversal
                segment = chain[i + 1:j + 1]
                segment.reverse()
                new_chain = chain[:i + 1] + segment + chain[j + 1:]
                new_energy = _energy(sequence, new_chain)
                if new_energy < best_energy:
                    chain = new_chain
                    best_energy = new_energy
                    improved = True
                    break
            if improved:
                break

    return chain, best_energy


def fold(sequence, dimension):
    """Best conformation with deadline-aware search."""
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 1
    restarts = 0
    while time.time() < _DEADLINE:
        restarts += 1
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        chain, value = _local_optimize(sequence, chain)
        if value < best_energy:
            best, best_energy = chain, value
        if restarts > 2000:
            break

    # Final polish: more local optimization on the best chain
    if best is not None and time.time() < _DEADLINE:
        best, best_energy = _local_optimize(best, max_iters=5000)

    if best is None:
        best = [(i, 0) for i in range(n)]
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
