"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

_DEADLINE = time.time() + 175.0


def _neighbours2d(point):
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    n = len(sequence)
    for i in range(1, n):
        options = []
        last = chain[-1]
        for site in _neighbours2d(last):
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
                if q not in occupied and q != last:
                    free += 1
            score = gain * 2.5 + 0.15 * free + rng.random() * 0.5
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
            for q in _neighbours2d(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _contact_count_at(sequence, occupied, site, exclude_i=None):
    """Count H-H contacts a residue at `site` would have with already-placed residues.
    exclude_i: if set, skip the residue at index exclude_i (for moves where we're relocating)."""
    count = 0
    for q in _neighbours2d(site):
        j = occupied.get(q)
        if j is not None and j != exclude_i and sequence[j] == "H":
            count += 1
    return count


def _local_optimize(sequence, chain, occupied, max_iters=3000):
    """Local optimization with single-residue and kink moves using incremental energy deltas."""
    n = len(chain)
    if n < 4:
        return chain, occupied, _energy(sequence, chain)

    best_energy = _energy(sequence, chain)
    current_energy = best_energy
    best_chain = list(chain)
    best_occupied = dict(occupied)

    # Simulated annealing parameters
    T = max(0.5, n * 0.04)
    T_min = 0.01
    cooling = 0.998

    iters = 0
    while iters < max_iters and time.time() < _DEADLINE:
        iters += 1
        move_type = random.random()

        if move_type < 0.6:
            # Single-residue move
            i = random.randint(1, n - 2)
            x, y = chain[i]
            new_site = None
            for nx, ny in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
                if (nx, ny) != (x, y):
                    new_site = (nx, ny)
                    break
            if new_site is None:
                continue

            # Check validity
            if new_site in occupied and occupied[new_site] != i:
                continue
            # Connectivity check
            if i > 0:
                px, py = chain[i-1]
                if abs(new_site[0] - px) + abs(new_site[1] - py) != 1:
                    continue
            if i < n - 1:
                px, py = chain[i+1]
                if abs(new_site[0] - px) + abs(new_site[1] - py) != 1:
                    continue

            # Compute energy delta incrementally
            old_site = chain[i]
            delta = 0
            if sequence[i] == "H":
                # Remove old contacts
                for q in _neighbours2d(old_site):
                    j = occupied.get(q)
                    if j is not None and j != i and sequence[j] == "H":
                        delta += 1  # losing a contact increases energy
                # Add new contacts
                for q in _neighbours2d(new_site):
                    j = occupied.get(q)
                    if j is not None and j != i and sequence[j] == "H":
                        delta -= 1  # gaining a contact decreases energy

            # Acceptance
            if delta < 0 or random.random() < math.exp(-delta / T):
                # Apply move
                del occupied[old_site]
                occupied[new_site] = i
                chain[i] = new_site
                current_energy += delta
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
                    best_occupied = dict(occupied)
        else:
            # Kink move: rotate consecutive pair (i, i+1)
            i = random.randint(1, n - 3)
            p1, p2 = chain[i], chain[i+1]
            x1, y1 = p1
            x2, y2 = p2
            dx, dy = x2 - x1, y2 - y1

            # Generate perpendicular rotation candidates
            candidates = []
            if dx == 1 or dx == -1:
                candidates = [(x1, y1+1), (x1, y1-1)]
                new_p2s = [(x2, y2+1), (x2, y2-1)]
            elif dy == 1 or dy == -1:
                candidates = [(x1+1, y1), (x1-1, y1)]
                new_p2s = [(x2+1, y2), (x2-1, y2)]
            else:
                continue

            for sign in (1, -1):
                if dx == 1 or dx == -1:
                    if sign == 1:
                        new_p1 = (x1, y1+1)
                        new_p2 = (x2, y2+1)
                    else:
                        new_p1 = (x1, y1-1)
                        new_p2 = (x2, y2-1)
                else:
                    if sign == 1:
                        new_p1 = (x1+1, y1)
                        new_p2 = (x2+1, y2)
                    else:
                        new_p1 = (x1-1, y1)
                        new_p2 = (x2-1, y2)

                # Check validity for both residues
                if new_p1 in occupied and occupied[new_p1] not in (i, i+1):
                    continue
                if new_p2 in occupied and occupied[new_p2] not in (i, i+1):
                    continue
                # Connectivity: residue i-1 to new_p1
                if i > 0:
                    px, py = chain[i-1]
                    if abs(new_p1[0] - px) + abs(new_p1[1] - py) != 1:
                        continue
                # Connectivity: new_p2 to residue i+2
                if i + 2 < n:
                    px, py = chain[i+2]
                    if abs(new_p2[0] - px) + abs(new_p2[1] - py) != 1:
                        continue
                # new_p1 and new_p2 must be adjacent
                if abs(new_p1[0] - new_p2[0]) + abs(new_p1[1] - new_p2[1]) != 1:
                    continue

                # Compute energy delta for both residues
                delta = 0
                for idx, old_s, new_s in ((i, p1, new_p1), (i+1, p2, new_p2)):
                    if sequence[idx] == "H":
                        for q in _neighbours2d(old_s):
                            j = occupied.get(q)
                            if j is not None and j not in (i, i+1) and sequence[j] == "H":
                                delta += 1
                        for q in _neighbours2d(new_s):
                            j = occupied.get(q)
                            if j is not None and j not in (i, i+1) and sequence[j] == "H":
                                delta -= 1

                if delta < 0 or random.random() < math.exp(-delta / T):
                    # Apply kink move
                    del occupied[p1]
                    del occupied[p2]
                    occupied[new_p1] = i
                    occupied[new_p2] = i + 1
                    chain[i] = new_p1
                    chain[i+1] = new_p2
                    current_energy += delta
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                        best_occupied = dict(occupied)
                    break  # try next kink position

        # Cooling
        T = max(T_min, T * cooling)

    return best_chain, best_occupied, best_energy


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
        # Build occupied set
        occupied = {p: i for i, p in enumerate(chain)}
        # Local optimization
        chain, occupied, value = _local_optimize(sequence, chain, occupied)
        if value < best_energy:
            best, best_energy = chain, value
        # If we've done enough restarts and no improvement recently, stop
        if restarts > 800:
            break
    if best is None:
        best = [(i, 0) for i in range(n)]
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
