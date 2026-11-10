"""HP lattice protein folding on the square lattice – global stochastic local search with pivot moves."""
# EVOLVE-BLOCK-START
import random
import time
import math

_DEADLINE = time.time() + 175.0


def _neighbours(point):
    """Yield the 4 unit-step neighbours of a 2D point (tuple)."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _compute_energy(sequence, chain):
    """Compute the negative number of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2


def _generate_random_saw(n, rng):
    """Generate a random self-avoiding walk of length n on the square lattice."""
    if n <= 0:
        return []
    chain = [(0, 0)]
    occupied = {(0, 0)}
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for _ in range(1, n):
        x, y = chain[-1]
        rng.shuffle(dirs)
        moved = False
        for dx, dy in dirs:
            np = (x + dx, y + dy)
            if np not in occupied:
                chain.append(np)
                occupied.add(np)
                moved = True
                break
        if not moved:
            return None
    return chain


def _apply_pivot(chain, i, rng):
    """
    Apply a pivot move at position i: rotate the suffix chain[i:] by 0, 90, 180, or 270 degrees
    about the point chain[i-1] (the attachment point).
    Returns the new chain or None if the move creates a self-intersection.
    """
    n = len(chain)
    if i <= 0 or i >= n:
        return None

    # Pivot point is chain[i-1]
    px, py = chain[i - 1]

    # Rotation matrices for 90, 180, 270 degrees
    angle = rng.choice([1, 2, 3])  # 90, 180, 270
    if angle == 1:   # 90 deg CCW: (x,y) -> (-y, x)
        rot = lambda x, y: (-y, x)
    elif angle == 2:  # 180 deg: (x,y) -> (-x, -y)
        rot = lambda x, y: (-x, -y)
    else:             # 270 deg CCW: (x,y) -> (y, -x)
        rot = lambda x, y: (y, -x)

    # Rotate suffix
    new_suffix = []
    for j in range(i, n):
        x, y = chain[j]
        rx, ry = rot(x - px, y - py)
        new_suffix.append((px + rx, py + ry))

    new_chain = chain[:i] + new_suffix

    # Check self-avoidance
    occupied = set()
    for p in new_chain:
        if p in occupied:
            return None
        occupied.add(p)

    return new_chain


def _regrow_suffix(sequence, chain, start_idx, rng, temperature=0.0):
    """
    Regrow the suffix starting at index start_idx (inclusive) from chain[start_idx-1].
    Returns the new suffix (list of tuples) or None if stuck.
    """
    n = len(sequence)
    if start_idx >= n:
        return []

    prefix = chain[:start_idx]
    occupied = {p: i for i, p in enumerate(prefix)}
    last = prefix[-1]
    new_suffix = []

    for i in range(start_idx, n):
        res = sequence[i]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if res == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 10.0 + free * 0.1 + rng.random() * temperature
            options.append((score, site))

        if not options:
            return None

        best_score, best_site = max(options, key=lambda x: x[0])
        occupied[best_site] = i
        new_suffix.append(best_site)
        last = best_site

    return new_suffix


def _local_improve_pivot(sequence, chain, rng, max_moves=3000, start_temp=2.0, cool_rate=0.998):
    """
    Stochastic local search using pivot moves and regrowth.
    """
    n = len(chain)
    if n < 4:
        return chain, _compute_energy(sequence, chain)

    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    temp = start_temp
    no_improve = 0

    for it in range(max_moves):
        if time.time() > _DEADLINE - 15:
            break
        if temp < 0.03:
            break

        # Choose move type: pivot (70%) or regrow suffix (30%)
        move_type = rng.random()
        new_chain = None

        if move_type < 0.7:
            # Pivot move
            i = rng.randint(1, n - 1)
            new_chain = _apply_pivot(chain, i, rng)
            if new_chain is None:
                # Try regrowing the suffix after pivot failure
                # Find where the conflict started
                # For simplicity, just skip this move
                temp *= cool_rate
                no_improve += 1
                continue
        else:
            # Regrow suffix from a random point
            min_split = 3
            max_split = max(min_split, n - 2)
            if max_split <= min_split:
                break
            split = rng.randint(min_split, max_split)
            suffix = _regrow_suffix(sequence, chain, split, rng, temperature=temp)
            if suffix is None:
                temp *= cool_rate
                no_improve += 1
                continue
            new_chain = chain[:split] + suffix

        new_energy = _compute_energy(sequence, new_chain)
        delta = new_energy - current_energy

        if delta < 0 or (temp > 0 and rng.random() < (10 ** (-delta / max(temp, 0.1)))):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        temp *= cool_rate

        if no_improve > 400:
            break

    return best_chain, best_energy


def _greedy_grow(sequence, dimension, rng, temperature=0.0):
    """
    Randomized greedy chain growth for initial solutions.
    """
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)

    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 10.0 + free * 0.1 + rng.random() * temperature
            options.append((score, site))

        if not options:
            return None

        best_score, best_site = max(options, key=lambda x: x[0])
        occupied[best_site] = i
        chain.append(best_site)

    return chain


def _h_h_contact_repair(sequence, chain, rng, max_moves=2000):
    """
    Targeted repair: identify H residues that lack contacts and try to move them
    (via local pivots) closer to other H residues.
    """
    n = len(chain)
    if n < 4:
        return chain, _compute_energy(sequence, chain)

    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    no_improve = 0

    for _ in range(max_moves):
        if time.time() > _DEADLINE - 10:
            break

        # Find H residues with fewest contacts
        where = {p: i for i, p in enumerate(chain)}
        h_indices = [i for i in range(n) if sequence[i] == 'H']
        if not h_indices:
            break

        # Pick a random H residue
        i = rng.choice(h_indices)
        p = chain[i]

        # Try to find a nearby H residue to contact
        target_h = None
        for j in h_indices:
            if abs(j - i) > 1 and abs(j - i) < n // 2:
                tp = chain[j]
                dist = abs(tp[0] - p[0]) + abs(tp[1] - p[1])
                if dist <= 4:
                    target_h = j
                    break

        if target_h is None:
            no_improve += 1
            if no_improve > 200:
                break
            continue

        # Try pivot moves near this residue
        for pivot_i in range(max(1, i - 3), min(n - 1, i + 4)):
            new_chain = _apply_pivot(chain, pivot_i, rng)
            if new_chain is None:
                continue

            new_energy = _compute_energy(sequence, new_chain)
            if new_energy < current_energy:
                chain = new_chain
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
                no_improve = 0
                break
        else:
            no_improve += 1

        if no_improve > 300:
            break

    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Fold the sequence using a global stochastic local search with pivot moves,
    combined with greedy initial solutions and targeted H-H contact repair.
    """
    rng = random.Random(42)
    best_chain = None
    best_energy = 0  # We want to minimize energy (more negative is better)

    n = len(sequence)

    # Phase 1: Generate diverse initial solutions via greedy growth and random SAWs
    phase1_end = time.time() + 40.0
    for attempt in range(2000):
        if time.time() > phase1_end:
            break

        # Alternate between greedy growth and random SAW + regrow
        if attempt % 3 == 0:
            temp = rng.uniform(0.0, 2.0)
            chain = _greedy_grow(sequence, dimension, rng, temperature=temp)
        else:
            # Generate random SAW and then regrow with greedy
            random_saw = _generate_random_saw(n, rng)
            if random_saw is None:
                continue
            # Regrow the last 30% with greedy
            split = max(3, int(n * 0.7))
            suffix = _regrow_suffix(sequence, random_saw, split, rng, temperature=0.5)
            if suffix is None:
                continue
            chain = random_saw[:split] + suffix

        if chain is None:
            continue

        energy = _compute_energy(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy

    # Phase 2: Global stochastic local search with pivot moves
    if best_chain is not None:
        phase2_end = time.time() + 60.0
        while time.time() < phase2_end:
            sa_chain, sa_energy = _local_improve_pivot(
                sequence, best_chain, rng, max_moves=2000, start_temp=2.0, cool_rate=0.998)
            if sa_energy < best_energy:
                best_chain = sa_chain
                best_energy = sa_energy
            else:
                # Perturb and try again
                if time.time() > phase2_end - 10:
                    break
                # Apply a random pivot to escape local minimum
                i = rng.randint(1, len(best_chain) - 1)
                perturbed = _apply_pivot(best_chain, i, rng)
                if perturbed is not None:
                    best_chain = perturbed

    # Phase 3: Targeted H-H contact repair
    if best_chain is not None:
        phase3_end = time.time() + 40.0
        while time.time() < phase3_end:
            repaired_chain, repaired_energy = _h_h_contact_repair(
                sequence, best_chain, rng, max_moves=1500)
            if repaired_energy < best_energy:
                best_chain = repaired_chain
                best_energy = repaired_energy
            else:
                if time.time() > phase3_end - 10:
                    break
                # Perturb
                i = rng.randint(1, len(best_chain) - 1)
                perturbed = _apply_pivot(best_chain, i, rng)
                if perturbed is not None:
                    best_chain = perturbed

    # Fallback: straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
