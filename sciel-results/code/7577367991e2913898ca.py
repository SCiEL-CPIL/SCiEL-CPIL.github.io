"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

_DEADLINE = time.time() + 325  # safety margin before 330s kill


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Energy = -(# of H-H non-adjacent lattice contacts). Only H residues are scanned."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts,
    then free space, with a small penalty for sites that have very little free
    space (avoids dead-ends that force early failure)."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            # Slight penalty if the site is nearly enclosed (free == 1 means a
            # cul-de-sac), which helps avoid chains that trap themselves.
            trap_penalty = -0.3 if free == 1 else 0.0
            options.append((gain + 0.1 * free + trap_penalty + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _is_valid_move(chain, i, new_site, dimension):
    """Check that moving residue i to new_site keeps the chain self-avoiding and connected."""
    if new_site in chain:
        return False
    # Connectivity check around the moved residue
    prev = chain[i - 1]
    nxt = chain[i + 1]
    d_prev = sum(abs(new_site[a] - prev[a]) for a in range(dimension))
    d_next = sum(abs(new_site[a] - nxt[a]) for a in range(dimension))
    if d_prev != 1 or d_next != 1:
        return False
    return True


def _is_valid_kink(chain, i, site1, site2, dimension):
    """Check that a 2-residue kink (i -> site1, i+1 -> site2) is valid."""
    if site1 in chain or site2 in chain:
        return False
    if site1 == site2:
        return False
    # Connectivity
    prev = chain[i - 1]
    nxt = chain[i + 2] if i + 2 < len(chain) else None
    d_prev = sum(abs(site1[a] - prev[a]) for a in range(dimension))
    d_mid = sum(abs(site2[a] - site1[a]) for a in range(dimension))
    if d_prev != 1 or d_mid != 1:
        return False
    if nxt is not None:
        d_next = sum(abs(site2[a] - nxt[a]) for a in range(dimension))
        if d_next != 1:
            return False
    return True


def _local_refine(sequence, chain, dimension, rng, max_steps=30):
    """Local search with 1-residue moves, 2-residue kink moves, and occasional
    simulated-annealing-style neutral/worse moves to escape shallow basins."""
    current_energy = _energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    n = len(chain)
    if n < 3:
        return best_chain

    stall_count = 0
    for step in range(max_steps):
        if time.time() > _DEADLINE:
            break

        moved = False

        # --- 1-residue move at a random internal position ---
        i = rng.randint(1, n - 2)
        for site in _neighbours(chain[i]):
            if site in chain:
                continue
            if not _is_valid_move(chain, i, site, dimension):
                continue
            new_chain = chain[:]
            new_chain[i] = site
            new_energy = _energy(sequence, new_chain)
            if new_energy < best_energy:
                best_chain = new_chain
                best_energy = new_energy
                chain = new_chain
                moved = True
                stall_count = 0
                break
            elif new_energy == best_energy and rng.random() < 0.3:
                # Accept neutral move with some probability to wander
                chain = new_chain
                moved = True
                stall_count = 0
                break

        # --- 2-residue kink move ---
        if not moved and n >= 4:
            i = rng.randint(1, n - 3)
            for site1 in _neighbours(chain[i]):
                if site1 in chain:
                    continue
                for site2 in _neighbours(site1):
                    if site2 in chain or site2 == site1:
                        continue
                    if not _is_valid_kink(chain, i, site1, site2, dimension):
                        continue
                    new_chain = chain[:]
                    new_chain[i] = site1
                    new_chain[i + 1] = site2
                    new_energy = _energy(sequence, new_chain)
                    if new_energy < best_energy:
                        best_chain = new_chain
                        best_energy = new_energy
                        chain = new_chain
                        moved = True
                        stall_count = 0
                        break
                    elif new_energy == best_energy and rng.random() < 0.2:
                        chain = new_chain
                        moved = True
                        stall_count = 0
                        break
                if moved:
                    break

        if not moved:
            # Simulated-annealing style: try a few random 1-moves even if they
            # worsen energy, to escape the local basin.
            escaped = False
            for _ in range(3):
                j = rng.randint(1, n - 2)
                for site in _neighbours(chain[j]):
                    if site in chain:
                        continue
                    if not _is_valid_move(chain, j, site, dimension):
                        continue
                    new_chain = chain[:]
                    new_chain[j] = site
                    new_energy = _energy(sequence, new_chain)
                    # Accept if not too much worse (within 2 contacts)
                    if new_energy <= best_energy + 2:
                        chain = new_chain
                        escaped = True
                        break
                if escaped:
                    break
            if escaped:
                stall_count += 1
                if stall_count >= 5:
                    break
            else:
                break

    return best_chain


def fold(sequence, dimension):
    """Best conformation using many randomized greedy growths with local refinement
    including kink moves and SA-style escape."""
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))

    best_chain = None
    best_energy = 1

    # Scale number of attempts based on chain length
    if n <= 48:
        num_attempts = 500
    elif n <= 80:
        num_attempts = 300
    else:
        num_attempts = 150

    for attempt in range(num_attempts):
        if time.time() > _DEADLINE:
            break

        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue

        value = _energy(sequence, chain)

        if value < best_energy:
            best_chain = chain
            best_energy = value

        # Do refinement on promising candidates (within 2 contacts of best)
        if value <= best_energy + 2:
            refined = _local_refine(sequence, chain, dimension, rng, max_steps=30)
            refined_energy = _energy(sequence, refined)
            if refined_energy < best_energy:
                best_chain = refined
                best_energy = refined_energy

    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
