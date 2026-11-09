"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
from math import exp

_START_TIME = time.time()
_DEADLINE = 205.0  # Safety margin before 210s kill


def _remaining_time():
    return max(0.0, _DEADLINE - (time.time() - _START_TIME))


def _energy(sequence, chain):
    """Compute HP energy: -number of H-H contacts (non-adjacent in chain)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            x, y = p[0], p[1]
            for q in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _energy_delta(sequence, chain, where, idx):
    """
    Compute the change in energy if we move residue idx to a new position.
    Returns (delta, new_where) where delta = new_energy - old_energy.
    """
    old_pos = chain[idx]
    # Remove old position from where
    del where[old_pos]
    
    # Count H-H contacts involving idx at old position
    old_contacts = 0
    if sequence[idx] == "H":
        x, y = old_pos[0], old_pos[1]
        for q in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
            j = where.get(q)
            if j is not None and j != idx and sequence[j] == "H":
                # Only count once per pair
                if j < idx:
                    old_contacts += 1
    # Also count contacts where idx is the "second" residue
    if sequence[idx] == "H":
        x, y = old_pos[0], old_pos[1]
        for q in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
            j = where.get(q)
            if j is not None and j != idx and j > idx + 1 and sequence[j] == "H":
                old_contacts += 1
    
    return old_contacts


def _grow(sequence, dimension, rng, bias=1.0):
    """
    One randomized greedy chain growth.
    bias > 1 favors contacts, bias < 1 favors free space.
    """
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        px, py = chain[-1]
        for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
            site = (px + dx, py + dy)
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                sx, sy = site
                for qx, qy in ((sx+1, sy), (sx-1, sy), (sx, sy+1), (sx, sy-1)):
                    q = (qx, qy)
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = 0
            sx, sy = site
            for qx, qy in ((sx+1, sy), (sx-1, sy), (sx, sy+1), (sx, sy-1)):
                if (qx, qy) not in occupied:
                    free += 1
            score = gain * bias + 0.1 * free + rng.random() * 0.5
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _move_endpoint(sequence, chain, where, current_energy, T, rng):
    """
    Attempt to grow or shrink the endpoint.
    Returns (new_chain, new_where, new_energy, accepted).
    """
    n = len(chain)
    if n < 2:
        return chain, where, current_energy, False
    
    # Try growing the last residue to a new adjacent site
    if rng.random() < 0.7:
        # Grow: try to place a new residue at the end
        # But we can't change sequence length, so instead try moving last residue
        px, py = chain[-1]
        candidates = []
        for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
            site = (px + dx, py + dy)
            if site == chain[-2]:
                continue
            if site in where and where[site] != n - 1:
                continue
            candidates.append(site)
        if not candidates:
            return chain, where, current_energy, False
        new_site = rng.choice(candidates)
        old_site = chain[-1]
        del where[old_site]
        where[new_site] = n - 1
        chain[-1] = new_site
        
        # Recompute energy incrementally
        new_energy = _energy(sequence, chain)
        delta_e = new_energy - current_energy
        
        if delta_e <= 0 or rng.random() < exp(-delta_e / T):
            return chain, where, new_energy, True
        else:
            del where[new_site]
            where[old_site] = n - 1
            chain[-1] = old_site
            return chain, where, current_energy, False
    else:
        # Shrink: remove last residue and regrow
        if n <= 3:
            return chain, where, current_energy, False
        last = chain.pop()
        del where[last]
        # Try regrowing
        px, py = chain[-1]
        candidates = []
        for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
            site = (px + dx, py + dy)
            if site in where:
                continue
            candidates.append(site)
        if not candidates:
            # Restore
            chain.append(last)
            where[last] = n - 1
            return chain, where, current_energy, False
        new_site = rng.choice(candidates)
        chain.append(new_site)
        where[new_site] = n - 1
        new_energy = _energy(sequence, chain)
        delta_e = new_energy - current_energy
        if delta_e <= 0 or rng.random() < exp(-delta_e / T):
            return chain, where, new_energy, True
        else:
            # Revert
            chain.pop()
            del where[new_site]
            chain.append(last)
            where[last] = n - 1
            return chain, where, current_energy, False


def _reverse_move(sequence, chain, where, current_energy, T, rng):
    """
    Attempt a segment reversal. Returns (new_chain, new_where, new_energy, accepted).
    """
    n = len(chain)
    if n < 4:
        return chain, where, current_energy, False

    i = rng.randint(0, n - 3)
    j = rng.randint(i + 2, n - 1)

    # Check boundary connectivity after reversal
    # chain[i-1] must be adjacent to chain[j] (new position of what was chain[i])
    if i > 0:
        a, b = chain[i - 1]
        c, d = chain[j]
        if abs(a - c) + abs(b - d) != 1:
            return chain, where, current_energy, False
    # chain[j+1] must be adjacent to chain[i] (new position of what was chain[j])
    if j < n - 1:
        a, b = chain[i]
        c, d = chain[j + 1]
        if abs(a - c) + abs(b - d) != 1:
            return chain, where, current_energy, False

    # Check self-avoidance of the reversed segment
    # The reversed segment is chain[i..j][::-1]
    # We need to ensure no collisions with the rest of the chain
    temp_seg = chain[i:j+1][::-1]
    # Remove old segment from where temporarily
    for k in range(i, j+1):
        del where[chain[k]]
    
    # Check if any point in temp_seg collides with existing where
    for pt in temp_seg:
        if pt in where:
            # Restore
            for k in range(i, j+1):
                where[chain[k]] = k
            return chain, where, current_energy, False
    
    # Apply reversal
    for k in range(i, j+1):
        chain[k] = temp_seg[k - i]
        where[chain[k]] = k
    
    new_energy = _energy(sequence, chain)
    delta_e = new_energy - current_energy

    if delta_e <= 0 or rng.random() < exp(-delta_e / T):
        return chain, where, new_energy, True
    else:
        # Revert
        for k in range(i, j+1):
            chain[k] = temp_seg[k - i]
            where[chain[k]] = k
        return chain, where, current_energy, False


def _pull_move(sequence, chain, where, current_energy, T, rng):
    """
    Attempt a 'pull' move: move a residue to an adjacent empty site.
    Returns (new_chain, new_where, new_energy, accepted).
    """
    n = len(chain)
    if n <= 2:
        return chain, where, current_energy, False

    idx = rng.randint(1, n - 2)
    old_site = chain[idx]
    prev_site = chain[idx - 1]
    next_site = chain[idx + 1]

    candidates = []
    ox, oy = old_site
    for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
        site = (ox + dx, oy + dy)
        # Must be adjacent to prev and next
        px, py = prev_site
        nx, ny = next_site
        if abs(site[0] - px) + abs(site[1] - py) != 1:
            continue
        if abs(site[0] - nx) + abs(site[1] - ny) != 1:
            continue
        # Must not collide
        if site in where:
            occ_idx = where[site]
            if occ_idx != idx:
                continue
        candidates.append(site)

    if not candidates:
        return chain, where, current_energy, False

    new_site = rng.choice(candidates)
    if new_site == old_site:
        return chain, where, current_energy, False

    del where[old_site]
    where[new_site] = idx
    chain[idx] = new_site

    new_energy = _energy(sequence, chain)
    delta_e = new_energy - current_energy

    if delta_e <= 0 or rng.random() < exp(-delta_e / T):
        return chain, where, new_energy, True
    else:
        del where[new_site]
        where[old_site] = idx
        chain[idx] = old_site
        return chain, where, current_energy, False


def _anneal(sequence, chain, max_time=10.0, seed=12345):
    """Simulated annealing with multiple move types."""
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy

    T = 8.0
    T_min = 0.001
    cooling = 0.9999
    rng = random.Random(seed)

    start = time.time()
    steps = 0
    while T > T_min and (time.time() - start) < max_time:
        r = rng.random()
        if r < 0.35:
            chain, where, current_energy, accepted = _pull_move(sequence, chain, where, current_energy, T, rng)
        elif r < 0.65:
            chain, where, current_energy, accepted = _reverse_move(sequence, chain, where, current_energy, T, rng)
        else:
            chain, where, current_energy, accepted = _move_endpoint(sequence, chain, where, current_energy, T, rng)

        if accepted and current_energy < best_energy:
            best_energy = current_energy
            best_chain = list(chain)

        T *= cooling
        steps += 1
        if steps % 10000 == 0:
            if _remaining_time() < 3:
                break

    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Fold a sequence using multiple growth strategies and annealing.
    """
    n = len(sequence)
    rng = random.Random(12345 + n)
    best, best_energy = None, 0

    # Phase 1: Generate diverse initial conformations
    initial_chains = []
    growth_budget = 200
    biases = [1.0, 1.5, 2.0, 0.5, 0.8, 1.2, 1.8]
    for trial in range(growth_budget):
        if _remaining_time() < 15:
            break
        bias = biases[trial % len(biases)]
        chain = _grow(sequence, dimension, rng, bias=bias)
        if chain is not None:
            initial_chains.append(chain)

    if not initial_chains:
        # Fallback: simple line
        best = [[i, 0] for i in range(n)]
        return best

    # Sort by initial energy to prioritize better starts
    initial_chains.sort(key=lambda c: _energy(sequence, c))

    # Phase 2: Anneal each initial conformation
    time_per_anneal = max(0.5, _remaining_time() / max(1, len(initial_chains)) * 0.85)

    for idx, chain in enumerate(initial_chains):
        if _remaining_time() < 5:
            break
        # Give more time to earlier (better) chains
        if idx < 3:
            t = time_per_anneal * 2.0
        elif idx < 8:
            t = time_per_anneal * 1.5
        else:
            t = time_per_anneal
        t = min(t, _remaining_time() - 2)
        if t <= 0:
            break
        annealed_chain, annealed_energy = _anneal(sequence, chain, max_time=t, seed=12345 + idx)
        if best is None or annealed_energy < best_energy:
            best, best_energy = annealed_chain, annealed_energy

    if best is None:
        best = [[i, 0] for i in range(n)]
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
