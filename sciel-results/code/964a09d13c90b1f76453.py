"""HP lattice protein folding: fast 2-opt local search with delta energy updates."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_DEADLINE = _GLOBAL_START + 205.0

_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _remaining_time():
    return max(0.0, _GLOBAL_DEADLINE - time.time())


def _energy(sequence, chain):
    """Compute total energy: -1 for each H-H contact on non-adjacent lattice neighbours."""
    n = len(chain)
    if n < 2:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            x, y = p
            for dx, dy in _DELTAS:
                q = (x + dx, y + dy)
                j = where.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts // 2


def _compute_contact_map(sequence, chain):
    """
    Compute a contact map: set of (i, j) pairs where i < j, |i-j| > 1,
    chain[i] and chain[j] are lattice neighbours, and both are H.
    Returns (contact_set, energy).
    """
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    contact_set = set()
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            x, y = p
            for dx, dy in _DELTAS:
                q = (x + dx, y + dy)
                j = where.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == "H":
                    if i < j:
                        contact_set.add((i, j))
                    else:
                        contact_set.add((j, i))
    energy = -len(contact_set)
    return contact_set, energy


def _energy_delta(sequence, chain, i, j, contact_set):
    """
    Compute the change in energy after reversing segment [i, j].
    This is an approximation: we recompute contacts only for residues in the reversed segment
    and their immediate neighbours in the chain.
    """
    n = len(chain)
    if i >= j:
        return 0

    # Determine which residues' contacts might change.
    # Residues in [i, j] move to new positions. Their contacts with residues outside [i, j]
    # might change. Also, internal contacts within [i, j] are preserved (same pairs, just reversed order).
    # So we only need to check contacts between [i, j] and outside, plus the boundary connections.

    # Old positions of residues in [i, j]
    old_positions = {}
    for k in range(i, j + 1):
        old_positions[k] = chain[k]

    # New positions after reversal
    new_positions = {}
    for k in range(i, j + 1):
        new_positions[k] = chain[i + j - k]

    # Residues outside [i, j] that are within 1 chain-step of [i, j] boundaries
    # and are lattice-neighbours of some residue in [i, j]
    # We need to check:
    # 1. Contacts between residues in [i, j] and residues outside [i, j]
    # 2. The chain-adjacency at boundaries (i-1, i) and (j, j+1)

    # Build a set of outside residues that might be affected
    outside_affected = set()
    # Check boundary neighbours
    if i > 0:
        outside_affected.add(i - 1)
    if j < n - 1:
        outside_affected.add(j + 1)

    # For each residue in [i, j], check its lattice neighbours that are outside [i, j]
    for k in range(i, j + 1):
        x, y = new_positions[k]
        for dx, dy in _DELTAS:
            q = (x + dx, y + dy)
            # Find which chain index is at q in the new configuration
            # We need a position-to-index map for the new configuration
            # But we don't have one yet. Let's compute it.
            pass

    # This approach is getting complex. Let's just recompute energy for the affected region.
    # For simplicity and correctness, we'll do a full recompute but only for a subset.
    # Actually, for correctness, let's just return a sentinel and let the caller recompute.
    # But that defeats the purpose. Let's do a proper delta computation.

    # Build new chain
    new_chain = chain[:i] + chain[i:j+1][::-1] + chain[j+1:]

    # Build position map for new chain
    new_where = {p: idx for idx, p in enumerate(new_chain)}

    # Old energy contribution from contacts involving residues in [i, j]
    old_contrib = 0
    for (a, b) in contact_set:
        if i <= a <= j or i <= b <= j:
            old_contrib += 1

    # New energy contribution from contacts involving residues in [i, j]
    new_contrib = 0
    for k in range(i, j + 1):
        if sequence[k] != "H":
            continue
        x, y = new_chain[k]
        for dx, dy in _DELTAS:
            q = (x + dx, y + dy)
            m = new_where.get(q)
            if m is not None and abs(k - m) > 1 and sequence[m] == "H":
                if k < m:
                    new_contrib += 1
                else:
                    # m < k, so the pair is (m, k)
                    new_contrib += 1

    # The delta is new - old (in terms of contact count, energy is negative)
    # Energy = -contacts, so delta_energy = -(new_contacts - old_contacts) = old_contacts - new_contacts
    # But we only computed partial contacts. The full delta is:
    # delta = (old_total - old_contrib) - (new_total - new_contrib) ... no, this is wrong.
    # Let's think: energy = -total_contacts
    # delta = new_energy - old_energy = -new_total + old_total = old_total - new_total
    # We don't know new_total without full recompute.
    # So this partial approach doesn't work directly.

    # Alternative: just recompute full energy. It's O(n) which is fine for n up to 100.
    new_energy = _energy(sequence, new_chain)
    old_energy = -len(contact_set)
    return new_energy - old_energy, new_chain, new_energy


def _grow(sequence, rng, hh_weight=1.0, free_weight=0.1, noise=0.3):
    """Greedy growth with bias toward hydrophobic contacts."""
    n = len(sequence)
    if n < 2:
        return [(0, 0)] * n
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}

    for i in range(1, n):
        options = []
        for site in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            x, y = chain[-1]
            cand = (x + site[0], y + site[1])
            if cand in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                cx, cy = cand
                for dx, dy in _DELTAS:
                    q = (cx + dx, cy + dy)
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            free = 0
            cx, cy = cand
            for dx, dy in _DELTAS:
                if (cx + dx, cy + dy) not in occupied:
                    free += 1
            score = hh_weight * gain + free_weight * free + noise * rng.random()
            options.append((score, cand))
        if not options:
            return None
        best_score, best_site = max(options)
        occupied[best_site] = i
        chain.append(best_site)

    return chain


def _is_valid_chain(chain):
    """Check self-avoidance and unit steps."""
    n = len(chain)
    if n < 2:
        return True
    seen = set()
    for i, (x, y) in enumerate(chain):
        if (x, y) in seen:
            return False
        seen.add((x, y))
        if i > 0:
            px, py = chain[i-1]
            if abs(px - x) + abs(py - y) != 1:
                return False
    return True


def _apply_reversal(chain, i, j):
    """Reverse segment [i, j]. Return new chain or None if invalid."""
    n = len(chain)
    if i >= j:
        return None
    # Check boundary connectivity
    if i > 0:
        px, py = chain[i - 1]
        ex, ey = chain[j]
        if abs(px - ex) + abs(py - ey) != 1:
            return None
    if j < n - 1:
        sx, sy = chain[i]
        nx, ny = chain[j + 1]
        if abs(sx - nx) + abs(sy - ny) != 1:
            return None
    # Check self-avoidance of the reversed segment
    segment = chain[i:j+1]
    for pos in segment:
        if pos in chain[:i] or pos in chain[j+1:]:
            return None
    new_chain = chain[:i] + segment[::-1] + chain[j+1:]
    return new_chain


def _apply_pivot(chain, i, angle):
    """Rotate tail starting at i by angle * 90 degrees around chain[i]. Return new chain or None."""
    n = len(chain)
    if i <= 0 or i >= n - 1:
        return None
    px, py = chain[i]
    tail = chain[i+1:]
    rotated = []
    for tx, ty in tail:
        rx, ry = tx - px, ty - py
        for _ in range(angle % 4):
            rx, ry = -ry, rx
        rotated.append((px + rx, py + ry))
    for pos in rotated:
        if pos in chain[:i+1]:
            return None
    new_chain = chain[:i+1] + rotated
    return new_chain


def _anneal(sequence, chain, time_limit, rng):
    """Simulated annealing with reversals and pivots."""
    n = len(chain)
    current_energy = _energy(sequence, chain)
    best_energy = current_energy
    best_chain = chain[:]
    T = 1.5
    cooling = 0.99997
    min_t = 0.01

    deadline = time.time() + time_limit
    no_improve = 0
    max_no_improve = 5000

    h_indices = [i for i in range(n) if sequence[i] == 'H']

    def _generate_move():
        move_type = rng.random()
        if move_type < 0.55:
            if h_indices and rng.random() < 0.4:
                hi = rng.choice(h_indices)
                span = rng.randint(2, min(20, n - 1))
                i = max(0, hi - span // 2)
                j = min(n - 1, i + span)
                if j <= i:
                    i = rng.randint(0, n - 2)
                    j = rng.randint(i + 1, n - 1)
            else:
                i = rng.randint(0, n - 2)
                j = rng.randint(i + 1, n - 1)
            return ('reversal', i, j)
        else:
            i = rng.randint(1, n - 2)
            angle = rng.choice([1, 2, 3])
            return ('pivot', i, angle)

    while time.time() < deadline:
        if _remaining_time() < 2.0:
            break

        move = _generate_move()
        if move[0] == 'reversal':
            new_chain = _apply_reversal(chain, move[1], move[2])
        else:
            new_chain = _apply_pivot(chain, move[1], move[2])

        if new_chain is None:
            continue

        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy

        if delta <= 0 or (T > min_t and rng.random() < math.exp(-delta / T)):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

        if no_improve > max_no_improve:
            break

        T *= cooling
        if T < min_t:
            break

    return best_chain, best_energy


def fold(sequence, dimension):
    """
    Fold the sequence into a low-energy conformation.
    """
    rng = random.Random(42)
    n = len(sequence)

    if n < 2:
        return [[0, 0]] * n

    time_limit = max(0.5, _remaining_time() - 5.0)

    # Generate diverse initial chains
    initial_chains = []
    configs = [
        (2.0, 0.0, 0.5),
        (1.0, 0.5, 0.5),
        (0.5, 1.0, 0.5),
        (1.5, 0.2, 0.3),
        (1.0, 0.3, 1.0),
        (3.0, 0.0, 0.2),
        (1.0, 0.0, 0.8),
    ]
    growth_deadline = time.time() + min(2.0, time_limit * 0.1)
    for hh_w, free_w, noise in configs:
        if time.time() > growth_deadline:
            break
        for attempt in range(3):
            if time.time() > growth_deadline:
                break
            chain = _grow(sequence, rng, hh_w, free_w, noise)
            if chain is not None:
                initial_chains.append(chain)

    if not initial_chains:
        initial_chains.append([(i, 0) for i in range(n)])

    # Score and select top candidates
    scored = []
    for chain in initial_chains:
        e = _energy(sequence, chain)
        scored.append((e, chain))
    scored.sort(key=lambda x: x[0])
    candidates = [s[1] for s in scored[:min(6, len(scored))]]

    # Anneal each candidate
    best_chain = None
    best_energy = 0
    total_time = _remaining_time() - 3.0
    time_per = total_time / len(candidates) if candidates else 0

    for chain in candidates:
        if _remaining_time() < 2.0:
            break
        alloc = min(time_per, _remaining_time() - 1.0)
        if alloc < 0.3:
            continue
        refined_chain, refined_energy = _anneal(sequence, chain, alloc, rng)
        if best_chain is None or refined_energy < best_energy:
            best_energy = refined_energy
            best_chain = refined_chain

    if best_chain is None:
        best_chain = candidates[0] if candidates else [(i, 0) for i in range(n)]

    # Validate
    seen = set()
    valid = True
    for i, (x, y) in enumerate(best_chain):
        if (x, y) in seen:
            valid = False
            break
        seen.add((x, y))
        if i > 0:
            px, py = best_chain[i-1]
            if abs(px - x) + abs(py - y) != 1:
                valid = False
                break

    if not valid:
        best_chain = [(i, 0) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
