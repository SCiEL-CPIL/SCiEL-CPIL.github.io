"""HP lattice protein folding on the cubic lattice.

Strategy:
  1. Multi-start randomized greedy growth with contact-aware scoring.
  2. Fast local search (greedy H-residue moves) to polish each grown chain.
  3. Simulated annealing with geometric cooling on the best chain.
  4. Return the best conformation found within the time budget.
"""
# EVOLVE-BLOCK-START
import random
import time
import math

# Pre-computed neighbor offsets for 3D cubic lattice
_NEIGHBOR_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]


def _neighbours(point):
    """Generate 6 neighbours of a 3D lattice point."""
    x, y, z = point
    for dx, dy, dz in _NEIGHBOR_OFFSETS:
        yield (x + dx, y + dy, z + dz)


def _energy(sequence, chain):
    """Compute HP energy: -number of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _grow(sequence, dimension, rng):
    """Randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)

    for i in range(1, n):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            score = gain * 10.0 + free * 0.1 + rng.random()
            options.append((score, site))

        if not options:
            return None

        # Pick best with some randomness
        options.sort(key=lambda x: -x[0])
        k = min(len(options), max(1, int(3 * rng.random())))
        chosen = options[rng.randint(0, k - 1)][1]
        occupied[chosen] = i
        chain.append(chosen)

    return chain


def _local_search(sequence, chain, rng, deadline, max_iter=5000):
    """Greedy local search: try moving each H residue to improve energy."""
    n = len(chain)
    pos_set = set(chain)
    current_energy = _energy(sequence, chain)
    best_energy = current_energy
    best_chain = list(chain)

    for it in range(max_iter):
        if time.time() >= deadline - 5.0:
            break

        improved = False
        # Try moving each H residue
        h_indices = [i for i in range(1, n - 1) if sequence[i] == 'H']
        rng.shuffle(h_indices)

        for i in h_indices:
            old_pos = chain[i]
            prev = chain[i - 1]
            nxt = chain[i + 1]

            for new_pos in _neighbours(old_pos):
                if new_pos == old_pos:
                    continue
                # Check unit step constraints
                d1 = abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) + abs(new_pos[2] - prev[2])
                d2 = abs(new_pos[0] - nxt[0]) + abs(new_pos[1] - nxt[1]) + abs(new_pos[2] - nxt[2])
                if d1 != 1 or d2 != 1:
                    continue
                if new_pos in pos_set:
                    continue

                # Compute delta energy
                # Remove old contacts involving i
                old_contacts = 0
                for q in _neighbours(old_pos):
                    j = None
                    for idx, p in enumerate(chain):
                        if p == q and idx != i:
                            j = idx
                            break
                    if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                        old_contacts += 1

                new_contacts = 0
                for q in _neighbours(new_pos):
                    j = None
                    for idx, p in enumerate(chain):
                        if p == q and idx != i:
                            j = idx
                            break
                    if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                        new_contacts += 1

                delta = -(new_contacts - old_contacts)
                if delta < 0:
                    # Accept the move
                    pos_set.discard(old_pos)
                    pos_set.add(new_pos)
                    chain[i] = new_pos
                    current_energy += delta
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                    improved = True
                    break
            if improved:
                break

        if not improved:
            break

    return best_chain, best_energy


def _anneal(sequence, chain, dimension, rng, deadline):
    """Simulated annealing with geometric cooling."""
    n = len(chain)
    if n < 4:
        return chain

    best_chain = list(chain)
    best_energy = _energy(sequence, chain)
    current_chain = list(chain)
    current_energy = best_energy

    T0 = max(1.0, abs(current_energy) * 0.5)
    T_min = 0.01
    margin = 5.0

    time_left = deadline - time.time()
    if time_left <= margin:
        return best_chain

    total_steps = int((time_left - margin) * 3000)
    total_steps = max(500, min(total_steps, 30000))

    cooling_factor = (T_min / T0) ** (1.0 / total_steps)
    T = T0

    pos_set = set(current_chain)

    for step in range(total_steps):
        if time.time() >= deadline - margin:
            break
        T *= cooling_factor
        if T < T_min:
            T = T_min

        move_type = rng.random()

        if move_type < 0.6:
            # Flip: move one residue to a random valid neighbor
            i = rng.randint(1, n - 2)
            old_pos = current_chain[i]
            prev = current_chain[i - 1]
            nxt = current_chain[i + 1]

            valid_options = []
            for new_pos in _neighbours(old_pos):
                d1 = abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) + abs(new_pos[2] - prev[2])
                d2 = abs(new_pos[0] - nxt[0]) + abs(new_pos[1] - nxt[1]) + abs(new_pos[2] - nxt[2])
                if d1 != 1 or d2 != 1:
                    continue
                if new_pos == old_pos:
                    continue
                if new_pos in pos_set:
                    continue
                valid_options.append(new_pos)

            if not valid_options:
                continue

            new_pos = rng.choice(valid_options)

            # Compute delta energy
            is_h = sequence[i] == 'H'
            if not is_h:
                delta = 0
            else:
                old_contacts = 0
                for q in _neighbours(old_pos):
                    for j, p in enumerate(current_chain):
                        if p == q and j != i and abs(j - i) > 1 and sequence[j] == 'H':
                            old_contacts += 1
                            break

                new_contacts = 0
                for q in _neighbours(new_pos):
                    for j, p in enumerate(current_chain):
                        if p == q and j != i and abs(j - i) > 1 and sequence[j] == 'H':
                            new_contacts += 1
                            break

                delta = -(new_contacts - old_contacts)

            if delta <= 0 or rng.random() < math.exp(-delta / T):
                pos_set.discard(old_pos)
                pos_set.add(new_pos)
                current_chain[i] = new_pos
                current_energy += delta
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)

        elif move_type < 0.85:
            # Pivot move
            i = rng.randint(1, n - 3)
            axis = rng.randint(0, 2)
            sign = rng.choice([1, -1])

            pivot = current_chain[i]
            tail = current_chain[i + 1:]

            new_tail = []
            for pos in tail:
                rel = (pos[0] - pivot[0], pos[1] - pivot[1], pos[2] - pivot[2])
                if axis == 0:
                    if sign == 1:
                        new_rel = (rel[0], -rel[2], rel[1])
                    else:
                        new_rel = (rel[0], rel[2], -rel[1])
                elif axis == 1:
                    if sign == 1:
                        new_rel = (rel[2], rel[1], -rel[0])
                    else:
                        new_rel = (-rel[2], rel[1], rel[0])
                else:
                    if sign == 1:
                        new_rel = (-rel[1], rel[0], rel[2])
                    else:
                        new_rel = (rel[1], -rel[0], rel[2])
                new_tail.append((pivot[0] + new_rel[0], pivot[1] + new_rel[1], pivot[2] + new_rel[2]))

            new_chain = current_chain[:i + 1] + new_tail

            if len(set(new_chain)) != n:
                continue

            diff = abs(new_chain[i + 1][0] - new_chain[i][0]) + abs(new_chain[i + 1][1] - new_chain[i][1]) + abs(new_chain[i + 1][2] - new_chain[i][2])
            if diff != 1:
                continue

            new_energy = _energy(sequence, new_chain)
            delta = new_energy - current_energy

            if delta <= 0 or rng.random() < math.exp(-delta / T):
                current_chain = new_chain
                pos_set = set(current_chain)
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)

        else:
            # Reversal
            a = rng.randint(0, n - 2)
            b = rng.randint(a + 1, n - 1)

            new_chain = list(current_chain)
            new_chain[a:b + 1] = reversed(new_chain[a:b + 1])

            if len(set(new_chain)) != n:
                continue

            valid = True
            for k in range(n - 1):
                d = abs(new_chain[k + 1][0] - new_chain[k][0]) + abs(new_chain[k + 1][1] - new_chain[k][1]) + abs(new_chain[k + 1][2] - new_chain[k][2])
                if d != 1:
                    valid = False
                    break
            if not valid:
                continue

            new_energy = _energy(sequence, new_chain)
            delta = new_energy - current_energy

            if delta <= 0 or rng.random() < math.exp(-delta / T):
                current_chain = new_chain
                pos_set = set(current_chain)
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)

    return best_chain


def fold(sequence, dimension):
    """Fold an HP sequence on the 3D cubic lattice."""
    deadline = time.time() + 230.0  # 230s budget with safety margin

    rng = random.Random(12345 + len(sequence))
    best = None
    best_energy = 0

    n = len(sequence)

    # Phase 1: Multiple growth starts with local search
    num_restarts = min(20, max(5, int(30.0 / max(1, n) * 200)))

    for restart in range(num_restarts):
        if time.time() >= deadline - 60.0:
            break

        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue

        # Local search to polish
        local_chain, local_energy = _local_search(sequence, chain, rng, deadline, max_iter=2000)

        if local_energy < best_energy:
            best, best_energy = local_chain, local_energy

    if best is None:
        # Fallback: straight line
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = 0

    # Phase 2: Anneal the best chain
    time_left = deadline - time.time()
    if time_left > 15:
        annealed = _anneal(sequence, best, dimension, rng, deadline)
        anneal_energy = _energy(sequence, annealed)
        if anneal_energy < best_energy:
            best, best_energy = annealed, anneal_energy

    # Phase 3: If time remains, try another growth + anneal
    if time.time() < deadline - 30:
        chain2 = _grow(sequence, dimension, rng)
        if chain2 is not None:
            e2 = _energy(sequence, chain2)
            if e2 < best_energy:
                best, best_energy = chain2, e2
            annealed2 = _anneal(sequence, best, dimension, rng, deadline)
            e2a = _energy(sequence, annealed2)
            if e2a < best_energy:
                best, best_energy = annealed2, e2a

    return [list(p) for p in best]
# EVOLVE-BLOCK-END
