"""HP lattice protein folding on the cubic lattice.

Strategy:
  1. Fast contact-map-aware local search (greedy H-residue moves).
  2. Simulated annealing with efficient delta-energy computation.
  3. Multi-start growth with diverse bias parameters.
  4. Repeated annealing cycles with restarts for best chain.
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


def _get_neighbours_fast(x, y, z):
    """Fast neighbor generation using precomputed offsets."""
    return [
        (x + 1, y, z), (x - 1, y, z),
        (x, y + 1, z), (x, y - 1, z),
        (x, y, z + 1), (x, y, z - 1)
    ]


def _energy(sequence, chain):
    """Compute HP energy: -number of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _get_neighbours_fast(*p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _delta_energy_fast(sequence, chain, pos_map, i, new_pos):
    """Fast delta energy computation using precomputed position map."""
    old_pos = chain[i]
    if old_pos == new_pos:
        return 0
    
    is_h = sequence[i] == 'H'
    if not is_h:
        return 0
    
    # Count H-H contacts involving residue i at old_pos
    old_contacts = 0
    for q in _get_neighbours_fast(*old_pos):
        j = pos_map.get(q)
        if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
            old_contacts += 1
    
    # Count H-H contacts involving residue i at new_pos
    new_contacts = 0
    for q in _get_neighbours_fast(*new_pos):
        j = pos_map.get(q)
        if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
            new_contacts += 1
    
    return -(new_contacts - old_contacts)


def _grow(sequence, dimension, rng, bias=0.7):
    """Randomized greedy chain growth with tunable contact bias."""
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)

    for i in range(1, n):
        options = []
        last = chain[-1]
        for site in _get_neighbours_fast(*last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for q in _get_neighbours_fast(*site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            free = 0
            for q in _get_neighbours_fast(*site):
                if q not in occupied:
                    free += 1
            # Mix contact gain and free space with tunable bias
            score = gain * bias * 10.0 + free * (1 - bias) * 0.5 + rng.random() * 0.1
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
    if n < 4:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    pos_map = {p: i for i, p in enumerate(current_chain)}
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Identify H positions (excluding endpoints)
    h_indices = [i for i in range(1, n - 1) if sequence[i] == 'H']
    
    for iteration in range(max_iter):
        if time.time() >= deadline - 5.0:
            break
        
        improved = False
        rng.shuffle(h_indices)
        
        for i in h_indices:
            old_pos = current_chain[i]
            prev = current_chain[i - 1]
            nxt = current_chain[i + 1]
            
            for new_pos in _get_neighbours_fast(*old_pos):
                if new_pos == old_pos:
                    continue
                # Check unit step constraints
                d1 = abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) + abs(new_pos[2] - prev[2])
                d2 = abs(new_pos[0] - nxt[0]) + abs(new_pos[1] - nxt[1]) + abs(new_pos[2] - nxt[2])
                if d1 != 1 or d2 != 1:
                    continue
                if new_pos in pos_map:
                    continue

                # Compute delta energy
                delta = _delta_energy_fast(sequence, current_chain, pos_map, i, new_pos)
                
                if delta < 0:
                    # Accept the move
                    pos_map.pop(old_pos)
                    pos_map[new_pos] = i
                    current_chain[i] = new_pos
                    current_energy += delta
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                    improved = True
                    break
            if improved:
                break

        if not improved:
            break

    return best_chain, best_energy


def _anneal(sequence, chain, dimension, rng, deadline):
    """Simulated annealing with geometric cooling and efficient delta energy."""
    n = len(chain)
    if n < 4:
        return chain

    best_chain = list(chain)
    best_energy = _energy(sequence, chain)
    current_chain = list(chain)
    current_energy = best_energy
    pos_set = set(current_chain)
    pos_map = {p: i for i, p in enumerate(current_chain)}

    T0 = max(1.0, abs(current_energy) * 0.5)
    T_min = 0.01
    margin = 5.0

    time_left = deadline - time.time()
    if time_left <= margin:
        return best_chain

    total_steps = int((time_left - margin) * 5000)
    total_steps = max(1000, min(total_steps, 50000))

    cooling_factor = (T_min / T0) ** (1.0 / total_steps)
    T = T0

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
            for new_pos in _get_neighbours_fast(*old_pos):
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

            # Compute delta energy efficiently
            delta = _delta_energy_fast(sequence, current_chain, pos_map, i, new_pos)

            if delta <= 0 or rng.random() < math.exp(-delta / T):
                pos_set.discard(old_pos)
                pos_set.add(new_pos)
                pos_map.pop(old_pos)
                pos_map[new_pos] = i
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
                pos_map = {p: idx for idx, p in enumerate(current_chain)}
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
                pos_map = {p: idx for idx, p in enumerate(current_chain)}
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

    # Phase 1: Multiple growth starts with diverse biases
    biases = [0.5, 0.7, 0.9, 0.3, 0.8, 0.6, 1.0, 0.4, 0.75, 0.65]
    num_restarts = min(15, max(5, int(40.0 / max(1, n) * 150)))
    
    for restart in range(num_restarts):
        if time.time() >= deadline - 60.0:
            break

        bias = biases[restart % len(biases)]
        chain = _grow(sequence, dimension, rng, bias=bias)
        if chain is None:
            continue

        # Local search to polish
        local_chain, local_energy = _local_search(sequence, chain, rng, deadline, max_iter=3000)

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

    # Phase 3: If time remains, try another growth + local search + anneal
    if time.time() < deadline - 30:
        chain2 = _grow(sequence, dimension, rng, bias=0.8)
        if chain2 is not None:
            e2 = _energy(sequence, chain2)
            if e2 < best_energy:
                best, best_energy = chain2, e2
            else:
                # Local search on new chain
                local_chain2, local_energy2 = _local_search(sequence, chain2, rng, deadline, max_iter=2000)
                if local_energy2 < best_energy:
                    best, best_energy = local_chain2, local_energy2
            
            if time.time() < deadline - 10:
                annealed2 = _anneal(sequence, best, dimension, rng, deadline)
                e2a = _energy(sequence, annealed2)
                if e2a < best_energy:
                    best, best_energy = annealed2, e2a

    # Phase 4: Final local search polish
    if time.time() < deadline - 5:
        final_chain, final_energy = _local_search(sequence, best, rng, deadline, max_iter=1000)
        if final_energy < best_energy:
            best, best_energy = final_chain, final_energy

    return [list(p) for p in best]
# EVOLVE-BLOCK-END
