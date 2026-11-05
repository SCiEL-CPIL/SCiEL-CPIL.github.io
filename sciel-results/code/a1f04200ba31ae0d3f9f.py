"""HP lattice protein folding: greedy growth + pivot-move simulated annealing + polish."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_TOTAL_BUDGET = 195.0  # wall-clock safety margin before 210s kill

def _neighbours(point):
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))

def _energy(sequence, chain):
    """Compute HP contact energy: -1 per H-H pair on neighbouring lattice sites not adjacent in chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _greedy_grow(sequence, rng, temperature=1.0):
    """Randomized greedy chain growth with temperature-based selection."""
    n = len(sequence)
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}

    for i in range(1, n):
        last = chain[-1]
        candidates = []
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
            score = gain * 5.0 + free * 0.1 - rng.random() * temperature
            candidates.append((score, site))

        if not candidates:
            return None

        if temperature > 0.1 and len(candidates) > 1:
            max_score = max(s for s, _ in candidates)
            scale = temperature * 0.5 + 0.01
            weights = [math.exp((s - max_score) / scale) for s, _ in candidates]
            total = sum(weights)
            r = rng.random() * total
            cum = 0.0
            selected = candidates[-1][1]
            for (s, site), w in zip(candidates, weights):
                cum += w
                if r <= cum:
                    selected = site
                    break
        else:
            selected = max(candidates)[1]

        chain.append(selected)
        occupied[selected] = i

    return chain

def _try_pivot_move(chain, k, rot_choice):
    """
    Rigidly rotate the suffix (k+1..n-1) about residue k by 90/180/270 deg.
    Returns new_chain or None if any self-avoidance conflict.
    """
    n = len(chain)
    if k >= n - 1:
        return None

    px, py = chain[k]

    if rot_choice == 1:      # +90 deg
        def rot(x, y):
            return (-y, x)
    elif rot_choice == 2:    # 180 deg
        def rot(x, y):
            return (-x, -y)
    else:                    # -90 deg
        def rot(x, y):
            return (y, -x)

    prefix = set(chain[:k + 1])   # residues 0..k are fixed (k is the pivot)

    new_chain = list(chain[:k + 1])
    for j in range(k + 1, n):
        ox = chain[j][0] - px
        oy = chain[j][1] - py
        nx, ny = rot(ox, oy)
        new_pos = (px + nx, py + ny)
        if new_pos in prefix:
            return None
        new_chain.append(new_pos)
        prefix.add(new_pos)

    return new_chain

def _pivot_sa(sequence, chain, time_limit, rng):
    """Monte Carlo / simulated-annealing search over pivot moves."""
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)

    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy

    T_start = 1.6
    T_end = 0.02
    start_time = time.time()
    max_iters = 300000
    cooling = (T_end / T_start) ** (1.0 / max(1, max_iters))
    T = T_start

    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break

        k = rng.randint(0, n - 2)
        rot_choice = rng.randint(1, 3)

        new_chain = _try_pivot_move(current_chain, k, rot_choice)
        if new_chain is None:
            continue

        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy

        if delta < 0 or (T > 0.001 and rng.random() < math.exp(-delta / T)):
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)

        T *= cooling

    return best_chain, best_energy

def _polish(sequence, chain, time_limit, rng):
    """Final polish: fast single-residue hill-climbing using occupied map for O(1) lookups."""
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)

    current_chain = list(chain)
    occupied = {p: i for i, p in enumerate(current_chain)}
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy

    def contact_delta(idx, new_pos):
        """Net change in contact count if residue idx moves to new_pos."""
        delta = 0
        if sequence[idx] == 'H':
            for q in _neighbours(new_pos):
                j = occupied.get(q)
                if j is not None and j != idx and abs(j - idx) > 1 and sequence[j] == 'H':
                    delta += 1
        return delta

    start_time = time.time()
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(1, n - 1):
            if time.time() - start_time > time_limit:
                break
            x, y = current_chain[i]
            # Try the 4 neighbours of the current position
            for new_pos in _neighbours((x, y)):
                if new_pos in occupied and new_pos != (x, y):
                    continue
                # unit-step check with neighbours
                if i > 0:
                    px, py = current_chain[i - 1]
                    if abs(new_pos[0] - px) + abs(new_pos[1] - py) != 1:
                        continue
                if i < n - 1:
                    px, py = current_chain[i + 1]
                    if abs(new_pos[0] - px) + abs(new_pos[1] - py) != 1:
                        continue

                old_pos = current_chain[i]
                # remove old contacts of residue i, add new
                old_delta = 0
                new_delta = 0
                if sequence[i] == 'H':
                    for q in _neighbours(old_pos):
                        j = occupied.get(q)
                        if j is not None and j != i and abs(j - i) > 1 and sequence[j] == 'H':
                            old_delta += 1
                    for q in _neighbours(new_pos):
                        j = occupied.get(q)
                        if j is not None and j != i and abs(j - i) > 1 and sequence[j] == 'H':
                            new_delta += 1
                energy_delta = -(new_delta - old_delta)
                if energy_delta < 0:
                    # accept the move
                    del occupied[old_pos]
                    occupied[new_pos] = i
                    current_chain[i] = new_pos
                    current_energy += energy_delta
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                    improved = True
                    break

    return best_chain, best_energy

def fold(sequence, dimension):
    """Fold a sequence into a low-energy self-avoiding conformation."""
    n = len(sequence)
    if n == 0:
        return []

    elapsed = time.time() - _GLOBAL_START
    if elapsed > _TOTAL_BUDGET:
        return [[i, 0] for i in range(n)]

    remaining = _TOTAL_BUDGET - elapsed
    base_time = min(remaining, 18.0)
    if n <= 30:
        time_limit = base_time * 0.4
    elif n <= 50:
        time_limit = base_time * 0.6
    elif n <= 70:
        time_limit = base_time * 0.8
    else:
        time_limit = base_time

    if time_limit <= 0.5:
        rng = random.Random(12345 + n)
        initial = _greedy_grow(sequence, rng)
        if initial:
            return [list(p) for p in initial]
        return [[i, 0] for i in range(n)]

    rng = random.Random(12345 + n)

    best_chain = None
    best_energy = 0

    # Reserve time: 80% for growth+SA, 20% for polish
    sa_time = time_limit * 0.80
    polish_time = time_limit * 0.20

    # Phase 1: multiple greedy growths
    grow_deadline = time.time() + sa_time * 0.35
    grows_done = 0
    while grows_done < 4000 and time.time() < grow_deadline:
        temperature = 0.3 + rng.random() * 1.5
        chain = _greedy_grow(sequence, rng, temperature)
        if chain is not None:
            grows_done += 1
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = list(chain)

    if best_chain is None:
        best_chain = _greedy_grow(sequence, rng, 0.3)
        if best_chain is None:
            return [[i, 0] for i in range(n)]
        best_energy = _energy(sequence, best_chain)

    # Phase 2: pivot-move simulated annealing
    if time.time() < _GLOBAL_START + _TOTAL_BUDGET - polish_time - 2.0:
        sa_remaining = (_GLOBAL_START + _TOTAL_BUDGET - polish_time - 2.0) - time.time()
        improved_chain, improved_energy = _pivot_sa(
            sequence, best_chain, time_limit=max(0.5, sa_remaining), rng=rng
        )
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy

    # Phase 3: polish
    if time.time() < _GLOBAL_START + _TOTAL_BUDGET - 3.0:
        polish_remaining = max(0.5, (_GLOBAL_START + _TOTAL_BUDGET - 3.0) - time.time())
        polished_chain, polished_energy = _polish(
            sequence, best_chain, time_limit=min(polish_time, polish_remaining), rng=rng
        )
        if polished_energy < best_energy:
            best_chain = polished_chain
            best_energy = polished_energy

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
