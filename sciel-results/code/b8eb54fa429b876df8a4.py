"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time


def _neighbours2d(point):
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours2d(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _grow(sequence, dimension, rng, temperature=0.5):
    """Randomized greedy growth with temperature for exploration."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    n = len(sequence)
    for i in range(1, n):
        options = []
        for site in _neighbours2d(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours2d(site):
                    if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == 'H':
                        gain += 1
            free = sum(1 for q in _neighbours2d(site) if q not in occupied)
            score = gain + 0.05 * free
            options.append((score, site))
        if not options:
            return None
        # Boltzmann-like selection
        max_score = max(s for s, _ in options)
        weights = []
        for s, site in options:
            w = 1.0 if s == max_score else (0.3 ** (max_score - s))
            weights.append(w)
        total = sum(weights)
        r = rng.random() * total
        cum = 0
        chosen = options[-1][1]
        for (s, site), w in zip(options, weights):
            cum += w
            if r <= cum:
                chosen = site
                break
        occupied[chosen] = i
        chain.append(chosen)
    return chain


def _local_search_sa(sequence, chain, max_iter=20000, t0=0.5, t1=0.01):
    """Simulated annealing with single-residue moves and incremental energy."""
    n = len(chain)
    if n < 4:
        return list(chain), _energy(sequence, chain)

    best_chain = list(chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    cur_energy = _energy(sequence, best_chain)
    best_energy = cur_energy

    # Precompute: for each residue, which other residues are H-H neighbors (non-adjacent)
    # We'll track contacts incrementally.
    # A contact exists between i and j (|i-j|>1) if seq[i]==seq[j]=='H' and chain[i] adjacent to chain[j].

    def contact_count(i, j):
        """Return 1 if i,j form a non-adjacent H-H contact, else 0."""
        if abs(i - j) <= 1:
            return 0
        if sequence[i] != 'H' or sequence[j] != 'H':
            return 0
        pi = best_chain[i]
        pj = best_chain[j]
        dx = abs(pi[0] - pj[0])
        dy = abs(pi[1] - pj[1])
        return 1 if dx + dy == 1 else 0

    rng = random.Random(42)
    for it in range(max_iter):
        temp = t0 * (t1 / t0) ** (it / max(max_iter, 1))
        # Pick a random interior residue
        i = rng.randint(1, n - 2)
        cur_pos = best_chain[i]
        # Try each of 4 neighbors
        for new_pos in _neighbours2d(cur_pos):
            if new_pos == cur_pos:
                continue
            if new_pos in occupied:
                continue
            # Check self-avoidance: new_pos must not be adjacent to any residue j with |j-i|>1
            valid = True
            for q in _neighbours2d(new_pos):
                j = occupied.get(q)
                if j is not None and abs(j - i) > 1:
                    valid = False
                    break
            if not valid:
                continue

            # Compute energy delta: remove contacts involving i at cur_pos, add at new_pos
            delta = 0
            for j in range(n):
                if abs(j - i) <= 1:
                    continue
                if sequence[i] != 'H' or sequence[j] != 'H':
                    continue
                # Old: was there a contact?
                old_c = contact_count(i, j)
                # New: would there be a contact?
                pi_old = cur_pos
                pj = best_chain[j]
                dx = abs(pi_old[0] - pj[0])
                dy = abs(pi_old[1] - pj[1])
                old_c = 1 if (dx + dy == 1) else 0
                dx = abs(new_pos[0] - pj[0])
                dy = abs(new_pos[1] - pj[1])
                new_c = 1 if (dx + dy == 1) else 0
                delta += (new_c - old_c)

            new_energy = cur_energy + delta  # energy is -contacts, so delta in contacts means -delta in energy
            # Actually: energy = -contacts. If contacts increase by d, energy decreases by d.
            # delta computed above is (new_contacts - old_contacts), so new_energy = cur_energy - delta
            new_energy = cur_energy - delta

            accept = False
            if new_energy < cur_energy:
                accept = True
            else:
                diff = new_energy - cur_energy
                if diff < 0:
                    accept = True
                else:
                    if temp > 0.001:
                        import math
                        if rng.random() < math.exp(-diff / max(temp, 1e-9)):
                            accept = True

            if accept:
                # Make the move
                occupied.pop(cur_pos)
                best_chain[i] = new_pos
                occupied[new_pos] = i
                cur_energy = new_energy
                if cur_energy < best_energy:
                    best_energy = cur_energy
                break  # One move per iteration

    return best_chain, best_energy


def _local_search_hill(sequence, chain, max_iter=10000):
    """First-improvement hill climbing with incremental energy."""
    n = len(chain)
    if n < 4:
        return list(chain), _energy(sequence, chain)

    best_chain = list(chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    cur_energy = _energy(sequence, best_chain)
    best_energy = cur_energy

    for _ in range(max_iter):
        improved = False
        for i in range(1, n - 1):
            cur_pos = best_chain[i]
            for new_pos in _neighbours2d(cur_pos):
                if new_pos == cur_pos or new_pos in occupied:
                    continue
                # Check self-avoidance
                valid = True
                for q in _neighbours2d(new_pos):
                    j = occupied.get(q)
                    if j is not None and abs(j - i) > 1:
                        valid = False
                        break
                if not valid:
                    continue

                # Energy delta
                delta = 0
                for j in range(n):
                    if abs(j - i) <= 1 or sequence[i] != 'H' or sequence[j] != 'H':
                        continue
                    pj = best_chain[j]
                    dx = abs(cur_pos[0] - pj[0]) + abs(cur_pos[1] - pj[1])
                    old_c = 1 if dx == 1 else 0
                    dx = abs(new_pos[0] - pj[0]) + abs(new_pos[1] - pj[1])
                    new_c = 1 if dx == 1 else 0
                    delta += (new_c - old_c)

                new_energy = cur_energy - delta
                if new_energy < cur_energy:
                    occupied.pop(cur_pos)
                    best_chain[i] = new_pos
                    occupied[new_pos] = i
                    cur_energy = new_energy
                    if cur_energy < best_energy:
                        best_energy = cur_energy
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return best_chain, best_energy


def fold(sequence, dimension):
    start_time = time.time()
    deadline = start_time + 190.0  # Safety margin before 210s kill

    n = len(sequence)
    rng = random.Random(12345 + len(sequence))

    best = None
    best_energy = 0  # We want most negative

    # Number of restarts depends on length
    if n <= 20:
        num_restarts = 800
    elif n <= 40:
        num_restarts = 500
    elif n <= 60:
        num_restarts = 300
    else:
        num_restarts = 150

    for attempt in range(num_restarts):
        if time.time() > deadline - 10:
            break

        # Vary temperature for diversity
        temp = 0.3 + 0.4 * rng.random()
        chain = _grow(sequence, dimension, rng, temperature=temp)
        if chain is None:
            continue

        # Hill climb first (fast), then SA if time permits
        chain, energy = _local_search_hill(sequence, chain, max_iter=max(500, 5000 // max(n // 10, 1)))

        if time.time() < deadline - 30:
            chain, energy2 = _local_search_sa(sequence, chain, max_iter=max(2000, 20000 // max(n // 10, 1)))
            if energy2 < energy:
                energy = energy2

        if energy < best_energy:
            best_energy = energy
            best = chain

    if best is None:
        best = [(i, 0) for i in range(n)]

    return [list(p) for p in best]
# EVOLVE-BLOCK-END
