"""HP lattice protein folding on the square lattice – scratch rewrite."""
# EVOLVE-BLOCK-START
import random
import time
import sys

_DEADLINE = 200.0  # safety margin before 210s kill


def _neighbours(point):
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _energy(sequence, chain):
    """Compute HP contact energy: -1 per H-H pair on neighbouring lattice sites not adjacent in chain."""
    n = len(chain)
    pos = {}
    for i, p in enumerate(chain):
        pos[p] = i
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = pos.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _is_valid(chain):
    """Check self-avoidance and unit steps."""
    n = len(chain)
    if n < 2:
        return True
    seen = set()
    for i, p in enumerate(chain):
        if p in seen:
            return False
        seen.add(p)
    for i in range(1, n):
        dx = abs(chain[i][0] - chain[i - 1][0])
        dy = abs(chain[i][1] - chain[i - 1][1])
        if dx + dy != 1:
            return False
    return True


def _local_search(sequence, chain, max_iters=2000, time_budget=None):
    """Simulated annealing local search with pivot moves."""
    n = len(chain)
    if n < 3:
        return chain, _energy(sequence, chain)

    best_chain = list(chain)
    best_energy = _energy(sequence, chain)
    current_chain = list(chain)
    current_energy = best_energy

    # Track positions for fast lookup
    pos = {}
    for i, p in enumerate(current_chain):
        pos[p] = i

    start_time = time.time()
    T = 2.0
    T_min = 0.01
    cool = 0.995

    for it in range(max_iters):
        if time_budget is not None and time.time() - start_time > time_budget:
            break
        if time.time() > _DEADLINE - 5:
            break

        # Try a pivot move: pick a random segment and rotate it 90, 180, or 270 degrees
        # Simpler: try moving a single non-end residue to an adjacent free site
        i = random.randint(1, n - 2)
        old_pos = current_chain[i]
        old_pos_t = tuple(old_pos)

        # Try each of the 4 neighbours
        candidates = []
        for np_ in _neighbours(old_pos_t):
            if np_ == tuple(current_chain[i - 1]) or np_ == tuple(current_chain[i + 1]):
                continue
            if np_ in pos:
                continue
            candidates.append(np_)

        if not candidates:
            continue

        new_pos = random.choice(candidates)

        # Compute delta energy
        # Remove residue i from old_pos, add to new_pos
        # Contacts involving residue i change, and contacts between i's neighbours and others might change
        # Simpler: recompute energy for the whole chain (O(n)) but only for H residues
        # For efficiency, let's just recompute

        # Temporarily move
        pos[old_pos_t] = -1  # mark as removed
        current_chain[i] = list(new_pos)
        pos[tuple(new_pos)] = i

        new_energy = _energy(sequence, current_chain)
        delta = new_energy - current_energy

        if delta < 0 or random.random() < pow(2.718281828, -delta / T):
            # Accept
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(current_chain)
        else:
            # Reject: revert
            pos[tuple(new_pos)] = -1
            current_chain[i] = list(old_pos_t)
            pos[old_pos_t] = i

        T *= cool
        if T < T_min:
            T = T_min

    return best_chain, best_energy


def _greedy_grow(sequence, rng, temperature=0.5):
    """Randomized greedy chain growth with contact preference."""
    n = len(sequence)
    if n == 0:
        return []

    # Start at origin
    chain = [(0, 0)]
    occupied = {(0, 0): 0}

    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Score: prefer sites that create H-H contacts
            score = rng.random() * temperature
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        score += 1.0
            # Slight preference for sites with more free neighbours (to avoid trapping)
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score += 0.05 * free
            options.append((score, site))

        if not options:
            return None

        # Pick best with some randomness
        options.sort(reverse=True)
        # Pick from top few
        k = min(len(options), 3)
        chosen = random.Random(rng.random()).choice(options[:k])
        site = chosen[1]
        occupied[site] = i
        chain.append(site)

    return chain


def _beam_search_grow(sequence, beam_width=20):
    """Beam search growth: keep top beam_width partial chains."""
    n = len(sequence)
    if n == 0:
        return []

    # Each beam entry: (chain, occupied_dict, energy)
    # Start
    beams = [([(0, 0)], {(0, 0): 0}, 0)]

    for i in range(1, n):
        new_beams = []
        for chain, occupied, energy in beams:
            last = chain[-1]
            for site in _neighbours(last):
                if site in occupied:
                    continue
                # Compute new energy delta
                delta = 0
                if sequence[i] == 'H':
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            delta -= 1  # energy decreases by 1 per contact
                new_chain = chain + [site]
                new_occupied = dict(occupied)
                new_occupied[site] = i
                new_energy = energy + delta
                new_beams.append((new_chain, new_occupied, new_energy))

        if not new_beams:
            return None

        # Keep top beam_width by energy (most negative)
        new_beams.sort(key=lambda x: x[2])
        beams = new_beams[:beam_width]

    if not beams:
        return None
    # Return best
    return beams[0][0]


def _hill_climb(sequence, chain, max_moves=5000):
    """Simple first-improvement hill climbing with single-residue moves."""
    n = len(chain)
    if n < 3:
        return chain, _energy(sequence, chain)

    current_chain = list(chain)
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy

    pos = {}
    for i, p in enumerate(current_chain):
        pos[p] = i

    for _ in range(max_moves):
        if time.time() > _DEADLINE - 5:
            break
        improved = False
        i = random.randint(1, n - 2)
        old_pos = tuple(current_chain[i])

        for np_ in _neighbours(old_pos):
            if np_ == tuple(current_chain[i - 1]) or np_ == tuple(current_chain[i + 1]):
                continue
            if np_ in pos:
                continue

            # Try move
            pos[old_pos] = -1
            current_chain[i] = list(np_)
            pos[np_] = i

            new_energy = _energy(sequence, current_chain)
            if new_energy < current_energy:
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(current_chain)
                improved = True
                break
            else:
                # Revert
                pos[np_] = -1
                current_chain[i] = list(old_pos)
                pos[old_pos] = i

        if not improved:
            # Try a different residue
            pass

    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation found within time budget using multiple strategies."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    if n == 2:
        return [[0, 0], [1, 0]]

    start_time = time.time()
    best_chain = None
    best_energy = 0  # start with 0 (no contacts)

    def update_best(chain):
        nonlocal best_chain, best_energy
        if chain is None:
            return
        if not _is_valid(chain):
            return
        e = _energy(sequence, chain)
        if e < best_energy:
            best_energy = e
            best_chain = list(chain)

    # Strategy 1: Beam search (exact for small n, good for larger)
    if n <= 30:
        chain = _beam_search_grow(sequence, beam_width=min(50, max(10, 10000 // n)))
        update_best(chain)
    elif n <= 60:
        chain = _beam_search_grow(sequence, beam_width=20)
        update_best(chain)

    # Strategy 2: Multiple greedy grows with local search
    rng = random.Random(42 + n)
    num_grows = max(10, min(100, 2000 // n))

    for _ in range(num_grows):
        if time.time() > _DEADLINE - 10:
            break
        chain = _greedy_grow(sequence, rng, temperature=0.3)
        if chain is None:
            continue
        update_best(chain)
        # Local search on the result
        chain_ls, energy_ls = _local_search(sequence, chain, max_iters=500, time_budget=1.0)
        update_best(chain_ls)

    # Strategy 3: More greedy grows with lower temperature
    if time.time() < _DEADLINE - 15:
        num_grows2 = max(5, min(50, 1000 // n))
        for _ in range(num_grows2):
            if time.time() > _DEADLINE - 10:
                break
            chain = _greedy_grow(sequence, rng, temperature=0.1)
            if chain is None:
                continue
            update_best(chain)
            chain_ls, energy_ls = _local_search(sequence, chain, max_iters=1000, time_budget=1.5)
            update_best(chain_ls)

    # Strategy 4: If we have a best chain, do more intensive local search
    if best_chain is not None and time.time() < _DEADLINE - 10:
        chain_ls, energy_ls = _local_search(sequence, best_chain, max_iters=5000, time_budget=_DEADLINE - 10 - time.time())
        update_best(chain_ls)

    # Fallback: straight line if nothing found
    if best_chain is None:
        best_chain = [[i, 0] for i in range(n)]
        best_energy = _energy(sequence, best_chain)

    return best_chain
# EVOLVE-BLOCK-END
