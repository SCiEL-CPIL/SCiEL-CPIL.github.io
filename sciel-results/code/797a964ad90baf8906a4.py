"""HP lattice protein folding using contact-guided growth with simulated annealing refinement."""
# EVOLVE-BLOCK-START
import random
import time
import math
from collections import defaultdict

# Deadline guard: 180s wall-clock total for all sequences, leave safety margin
_DEADLINE = time.time() + 175.0


def _neighbours(point):
    """Yield the four lattice neighbours of a 2D point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _energy(sequence, chain):
    """Compute the HP contact energy (negative number of H-H contacts)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _count_contacts(sequence, chain):
    """Return the number of H-H contacts (positive)."""
    return -_energy(sequence, chain)


def _greedy_grow(sequence, dimension, rng, contact_priority=True):
    """
    Grow the chain greedily, preferring sites that create new H-H contacts.
    Returns None if the chain gets stuck.
    """
    n = len(sequence)
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}

    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            # Free space heuristic: prefer sites with more open neighbours
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 10.0 + free * 0.1 + rng.random() * 0.5
            options.append((score, site))

        if not options:
            return None

        options.sort(key=lambda x: -x[0])
        # Pick from top few with some randomness
        top_k = min(3, len(options))
        if rng.random() < 0.7:
            site = options[0][1]
        else:
            site = options[rng.randint(0, top_k - 1)][1]

        occupied[site] = i
        chain.append(site)

    return chain


def _simulated_annealing(sequence, chain, rng, max_iter=5000, t0=2.0, t1=0.01):
    """
    Refine a valid conformation using simulated annealing with move and reversal operations.
    Returns the best chain found.
    """
    n = len(sequence)
    current = list(chain)
    current_energy = _energy(sequence, current)
    best = list(current)
    best_energy = current_energy

    where = {p: i for i, p in enumerate(current)}
    h_set = set(i for i, c in enumerate(sequence) if c == 'H')

    def _delta_energy_for_move(pos, new_site):
        """Compute energy change if residue at index pos moves to new_site."""
        old_site = current[pos]
        # Remove contacts involving old_site
        delta = 0
        if sequence[pos] == 'H':
            for q in _neighbours(old_site):
                j = where.get(q)
                if j is not None and j != pos and j != pos - 1 and j != pos + 1:
                    if sequence[j] == 'H':
                        delta += 1  # losing a contact
        if sequence[pos] == 'H':
            for q in _neighbours(new_site):
                j = where.get(q)
                if j is not None and j != pos and j != pos - 1 and j != pos + 1:
                    if sequence[j] == 'H':
                        delta -= 1  # gaining a contact
        return -delta  # energy change (negative is better)

    for it in range(max_iter):
        temp = t0 * (t1 / t0) ** (it / max_iter)

        if rng.random() < 0.5:
            # Move operation: pick a random residue and try to move it to a free neighbour
            pos = rng.randint(1, n - 2)
            old_site = current[pos]
            candidates = []
            for site in _neighbours(old_site):
                if site in where:
                    # Check if it's an adjacent residue in the chain
                    idx = where[site]
                    if abs(idx - pos) <= 1:
                        continue
                    continue
                candidates.append(site)
            if not candidates:
                continue
            new_site = rng.choice(candidates)

            # Check self-avoidance: new_site must not be occupied
            if new_site in where:
                continue

            # Compute delta energy
            delta = _delta_energy_for_move(pos, new_site)

            if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-10)):
                # Accept move
                # Update where
                del where[old_site]
                where[new_site] = pos
                current[pos] = new_site

                new_energy = current_energy + delta
                if new_energy < best_energy:
                    best_energy = new_energy
                    best = list(current)
                current_energy = new_energy
        else:
            # Reversal operation: reverse a segment of the chain
            if n < 4:
                continue
            a = rng.randint(1, n - 3)
            b = rng.randint(a + 1, n - 2)
            if b - a < 2:
                continue

            # Check if reversal is valid: the reversed segment must not collide with itself or the rest
            segment = current[a:b + 1]
            # Build set of positions in the segment
            seg_set = set(segment)
            # Check internal collisions after reversal
            reversed_seg = segment[::-1]
            # The endpoints a-1 and b+1 stay fixed
            # Check that no position in reversed_seg (except endpoints which connect to a-1 and b+1) collides
            # Actually, we need to check that the reversed segment doesn't overlap with the rest of the chain
            rest_set = set(current[:a]) | set(current[b + 1:])
            # The new positions for the segment are the same set, just in reverse order
            # So we just need to check that consecutive residues in the reversed segment are adjacent
            # and that the connections at a-1->a and b->b+1 are valid
            # Check adjacency in reversed segment
            valid = True
            for k in range(len(reversed_seg) - 1):
                p1, p2 = reversed_seg[k], reversed_seg[k + 1]
                if abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) != 1:
                    valid = False
                    break
            if not valid:
                continue
            # Check connection at a-1 to reversed_seg[0] = current[b]
            p1, p2 = current[a - 1], current[b]
            if abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) != 1:
                continue
            # Check connection at b+1 to reversed_seg[-1] = current[a]
            p1, p2 = current[b + 1], current[a]
            if abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) != 1:
                continue
            # Check no overlap with rest
            for p in reversed_seg:
                if p in rest_set:
                    valid = False
                    break
            if not valid:
                continue

            # Compute delta energy for reversal
            # This is complex; approximate by recomputing
            old_segment = current[a:b + 1]
            current[a:b + 1] = reversed_seg
            new_energy = _energy(sequence, current)
            delta = new_energy - current_energy
            current_energy = new_energy
            # Update where
            for idx in range(a, b + 1):
                where[current[idx]] = idx

            if delta < 0 or rng.random() < math.exp(-delta / max(temp, 1e-10)):
                if current_energy < best_energy:
                    best_energy = current_energy
                    best = list(current)
            else:
                # Reject: undo
                current[a:b + 1] = old_segment
                current_energy = best_energy if False else current_energy - delta
                for idx in range(a, b + 1):
                    where[current[idx]] = idx

    return best


def fold(sequence, dimension):
    """
    Fold the HP sequence into a low-energy conformation on the square lattice.
    Returns a list of [x, y] integer coordinates.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    if n == 2:
        return [[0, 0], [1, 0]]

    rng = random.Random(12345 + len(sequence))

    # Phase 1: Generate diverse initial conformations via greedy growth
    initial_chains = []
    num_greedy = 50 if n <= 50 else 30

    for _ in range(num_greedy):
        if time.time() > _DEADLINE - 30:
            break
        chain = _greedy_grow(sequence, dimension, rng)
        if chain is not None:
            initial_chains.append(chain)

    if not initial_chains:
        # Fallback: straight line
        return [[i, 0] for i in range(n)]

    # Phase 2: Refine the best few initial chains with simulated annealing
    # Sort by energy
    scored = []
    for chain in initial_chains:
        e = _energy(sequence, chain)
        scored.append((e, chain))
    scored.sort(key=lambda x: x[0])

    # Take top candidates for refinement
    num_refine = min(5, len(scored))
    best_chain = scored[0][1]
    best_energy = scored[0][0]

    for idx in range(num_refine):
        if time.time() > _DEADLINE - 20:
            break
        e, chain = scored[idx]
        # Only refine if it's promising
        if e > best_energy + 3:
            continue
        refined = _simulated_annealing(sequence, chain, rng, max_iter=3000)
        re = _energy(sequence, refined)
        if re < best_energy:
            best_energy = re
            best_chain = refined

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
