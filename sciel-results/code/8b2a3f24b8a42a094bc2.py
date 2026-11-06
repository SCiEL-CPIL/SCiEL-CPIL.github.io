"""HP lattice protein folding on the square lattice – scratch rewrite."""
# EVOLVE-BLOCK-START
import random
import time
import math
from collections import deque

# Total wall-clock budget (seconds) – evaluator kills at 210 s; use 200 s to be safe
_TOTAL_DEADLINE = 200.0
_start_time = time.time()


def _remaining_time():
    return _TOTAL_DEADLINE - (time.time() - _start_time)


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _greedy_grow(sequence, dimension, rng, temperature=0.3):
    """Randomized greedy chain growth with temperature-controlled noise."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
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
            score = gain + 0.1 * free + rng.random() * temperature
            options.append((score, site))
        if not options:
            return None
        # Pick best with some randomness
        options.sort(reverse=True)
        # Boltzmann-like selection: pick from top few with probability
        k = min(len(options), 3)
        weights = [math.exp(options[j][0] - options[0][0] / max(temperature, 0.01)) for j in range(k)]
        total_w = sum(weights)
        r = rng.random() * total_w
        cum = 0
        chosen = k - 1
        for j in range(k):
            cum += weights[j]
            if r <= cum:
                chosen = j
                break
        site = options[chosen][1]
        occupied[site] = i
        chain.append(site)
    return chain


def _local_search(sequence, dimension, chain, max_iters=500):
    """Improve a conformation via single-residue moves and rotations."""
    where = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    n = len(sequence)

    for it in range(max_iters):
        if _remaining_time() < 5.0:
            break
        improved = False
        # Try moving each residue (except endpoints) to a neighbouring free site
        # Process in random order
        indices = list(range(1, n - 1))
        random.shuffle(indices)
        for i in indices:
            if _remaining_time() < 5.0:
                break
            p = chain[i]
            for q in _neighbours(p):
                if q == chain[i - 1] or q == chain[i + 1]:
                    continue
                if q in where:
                    continue
                # Check that moving i to q doesn't create self-intersection
                # (q is already checked as free)
                # Compute energy change
                old_energy = current_energy
                # Remove i from chain
                del where[p]
                chain[i] = q
                where[q] = i
                new_energy = _energy(sequence, chain)
                if new_energy < best_energy:
                    best_chain = list(chain)
                    best_energy = new_energy
                    current_energy = new_energy
                    improved = True
                else:
                    # Revert
                    del where[q]
                    chain[i] = p
                    where[p] = i
                    current_energy = old_energy
        if not improved:
            break
    return best_chain, best_energy


def _beam_search(sequence, dimension, beam_width=10, max_depth=None):
    """Beam search over partial conformations, keeping the top beam_width at each step."""
    n = len(sequence)
    if max_depth is None:
        max_depth = n

    # Each beam entry: (chain, occupied_dict, energy)
    start = (0,) * dimension
    beams = [([start], {start: 0}, 0)]

    for i in range(1, max_depth):
        if _remaining_time() < 3.0:
            break
        new_beams = []
        for chain, occupied, energy in beams:
            for site in _neighbours(chain[-1]):
                if site in occupied:
                    continue
                gain = 0
                if sequence[i] == "H":
                    gain = sum(1 for q in _neighbours(site)
                               if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                score = gain + 0.05 * free
                new_chain = chain + [site]
                new_occ = dict(occupied)
                new_occ[site] = i
                new_beams.append((new_chain, new_occ, energy - gain, score))
        if not new_beams:
            break
        # Keep top beam_width by score
        new_beams.sort(key=lambda x: x[3], reverse=True)
        beams = new_beams[:beam_width]
        if not beams:
            break

    if not beams:
        return None
    # Return best energy chain
    best = min(beams, key=lambda x: x[2])
    return best[0]


def fold(sequence, dimension):
    """Best conformation using multiple strategies with time management."""
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))

    best_chain = None
    best_energy = 0  # 0 means no contacts; energy is negative for contacts

    # Strategy 1: Beam search for a good initial solution
    beam_result = _beam_search(sequence, dimension, beam_width=8)
    if beam_result is not None:
        e = _energy(sequence, beam_result)
        if e < best_energy:
            best_energy = e
            best_chain = beam_result

    # Strategy 2: Multiple greedy growths with varying temperatures
    temps = [0.1, 0.2, 0.3, 0.5, 0.8]
    for temp in temps:
        if _remaining_time() < 10.0:
            break
        for _ in range(50):
            if _remaining_time() < 10.0:
                break
            chain = _greedy_grow(sequence, dimension, rng, temperature=temp)
            if chain is None:
                continue
            e = _energy(sequence, chain)
            if e < best_energy:
                best_energy = e
                best_chain = chain

    # Strategy 3: Local search on the best solution found so far
    if best_chain is not None:
        if _remaining_time() > 15.0:
            best_chain, best_energy = _local_search(sequence, dimension, best_chain, max_iters=300)

        # Run local search on a few more random solutions
        if _remaining_time() > 10.0:
            for _ in range(5):
                if _remaining_time() < 10.0:
                    break
                chain = _greedy_grow(sequence, dimension, rng, temperature=0.3)
                if chain is None:
                    continue
                chain, e = _local_search(sequence, dimension, chain, max_iters=200)
                if e < best_energy:
                    best_energy = e
                    best_chain = chain

    # Fallback: straight line
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
