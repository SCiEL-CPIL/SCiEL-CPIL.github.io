"""Rewritten program for HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Global deadline guard
_TOTAL_START = time.time()
_HARD_LIMIT = 320.0  # seconds, leave 10s margin before 330s kill


def _remaining_time():
    """Return remaining seconds before our safety deadline."""
    return _TOTAL_START + _HARD_LIMIT - time.time()


def _neighbours(point):
    """Generate the 6 neighbours of a point in 3D cubic lattice."""
    x, y, z = point
    yield (x + 1, y, z)
    yield (x - 1, y, z)
    yield (x, y + 1, z)
    yield (x, y - 1, z)
    yield (x, y, z + 1)
    yield (x, y, z - 1)


def _energy(sequence, chain):
    """Compute energy as minus the number of H-H non-bonded contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _grow(sequence, dimension, rng, bias=0.0):
    """
    One randomized greedy chain growth.
    bias: higher values prefer more contacts over free space.
    Returns None if chain gets stuck.
    """
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Count new H-H contacts this site would form
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            # Count free neighbours (space to grow)
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            # Score: prefer contacts, then free space, then random
            score = gain * (1.0 + bias) + 0.05 * free + rng.random() * 0.1
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick the best option
        best_score = max(opt[0] for opt in options)
        # Occasionally pick a non-best option for diversity
        if rng.random() < 0.15 and len(options) > 1:
            # Pick randomly from top 3
            top = sorted(options, key=lambda x: -x[0])[:3]
            site = top[rng.randint(0, len(top) - 1)][1]
        else:
            site = max(options, key=lambda x: x[0])[1]
        
        occupied[site] = i
        chain.append(site)
    
    return chain


def _local_search(sequence, chain, max_steps=500, rng=None):
    """
    Local search via random moves: try moving a single residue to an adjacent
    empty site, accepting if energy improves or with simulated annealing.
    """
    if rng is None:
        rng = random.Random()
    
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    # Simulated annealing parameters
    temperature = max(0.5, abs(current_energy) * 0.1)
    cooling = 0.995
    
    steps_done = 0
    while steps_done < max_steps and _remaining_time() > 5.0:
        # Pick a random internal residue to move
        i = rng.randint(1, n - 2)
        old_pos = chain[i]
        
        # Try each neighbour of old_pos
        moved = False
        for new_pos in _neighbours(old_pos):
            if new_pos in where:
                continue
            # Check if moving creates valid chain (no self-intersection)
            # The move is valid if new_pos is not occupied by any other residue
            # and the chain remains self-avoiding
            # Since we're just moving one residue, we need to ensure new_pos
            # is not where any other residue is
            if new_pos in where:
                continue
            
            # Temporarily move
            del where[old_pos]
            where[new_pos] = i
            chain[i] = new_pos
            
            # Compute new energy
            new_energy = _energy(sequence, chain)
            
            # Check if this is an improvement
            if new_energy < current_energy:
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(chain)
                moved = True
                break
            else:
                # Simulated annealing: accept worse with probability exp(-delta/T)
                delta = new_energy - current_energy
                if delta < 0:
                    # This should not happen since we check new_energy < current_energy above
                    pass
                else:
                    prob = math.exp(-delta / max(temperature, 0.01))
                    if rng.random() < prob:
                        current_energy = new_energy
                        moved = True
                        break
                    else:
                        # Revert
                        del where[new_pos]
                        where[old_pos] = i
                        chain[i] = old_pos
        
        if not moved:
            # Revert if we tried and failed
            # (already reverted in the loop if not moved)
            pass
        
        temperature *= cooling
        steps_done += 1
    
    return best_chain, best_energy


def _grow_and_optimize(sequence, dimension, rng, num_restarts=50):
    """
    Multiple growth restarts with local search on the best ones.
    """
    best_chain = None
    best_energy = 0  # energy is negative, so 0 means no valid chain yet
    
    n = len(sequence)
    # Scale number of restarts based on chain length
    if n <= 48:
        num_restarts = 100
    elif n <= 80:
        num_restarts = 60
    else:
        num_restarts = 40
    
    for restart in range(num_restarts):
        if _remaining_time() < 5.0:
            break
        
        # Vary the bias for diversity
        bias = rng.random() * 0.5
        
        chain = _grow(sequence, dimension, rng, bias=bias)
        if chain is None:
            continue
        
        energy = _energy(sequence, chain)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = list(chain)
        
        # Do local search on promising chains
        if _remaining_time() > 10.0:
            # Only do local search if the chain is reasonably good
            if energy < -n * 0.2:
                optimized_chain, optimized_energy = _local_search(
                    sequence, chain, max_steps=200, rng=rng
                )
                if optimized_energy < best_energy:
                    best_energy = optimized_energy
                    best_chain = optimized_chain
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return best_chain


def fold(sequence, dimension):
    """
    Best conformation found within the time budget.
    Returns a list of len(sequence) integer lattice points (lists of length dimension).
    """
    if dimension != 3:
        # Only handle 3D as per spec
        dimension = 3
    
    n = len(sequence)
    rng = random.Random(12345 + n)
    
    # For very short chains, just do many restarts
    if n <= 10:
        best_chain = None
        best_energy = 0
        for _ in range(500):
            chain = _grow(sequence, dimension, rng, bias=0.0)
            if chain is None:
                continue
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = list(chain)
        if best_chain is None:
            best_chain = [(i, 0, 0) for i in range(n)]
        return [list(p) for p in best_chain]
    
    # Main search
    best_chain = _grow_and_optimize(sequence, dimension, rng)
    
    # Final local search on the best chain
    if _remaining_time() > 3.0:
        final_chain, final_energy = _local_search(
            sequence, best_chain, max_steps=300, rng=rng
        )
        best_chain = final_chain
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
