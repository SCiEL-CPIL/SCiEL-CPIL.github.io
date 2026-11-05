"""HP lattice protein folding on the cubic lattice with multi-start local search."""
import time
import random
from collections import defaultdict

# Global deadline management
_START_TIME = time.time()
_DEADLINE_MARGIN = 15  # seconds of safety margin before the 330s kill


def _get_deadline():
    return _START_TIME + 330 - _DEADLINE_MARGIN


def _is_deadline_exceeded():
    return time.time() > _get_deadline()


def _neighbours(point):
    """Yield the 6 neighbours of a 3D lattice point."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute the energy (negative number of H-H contacts)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _initial_grow(sequence, dimension, rng):
    """One randomized greedy chain growth."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
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
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _local_search(sequence, chain, max_iter=5000, deadline=None):
    """
    Local search using single-residue moves (reinsert a residue at a nearby empty site).
    Returns the improved chain.
    """
    if deadline is None:
        deadline = _get_deadline()
    
    n = len(sequence)
    where = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    # Precompute H positions for faster contact counting
    h_indices = set(i for i, r in enumerate(sequence) if r == 'H')
    
    for iteration in range(max_iter):
        if _is_deadline_exceeded():
            break
        
        # Pick a random internal residue to move
        i = rng.randint(1, n - 2)
        old_pos = chain[i]
        
        # Try all empty neighbours of old_pos and also some random nearby sites
        candidate_sites = []
        for site in _neighbours(old_pos):
            if site not in where:
                candidate_sites.append(site)
        # Add a few random sites near the old position
        for _ in range(3):
            dx = rng.randint(-2, 2)
            dy = rng.randint(-2, 2)
            dz = rng.randint(-2, 2)
            candidate_site = (old_pos[0] + dx, old_pos[1] + dy, old_pos[2] + dz)
            if candidate_site not in where and candidate_site != old_pos:
                candidate_sites.append(candidate_site)
        
        if not candidate_sites:
            continue
        
        # Pick a random candidate site
        new_pos = rng.choice(candidate_sites)
        
        # Check if the move is valid (no self-intersection)
        # We need to check that new_pos is not occupied by any other residue
        # and that moving residue i to new_pos doesn't create a collision
        # with its immediate neighbors in the chain
        
        # Remove residue i from its old position
        del where[old_pos]
        chain[i] = new_pos
        where[new_pos] = i
        
        # Check validity: new_pos must not be adjacent to chain[i-1] or chain[i+1]
        # in a way that would create a non-consecutive contact issue
        # Actually, the main constraint is self-avoidance, which we already checked
        # by ensuring new_pos is not in where (before adding it back)
        
        # Compute new energy
        new_energy = _energy(sequence, chain)
        
        if new_energy < current_energy:
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(chain)
        else:
            # Revert the move
            del where[new_pos]
            chain[i] = old_pos
            where[old_pos] = i
    
    return best_chain


def fold(sequence, dimension):
    """
    Find the best conformation of the given sequence using multi-start local search.
    Returns a list of lattice points (lists of integers).
    """
    global _START_TIME
    _START_TIME = time.time()
    deadline = _get_deadline()
    
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0  # We want to minimize energy (more negative is better)
    
    # Multi-start: generate initial chains and improve them
    num_starts = 50
    for start_idx in range(num_starts):
        if _is_deadline_exceeded():
            break
        
        # Generate an initial chain
        chain = _initial_grow(sequence, dimension, rng)
        if chain is None:
            # Try a simpler growth strategy
            chain = _simple_grow(sequence, dimension, rng)
            if chain is None:
                continue
        
        # Compute initial energy
        energy = _energy(sequence, chain)
        
        # Improve with local search
        if energy < best_energy or best_chain is None:
            improved = _local_search(sequence, chain, max_iter=3000, deadline=deadline)
            improved_energy = _energy(sequence, improved)
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = improved
        
        # If we're doing well, do more local search iterations
        if best_chain is not None and _is_deadline_exceeded():
            break
    
    # Final improvement pass if we have time
    if best_chain is not None and not _is_deadline_exceeded():
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            final = _local_search(sequence, best_chain, max_iter=10000, deadline=deadline)
            final_energy = _energy(sequence, final)
            if final_energy < best_energy:
                best_energy = final_energy
                best_chain = final
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]


def _simple_grow(sequence, dimension, rng):
    """A simpler growth strategy that always tries to grow in a preferred direction."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    
    # Prefer directions that keep the chain compact
    preferred_dirs = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1)]
    
    for i in range(1, len(sequence)):
        # Shuffle directions with some bias toward compact growth
        dirs = list(preferred_dirs)
        rng.shuffle(dirs)
        
        placed = False
        for d in dirs:
            new_pos = (chain[-1][0] + d[0], chain[-1][1] + d[1], chain[-1][2] + d[2])
            if new_pos not in occupied:
                occupied[new_pos] = i
                chain.append(new_pos)
                placed = True
                break
        
        if not placed:
            return None
    
    return chain


# Re-initialize the start time when the module is loaded
_START_TIME = time.time()
