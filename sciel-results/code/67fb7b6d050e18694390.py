"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

_START_TIME = time.time()
_TIME_BUDGET = 300.0  # seconds, with safety margin

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _build_occupied(chain):
    return {tuple(p): i for i, p in enumerate(chain)}

def _energy(sequence, chain):
    """Compute energy as negative count of H-H contacts (non-adjacent in chain)."""
    where = {tuple(p): i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _try_move(chain, i, new_pos, occupied):
    """Try to move residue i to new_pos, return True if valid."""
    old_pos = tuple(chain[i])
    if new_pos == old_pos:
        return False
    if new_pos in occupied:
        return False
    # Check that new position doesn't conflict with neighbors in chain
    for j in (i - 1, i + 1):
        if 0 <= j < len(chain):
            if tuple(chain[j]) == new_pos:
                return False
    # Verify all consecutive pairs remain unit steps
    if i > 0:
        d = [abs(new_pos[k] - tuple(chain[i-1])[k]) for k in range(len(new_pos))]
        if sum(d) != 1:
            return False
    if i < len(chain) - 1:
        d = [abs(new_pos[k] - tuple(chain[i+1])[k]) for k in range(len(new_pos))]
        if sum(d) != 1:
            return False
    return True

def _local_search(sequence, chain, rng, max_iters=5000):
    """Local search with single-residue moves."""
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    occupied = _build_occupied(best_chain)
    n = len(sequence)
    
    for _ in range(max_iters):
        if time.time() - _START_TIME > _TIME_BUDGET - 10:
            break
        # Pick a random residue to move
        i = rng.randint(0, n - 1)
        old_pos = tuple(best_chain[i])
        # Try moving to a neighbor
        candidates = []
        for new_pos in _neighbours(old_pos):
            if new_pos in occupied:
                continue
            # Check if move is valid (preserves chain connectivity)
            if i > 0:
                d = [abs(new_pos[k] - tuple(best_chain[i-1])[k]) for k in range(len(new_pos))]
                if sum(d) != 1:
                    continue
            if i < n - 1:
                d = [abs(new_pos[k] - tuple(best_chain[i+1])[k]) for k in range(len(new_pos))]
                if sum(d) != 1:
                    continue
            candidates.append(new_pos)
        
        if not candidates:
            continue
        
        new_pos = rng.choice(candidates)
        # Compute energy change
        # Remove old contacts involving i
        old_contacts = 0
        if sequence[i] == 'H':
            for q in _neighbours(old_pos):
                j = occupied.get(q)
                if j is not None and j != i - 1 and j != i + 1 and sequence[j] == 'H':
                    old_contacts += 1
        # Add new contacts
        new_contacts = 0
        if sequence[i] == 'H':
            for q in _neighbours(new_pos):
                j = occupied.get(q)
                if j is not None and j != i - 1 and j != i + 1 and sequence[j] == 'H':
                    new_contacts += 1
        
        delta = -(new_contacts - old_contacts)
        if delta < 0 or rng.random() < 0.3:
            # Accept move
            occupied.pop(old_pos)
            best_chain[i] = list(new_pos)
            occupied[new_pos] = i
            best_energy += delta
    
    return best_chain, best_energy

def _initial_grow(sequence, dimension, rng):
    """Greedy growth with contact awareness."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
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
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        
        if not options:
            return None
        
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    
    return chain

def fold(sequence, dimension):
    """Fold the sequence using multiple strategies and local search."""
    if time.time() - _START_TIME > _TIME_BUDGET - 5:
        # Emergency: return a straight line
        return [[i] + [0] * (dimension - 1) for i in range(len(sequence))]
    
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0
    
    # Strategy 1: Multiple greedy growths + local search
    num_growths = min(50, max(10, 300 // n))
    for _ in range(num_growths):
        if time.time() - _START_TIME > _TIME_BUDGET - 15:
            break
        chain = _initial_grow(sequence, dimension, rng)
        if chain is None:
            continue
        chain, energy = _local_search(sequence, chain, rng, max_iters=2000)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
    
    # Strategy 2: Try a few more local search passes on the best
    if best_chain is not None:
        for _ in range(3):
            if time.time() - _START_TIME > _TIME_BUDGET - 10:
                break
            best_chain, best_energy = _local_search(sequence, best_chain, rng, max_iters=3000)
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [[i] + [0] * (dimension - 1) for i in range(n)]
    
    return best_chain
# EVOLVE-BLOCK-END
