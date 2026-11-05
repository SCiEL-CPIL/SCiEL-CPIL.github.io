"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time


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


def _grow(sequence, dimension, rng, temperature=1.0):
    """One randomized greedy chain growth with simulated annealing-like acceptance."""
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
            score = gain + 0.1 * free + rng.random() * temperature
            options.append((score, site))
        
        if not options:
            return None
        
        # Sort by score and pick with some randomness
        options.sort(reverse=True)
        # Pick from top candidates with probability
        if rng.random() < 0.7:
            site = options[0][1]
        elif len(options) > 1 and rng.random() < 0.9:
            site = options[1][1]
        else:
            site = options[min(len(options) - 1, 2)][1]
        
        occupied[site] = i
        chain.append(site)
    
    return chain


def _local_search(sequence, chain, rng, max_iter=5000):
    """Local search using single-residue moves (pivot moves)."""
    n = len(chain)
    if n < 3:
        return chain
    
    best_chain = list(chain)
    best_energy = _energy(sequence, chain)
    
    # Precompute occupied set for fast lookup
    occupied = {p: i for i, p in enumerate(best_chain)}
    
    for _ in range(max_iter):
        # Pick a random residue to move (not the first or last)
        i = rng.randint(1, n - 2)
        old_pos = best_chain[i]
        
        # Try moving this residue to a neighboring position
        moved = False
        for new_pos in _neighbours(old_pos):
            # Check if new position is available
            if new_pos in occupied:
                continue
            
            # Check that the move doesn't create conflicts with adjacent residues
            # The residue at i-1 and i+1 must still be adjacent to the new position
            # Actually, we need to check that the chain remains self-avoiding
            # and that the move is valid (unit steps preserved)
            
            # For a single residue move, we need to check:
            # 1. new_pos is not occupied by any other residue
            # 2. The residue at i-1 is adjacent to new_pos (or i=1, then start is adjacent)
            # 3. The residue at i+1 is adjacent to new_pos (or i=n-2, then end is adjacent)
            
            if i > 0:
                prev_pos = best_chain[i - 1]
                if abs(prev_pos[0] - new_pos[0]) + abs(prev_pos[1] - new_pos[1]) != 1:
                    continue
            
            if i < n - 1:
                next_pos = best_chain[i + 1]
                if abs(next_pos[0] - new_pos[0]) + abs(next_pos[1] - new_pos[1]) != 1:
                    continue
            
            # Check self-avoidance: new_pos must not be occupied by any other residue
            # We already checked new_pos not in occupied, but we need to make sure
            # that moving residue i doesn't cause overlap with itself (it won't since
            # we're moving it) and that the path remains valid
            
            # Perform the move
            occupied.pop(old_pos)
            best_chain[i] = new_pos
            occupied[new_pos] = i
            
            new_energy = _energy(sequence, best_chain)
            
            if new_energy < best_energy:
                best_energy = new_energy
                moved = True
                break
            else:
                # Revert
                occupied.pop(new_pos)
                best_chain[i] = old_pos
                occupied[old_pos] = i
        
        if not moved:
            # Try a different strategy: 2-residue moves or accept worse moves with probability
            pass
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation found within time budget."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 1
    
    n = len(sequence)
    
    # Phase 1: Greedy growth with multiple restarts
    n_restarts = 500
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Vary temperature and randomness
        temp = 0.5 + rng.random() * 1.5
        chain = _grow(sequence, dimension, rng, temperature=temp)
        if chain is None:
            continue
        
        value = _energy(sequence, chain)
        if value < best_energy:
            best_chain = chain
            best_energy = value
    
    # Phase 2: Local search on best chain
    if best_chain is not None and time.time() < deadline:
        local_iter = 20000 if n <= 30 else 10000
        best_chain = _local_search(sequence, best_chain, rng, max_iter=local_iter)
        best_energy = _energy(sequence, best_chain)
    
    # Phase 3: More local search iterations if time permits
    if best_chain is not None and time.time() < deadline - 5:
        best_chain = _local_search(sequence, best_chain, rng, max_iter=10000)
        best_energy = _energy(sequence, best_chain)
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
