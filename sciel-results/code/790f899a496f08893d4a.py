"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

_DEADLINE = time.time() + 325  # safety margin before 330s kill


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


def _grow(sequence, dimension, rng, temperature=0.5):
    """One randomized greedy chain growth with temperature-controlled exploration."""
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
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _is_valid_1move(new_chain, dimension):
    """Check if a 1-residue move produces a valid self-avoiding connected chain."""
    n = len(new_chain)
    if n != len(set(new_chain)):
        return False
    for j in range(n - 1):
        diff = sum(abs(new_chain[j][a] - new_chain[j+1][a]) for a in range(dimension))
        if diff != 1:
            return False
    return True


def _is_valid_kink(new_chain, dimension):
    """Check if a 2-residue kink move produces a valid self-avoiding connected chain."""
    n = len(new_chain)
    if n != len(set(new_chain)):
        return False
    for j in range(n - 1):
        diff = sum(abs(new_chain[j][a] - new_chain[j+1][a]) for a in range(dimension))
        if diff != 1:
            return False
    return True


def _local_refine(sequence, chain, dimension, rng, max_steps=100):
    """Local search with 1-residue moves and 2-residue kink moves."""
    current_energy = _energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    for step in range(max_steps):
        if time.time() > _DEADLINE:
            break
        
        n = len(chain)
        if n < 3:
            break
        
        moved = False
        
        # Try 1-residue move at a random internal position
        i = rng.randint(1, n - 2)
        
        occupied = set(chain)
        occupied.discard(chain[i])
        
        for site in _neighbours(chain[i]):
            if site in occupied:
                continue
            
            new_chain = chain[:]
            new_chain[i] = site
            
            if not _is_valid_1move(new_chain, dimension):
                continue
            
            new_energy = _energy(sequence, new_chain)
            if new_energy < best_energy:
                best_chain = new_chain
                best_energy = new_energy
                chain = new_chain
                moved = True
                break
        
        if moved:
            continue
        
        # Try 2-residue kink move: move residues i and i+1 together
        if n >= 4:
            i = rng.randint(1, n - 3)
            
            # Get the two residues to move
            p1, p2 = chain[i], chain[i+1]
            
            # The kink can rotate in different ways
            # For simplicity, try moving both residues to new positions
            # that maintain the unit step between them
            
            # Direction from p1 to p2
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1] if dimension > 1 else 0
            dz = p2[2] - p1[2] if dimension > 2 else 0
            
            # Generate possible new positions for the kink
            # The kink must maintain unit steps: new_p1 to new_p2 must be unit distance
            # and new_p1 must be adjacent to chain[i-1], new_p2 must be adjacent to chain[i+2]
            
            prev_point = chain[i-1]
            next_point = chain[i+2]
            
            # Find all valid positions for new_p1 that are adjacent to prev_point
            # and not occupied (except by p1 and p2 which we're moving)
            occupied = set(chain)
            occupied.discard(p1)
            occupied.discard(p2)
            
            candidates = []
            for new_p1 in _neighbours(prev_point):
                if new_p1 in occupied:
                    continue
                # new_p2 must be adjacent to new_p1 and next_point
                for new_p2 in _neighbours(new_p1):
                    if new_p2 in occupied:
                        continue
                    # Check if new_p2 is adjacent to next_point
                    diff = sum(abs(new_p2[a] - next_point[a]) for a in range(dimension))
                    if diff != 1:
                        continue
                    candidates.append((new_p1, new_p2))
            
            if candidates:
                new_p1, new_p2 = rng.choice(candidates)
                new_chain = chain[:]
                new_chain[i] = new_p1
                new_chain[i+1] = new_p2
                
                if _is_valid_kink(new_chain, dimension):
                    new_energy = _energy(sequence, new_chain)
                    if new_energy < best_energy:
                        best_chain = new_chain
                        best_energy = new_energy
                        chain = new_chain
                        moved = True
        
        if not moved:
            # Try a few more random positions before giving up
            for _ in range(5):
                i = rng.randint(1, n - 2)
                occupied = set(chain)
                occupied.discard(chain[i])
                
                for site in _neighbours(chain[i]):
                    if site in occupied:
                        continue
                    
                    new_chain = chain[:]
                    new_chain[i] = site
                    
                    if not _is_valid_1move(new_chain, dimension):
                        continue
                    
                    new_energy = _energy(sequence, new_chain)
                    if new_energy < best_energy:
                        best_chain = new_chain
                        best_energy = new_energy
                        chain = new_chain
                        moved = True
                        break
                if moved:
                    break
        
        if not moved:
            break
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation using many randomized greedy growths with local refinement."""
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 1
    
    # Scale number of attempts based on chain length
    if n <= 48:
        num_attempts = 600
    elif n <= 80:
        num_attempts = 400
    else:
        num_attempts = 200
    
    for attempt in range(num_attempts):
        if time.time() > _DEADLINE:
            break
        
        # Vary temperature for diversity
        temp = 0.3 + 0.4 * rng.random()
        chain = _grow(sequence, dimension, rng, temperature=temp)
        if chain is None:
            continue
        
        value = _energy(sequence, chain)
        
        if value < best_energy:
            best_chain = chain
            best_energy = value
        
        # Do refinement on promising candidates
        if value <= best_energy + 3:
            refined = _local_refine(sequence, chain, dimension, rng, max_steps=50)
            refined_energy = _energy(sequence, refined)
            if refined_energy < best_energy:
                best_chain = refined
                best_energy = refined_energy
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
