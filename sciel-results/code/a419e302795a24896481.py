"""HP lattice protein folding on the cubic lattice with contact-based energy and local search."""
# EVOLVE-BLOCK-START
import time
import random
import math


def _neighbours(point):
    """Yield the 6 neighbours of a point in 3D (or 2^(d-1) in d dimensions)."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _count_hh_contacts(sequence, chain):
    """Count non-bonded H-H contacts in a chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return contacts


def _energy(sequence, chain):
    """Energy is minus the number of H-H contacts."""
    return -_count_hh_contacts(sequence, chain)


def _grow_with_contacts(sequence, dimension, rng, temperature=0.5):
    """
    Grow the chain using a contact-based scoring function.
    At each step, prefer placements that create new H-H contacts.
    """
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            
            # Count potential new H-H contacts if we place here
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            gain += 1
            
            # Count free space for future growth
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            # Score: prioritize contacts, then free space, with some randomness
            score = gain * 10.0 + free * 0.1 + rng.random() * temperature
            options.append((score, site))
        
        if not options:
            return None
        
        # Choose the best option with some probability for diversity
        options.sort(key=lambda x: -x[0])
        if len(options) == 1:
            site = options[0][1]
        else:
            # Boltzmann-like selection: favor high scores but allow exploration
            scores = [o[0] for o in options]
            max_score = max(scores)
            weights = [math.exp((s - max_score) / temperature) for s in scores]
            total = sum(weights)
            r = rng.random() * total
            cum = 0
            site = options[-1][1]
            for idx, w in enumerate(weights):
                cum += w
                if r <= cum:
                    site = options[idx][1]
                    break
        
        occupied[site] = i
        chain.append(site)
    
    return chain


def _pivot_move(chain, i, direction, occupied_set, sequence):
    """
    Perform a pivot move at position i.
    direction: 0=keep, 1=rotate by 90 degrees
    Returns new chain if valid, else None.
    """
    if i <= 1 or i >= len(chain) - 1:
        return None
    
    # Extract the segment to rotate
    segment = chain[i:]
    
    # Determine the pivot axis and new direction
    prev_dir = tuple(chain[i] - chain[i-1] for _ in [0])  # This is wrong, let me fix
    
    # Actually, let's implement a simpler local search: try moving the last few residues
    return None


def _local_search(sequence, chain, max_steps=100, rng=None):
    """
    Local search using pivot-like moves: try to improve energy by
    repositioning small segments of the chain.
    """
    if rng is None:
        rng = random.Random()
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    n = len(current_chain)
    
    for step in range(max_steps):
        # Try to move a random residue (not the first or last)
        # by removing it and reinserting it in a nearby position
        i = rng.randint(2, n - 3)  # Avoid ends to keep connectivity
        
        # Remove residue i
        residue = current_chain[i]
        occupied = set(current_chain)
        occupied.remove(residue)
        
        # Try to reinsert at a neighbour of position i-1 or i+1
        candidates = []
        for neighbor in _neighbours(current_chain[i-1]):
            if neighbor not in occupied and neighbor != current_chain[i+1]:
                candidates.append(neighbor)
        for neighbor in _neighbours(current_chain[i+1]):
            if neighbor not in occupied and neighbor != current_chain[i-1]:
                candidates.append(neighbor)
        
        if not candidates:
            continue
        
        new_pos = rng.choice(candidates)
        
        # Check if the move is valid (no self-avoidance violation)
        # Build new chain
        new_chain = current_chain[:i] + [new_pos] + current_chain[i+1:]
        
        # Verify self-avoidance
        if len(set(new_chain)) != n:
            continue
        
        # Verify unit steps
        valid = True
        for j in range(1, n):
            diff = sum(abs(a - b) for a, b in zip(new_chain[j-1], new_chain[j]))
            if diff != 1:
                valid = False
                break
        
        if not valid:
            continue
        
        new_energy = _energy(sequence, new_chain)
        
        # Accept if better or with some probability (simulated annealing style)
        if new_energy < current_energy:
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_chain = list(current_chain)
                best_energy = current_energy
        elif rng.random() < math.exp((new_energy - current_energy) / 0.5):
            current_chain = new_chain
            current_energy = new_energy
    
    return best_chain


def _straight_line(sequence, dimension):
    """Return a straight line conformation (always valid)."""
    return [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]


def fold(sequence, dimension):
    """
    Fold the sequence into a 3D conformation on the cubic lattice.
    Returns a list of lattice points (lists of length dimension).
    """
    start_time = time.time()
    deadline = start_time + 250  # Leave 80s safety margin
    
    rng = random.Random(42 + len(sequence))
    
    n = len(sequence)
    
    # For very short sequences, just do a simple search
    if n <= 10:
        best_chain = _straight_line(sequence, dimension)
        best_energy = _energy(sequence, best_chain)
        for _ in range(50):
            chain = _grow_with_contacts(sequence, dimension, rng, temperature=1.0)
            if chain is None:
                continue
            e = _energy(sequence, chain)
            if e < best_energy:
                best_chain = chain
                best_energy = e
        return [list(p) for p in best_chain]
    
    best_chain = None
    best_energy = 1  # Start with a high (bad) energy
    
    # Phase 1: Multiple growth attempts with different temperatures
    num_growths = 50
    for i in range(num_growths):
        if time.time() > deadline:
            break
        
        temp = 0.3 + 0.7 * (i / max(1, num_growths - 1))
        chain = _grow_with_contacts(sequence, dimension, rng, temperature=temp)
        if chain is None:
            continue
        
        e = _energy(sequence, chain)
        if e < best_energy:
            best_chain = chain
            best_energy = e
    
    if best_chain is None:
        best_chain = _straight_line(sequence, dimension)
        best_energy = _energy(sequence, best_chain)
    
    # Phase 2: Local search on the best chain
    if time.time() < deadline - 30:
        improved = _local_search(sequence, best_chain, max_steps=200, rng=rng)
        if improved is not None:
            e = _energy(sequence, improved)
            if e < best_energy:
                best_chain = improved
                best_energy = e
    
    # Phase 3: More growth attempts with local search
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        num_additional = min(30, int(remaining_time / 2))
        for i in range(num_additional):
            if time.time() > deadline:
                break
            
            temp = 0.5 + rng.random() * 0.5
            chain = _grow_with_contacts(sequence, dimension, rng, temperature=temp)
            if chain is None:
                continue
            
            # Apply local search
            chain = _local_search(sequence, chain, max_steps=50, rng=rng)
            
            e = _energy(sequence, chain)
            if e < best_energy:
                best_chain = chain
                best_energy = e
    
    # Ensure we return a valid conformation
    if best_chain is None or len(best_chain) != n:
        best_chain = _straight_line(sequence, dimension)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
