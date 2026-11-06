"""HP lattice protein folding using simulated annealing with distance-matrix pre-computation."""
# EVOLVE-BLOCK-START
import random
import math


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _initial_conformation(sequence, dimension):
    """Create a simple self-avoiding initial conformation (spiral or straight line)."""
    n = len(sequence)
    if n <= 1:
        return [(0, 0)]
    
    # Try to make a compact spiral-like initial conformation
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    direction = 0  # 0=right, 1=down, 2=left, 3=up
    steps = 0
    turn_count = 0
    target_turns = 0
    
    for i in range(1, n):
        # Determine when to turn (spiral pattern)
        if steps == 1 and turn_count == 0:
            direction = 1  # turn down
            turn_count += 1
        elif steps == 1 and turn_count == 1:
            direction = 2  # turn left
            turn_count += 1
        elif steps == 2 and turn_count == 2:
            direction = 3  # turn up
            turn_count += 1
        elif steps == 2 and turn_count == 3:
            direction = 0  # turn right
            turn_count += 1
        elif steps == 3 and turn_count == 4:
            direction = 1
            turn_count += 1
        elif steps == 3 and turn_count == 5:
            direction = 2
            turn_count += 1
        elif steps == 4 and turn_count == 6:
            direction = 3
            turn_count += 1
        elif steps == 4 and turn_count == 7:
            direction = 0
            turn_count += 1
        
        dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        dx, dy = dirs[direction]
        new_pos = (chain[-1][0] + dx, chain[-1][1] + dy)
        
        if new_pos in occupied:
            # Try all directions
            found = False
            for d in range(4):
                dx2, dy2 = dirs[d]
                new_pos2 = (chain[-1][0] + dx2, chain[-1][1] + dy2)
                if new_pos2 not in occupied:
                    new_pos = new_pos2
                    direction = d
                    found = True
                    break
            if not found:
                # Fallback: just extend in a straight line
                new_pos = (chain[-1][0] + 1, 0)
                while new_pos in occupied:
                    new_pos = (new_pos[0] + 1, 0)
        
        chain.append(new_pos)
        occupied[new_pos] = i
        steps += 1
        
        # Reset steps when we turn
        if turn_count > 0 and (steps >= 1 if turn_count % 2 == 1 else steps >= 2):
            if turn_count % 2 == 1 and steps == 1:
                steps = 0
            elif turn_count % 2 == 0 and steps == 2:
                steps = 0
    
    return chain


def _compute_energy(sequence, chain):
    """Compute the energy of a conformation."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _try_move(chain, sequence, i, j, rng):
    """
    Try to move a segment of the chain from position i to j.
    Returns (new_chain, energy_delta) or (None, 0) if move is invalid.
    """
    n = len(chain)
    if i == j:
        return None, 0
    
    # Ensure i < j
    if i > j:
        i, j = j, i
    
    # The segment to move is from i to j inclusive
    # We need to check if moving this segment creates self-avoidance violations
    
    # Build occupied set excluding the segment
    segment = chain[i:j+1]
    occupied = set()
    for k in range(n):
        if k < i or k > j:
            occupied.add(chain[k])
    
    # Try to place the segment in a new position
    # The segment must maintain unit steps between consecutive residues
    # and must not overlap with the rest of the chain
    
    # Generate a new position for the segment
    # The first residue of the segment must be adjacent to chain[i-1] (if i > 0)
    # The last residue of the segment must be adjacent to chain[j+1] (if j < n-1)
    
    # For simplicity, try small local moves: move one residue at a time
    # This is a simpler approach that still allows exploration
    
    # Actually, let's try a different approach: try to move a single residue
    # to a nearby position, maintaining self-avoidance
    
    # Let's try moving residue i to a random nearby position
    # This is a "kink" move
    
    # Pick a random residue to move
    idx = rng.randint(0, n - 1)
    
    # Get the current position
    current_pos = chain[idx]
    
    # Try to move to a nearby position (within a small radius)
    for _ in range(10):
        dx = rng.randint(-2, 2)
        dy = rng.randint(-2, 2)
        new_pos = (current_pos[0] + dx, current_pos[1] + dy)
        
        # Check if new position is valid
        # 1. Not occupied by another residue
        # 2. If idx > 0, new_pos must be adjacent to chain[idx-1]
        # 3. If idx < n-1, new_pos must be adjacent to chain[idx+1]
        
        # Build occupied set excluding idx
        occupied = set()
        for k in range(n):
            if k != idx:
                occupied.add(chain[k])
        
        if new_pos in occupied:
            continue
        
        # Check adjacency constraints
        valid = True
        if idx > 0:
            prev = chain[idx - 1]
            if abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) != 1:
                valid = False
        if valid and idx < n - 1:
            next_pos = chain[idx + 1]
            if abs(new_pos[0] - next_pos[0]) + abs(new_pos[1] - next_pos[1]) != 1:
                valid = False
        
        if not valid:
            continue
        
        # This move is valid! Create new chain
        new_chain = chain.copy()
        new_chain[idx] = new_pos
        
        # Compute energy delta
        old_energy = _compute_energy(sequence, chain)
        new_energy = _compute_energy(sequence, new_chain)
        delta = new_energy - old_energy
        
        return new_chain, delta
    
    return None, 0


def _simulated_annealing(sequence, dimension, rng, max_iterations=5000):
    """
    Perform simulated annealing to find a low-energy conformation.
    """
    n = len(sequence)
    if n <= 1:
        return [(0, 0)]
    
    # Start with an initial conformation
    chain = _initial_conformation(sequence, dimension)
    current_energy = _compute_energy(sequence, chain)
    best_chain = chain.copy()
    best_energy = current_energy
    
    # Annealing parameters
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    iteration = 0
    while T > min_T and iteration < max_iterations:
        # Try a move
        new_chain, delta = _try_move(chain, sequence, 0, 0, rng)
        
        if new_chain is None:
            iteration += 1
            T *= cooling_rate
            continue
        
        # Acceptance criterion
        if delta < 0:
            # Always accept improvements
            chain = new_chain
            current_energy += delta
        else:
            # Accept with probability exp(-delta / T)
            if rng.random() < math.exp(-delta / T):
                chain = new_chain
                current_energy += delta
        
        # Update best
        if current_energy < best_energy:
            best_energy = current_energy
            best_chain = chain.copy()
        
        iteration += 1
        T *= cooling_rate
    
    return best_chain


def fold(sequence, dimension):
    """
    Find the best conformation for the given sequence using simulated annealing.
    """
    n = len(sequence)
    if n <= 1:
        return [[0, 0]]
    
    # Use multiple restarts with different random seeds
    best_chain = None
    best_energy = float('inf')
    
    num_restarts = 5
    for restart in range(num_restarts):
        rng = random.Random(12345 + len(sequence) + restart)
        chain = _simulated_annealing(sequence, dimension, rng, max_iterations=3000)
        energy = _compute_energy(sequence, chain)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    # If no valid chain was found, use a straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
