"""Rewritten program for HP lattice protein folding using Hamiltonian path templates."""
# EVOLVE-BLOCK-START
import random
import time
import sys

def _neighbours(point):
    """Yield the 4 neighbours of a point on the square lattice."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _energy(sequence, chain):
    """Compute the HP contact energy of a chain conformation.
    Energy = - (number of H-H contacts) where contacts are non-adjacent H residues on neighbouring lattice sites.
    Returns a negative number (more negative = better)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _build_occupied_dict(chain):
    """Build a dictionary mapping position to index for O(1) lookups."""
    return {p: i for i, p in enumerate(chain)}

def _try_single_move(sequence, chain, occupied_dict, i):
    """Try to move residue i to an adjacent empty position.
    Returns (new_chain, new_energy, delta) if valid and better, else None."""
    n = len(chain)
    if i < 1 or i > n - 2:
        return None
    
    old_pos = chain[i]
    for new_pos in _neighbours(old_pos):
        if new_pos in occupied_dict:
            continue
        
        # Check adjacency constraints
        dist1 = abs(new_pos[0] - chain[i-1][0]) + abs(new_pos[1] - chain[i-1][1])
        dist2 = abs(new_pos[0] - chain[i+1][0]) + abs(new_pos[1] - chain[i+1][1])
        if dist1 != 1 or dist2 != 1:
            continue
        
        # Valid move
        new_chain = list(chain)
        new_chain[i] = new_pos
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - _energy(sequence, chain)
        return (new_chain, new_energy, delta)
    
    return None

def _local_search_sa(sequence, chain, max_iters=500, rng=None):
    """Simulated annealing local search with efficient occupancy tracking."""
    if rng is None:
        rng = random.Random()
    
    n = len(sequence)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    occupied_dict = _build_occupied_dict(current_chain)
    
    temperature = 5.0
    cooling = 0.9995
    
    for it in range(max_iters):
        if time.time() > _local_search_sa.deadline:
            break
        
        i = rng.randint(1, n - 2)
        result = _try_single_move(sequence, current_chain, occupied_dict, i)
        
        if result is None:
            continue
        
        new_chain, new_energy, delta = result
        
        # Accept if better or with probability based on temperature
        if delta < 0 or rng.random() < pow(2.718281828, -delta / temperature):
            # Update occupancy dict
            old_pos = current_chain[i]
            new_pos = new_chain[i]
            del occupied_dict[old_pos]
            occupied_dict[new_pos] = i
            
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        
        temperature *= cooling
    
    return best_chain, best_energy

def _segment_reversal(sequence, chain, rng, max_attempts=10):
    """Try to reverse segments of the chain to improve energy."""
    n = len(sequence)
    if n < 4:
        return list(chain), _energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    
    for _ in range(max_attempts):
        # Pick two indices to reverse the segment between them
        i = rng.randint(1, n - 3)
        j = rng.randint(i + 2, n - 1)
        
        # Create a new chain with the segment reversed
        new_chain = list(chain)
        new_chain[i+1:j+1] = reversed(chain[i+1:j+1])
        
        # Check self-avoidance
        occupied = set(new_chain)
        if len(occupied) != n:
            continue
        
        # Check unit steps
        valid = True
        for k in range(n - 1):
            d = abs(new_chain[k+1][0] - new_chain[k][0]) + abs(new_chain[k+1][1] - new_chain[k][1])
            if d != 1:
                valid = False
                break
        
        if not valid:
            continue
        
        new_energy = _energy(sequence, new_chain)
        if new_energy < best_energy:
            best_energy = new_energy
            best_chain = new_chain
    
    return best_chain, best_energy

def _polish(sequence, chain, rng, time_budget=10.0):
    """Final polishing phase with multiple local search passes."""
    start_time = time.time()
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    while time.time() - start_time < time_budget:
        # Alternate between single moves and segment reversals
        if rng.random() < 0.7:
            new_chain, new_energy = _local_search_sa(sequence, current_chain, max_iters=100, rng=rng)
        else:
            new_chain, new_energy = _segment_reversal(sequence, current_chain, rng, max_attempts=5)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_chain = list(new_chain)
            current_chain = new_chain
        else:
            # No improvement, break to save time
            break
    
    return best_chain, best_energy

def _diverse_init(sequence, rng):
    """Generate diverse initial conformations using different strategies."""
    n = len(sequence)
    inits = []
    
    # Strategy 1: Straight line
    inits.append([(i, 0) for i in range(n)])
    
    # Strategy 2: Serpentine
    chain = []
    row = 0
    col = 0
    direction = 1
    for i in range(n):
        chain.append((col, row))
        col += direction
        if col >= n - row or col <= row:
            row += 1
            direction = -direction
            col = row if direction == 1 else n - 1 - row
    inits.append(chain)
    
    # Strategy 3: Random walk with backtracking avoidance
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    direction = (1, 0)
    for i in range(1, n):
        x, y = chain[-1]
        dx, dy = direction
        candidates = []
        for ndx, ndy in [(dx, dy), (dy, -dx), (-dx, -dy), (-dy, dx)]:
            new_pos = (x + ndx, y + ndy)
            if new_pos not in occupied:
                candidates.append((ndx, ndy))
        
        if not candidates:
            break
        
        if rng.random() < 0.5:
            ndx, ndy = rng.choice(candidates)
        else:
            ndx, ndy = candidates[0]
        
        direction = (ndx, ndy)
        new_pos = (x + ndx, y + ndy)
        chain.append(new_pos)
        occupied[new_pos] = i
    
    if len(chain) == n:
        inits.append(chain)
    
    # Strategy 4: Spiral
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    dx, dy = 1, 0
    steps = 1
    for i in range(1, n):
        x, y = chain[-1]
        new_pos = (x + dx, y + dy)
        if new_pos in occupied:
            # Turn right
            dx, dy = dy, -dx
            new_pos = (x + dx, y + dy)
            steps += 1
            if steps > 2:
                steps = 1
                dx, dy = dy, -dx
                new_pos = (x + dx, y + dy)
        
        if new_pos in occupied:
            break
        
        chain.append(new_pos)
        occupied[new_pos] = i
    
    if len(chain) == n:
        inits.append(chain)
    
    return inits

def fold(sequence, dimension):
    """Fold a sequence into a low-energy self-avoiding conformation.
    Uses diverse initialization, simulated annealing, and segment reversal."""
    start_time = time.time()
    deadline = start_time + 175.0  # Safety margin before 180s limit
    n = len(sequence)
    
    rng = random.Random(42 + n)
    
    # Set deadline for local search
    _local_search_sa.deadline = deadline - 5.0
    
    best_chain = None
    best_energy = 0  # Start with unfolded (energy 0)
    
    # Phase 1: Generate diverse initial conformations
    inits = _diverse_init(sequence, rng)
    
    for init_chain in inits:
        if time.time() > deadline - 10.0:
            break
        
        energy = _energy(sequence, init_chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = list(init_chain)
    
    # Phase 2: Simulated annealing on best initial
    if best_chain is not None and time.time() < deadline - 20.0:
        improved_chain, improved_energy = _local_search_sa(
            sequence, best_chain, max_iters=2000, rng=rng
        )
        if improved_energy < best_energy:
            best_energy = improved_energy
            best_chain = improved_chain
    
    # Phase 3: Polish with alternating moves
    if best_chain is not None and time.time() < deadline - 5.0:
        final_chain, final_energy = _polish(
            sequence, best_chain, rng, time_budget=15.0
        )
        if final_energy < best_energy:
            best_energy = final_energy
            best_chain = final_chain
    
    # Fallback: if no valid chain was found, return a straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    # Convert to list of lists as required
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
