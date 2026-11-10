"""HP lattice protein folding using Iterated Local Search with incremental energy updates."""
import random
import math
import time

def _initial_spiral(sequence, dimension):
    """Create a compact initial conformation using a spiral-like pattern."""
    n = len(sequence)
    if n <= 1:
        return [(0,) * dimension]
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    direction = (1, 0)
    
    for i in range(1, n):
        placed = False
        for turn in (1, 0, -1):
            if turn == 1:
                new_dir = (direction[1], -direction[0])
            elif turn == -1:
                new_dir = (-direction[1], direction[0])
            else:
                new_dir = direction
            
            new_pos = (chain[-1][0] + new_dir[0], chain[-1][1] + new_dir[1])
            if new_pos not in occupied:
                chain.append(new_pos)
                occupied[new_pos] = i
                direction = new_dir
                placed = True
                break
        
        if not placed:
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                new_pos = (chain[-1][0] + dx, chain[-1][1] + dy)
                if new_pos not in occupied:
                    chain.append(new_pos)
                    occupied[new_pos] = i
                    direction = (dx, dy)
                    placed = True
                    break
        
        if not placed:
            for j in range(i, n):
                chain.append((chain[-1][0], chain[-1][1] + 1))
            break
    
    return chain

def _compute_energy_full(sequence, chain):
    """Compute the negative H-H contact energy for verification."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for axis in range(len(p)):
                for step in (-1, 1):
                    q = list(p)
                    q[axis] += step
                    q = tuple(q)
                    j = where.get(q)
                    if j is not None and j > i + 1 and sequence[j] == "H":
                        contacts += 1
    return -contacts

def _count_h_contacts(i, chain, sequence, occupied):
    """Count H-H contacts involving residue i."""
    if sequence[i] != 'H':
        return 0
    pos = chain[i]
    count = 0
    for axis in range(len(pos)):
        for step in (-1, 1):
            q = list(pos)
            q[axis] += step
            q = tuple(q)
            j = occupied.get(q)
            if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                count += 1
    return count

def _local_search(sequence, chain, occupied, current_energy, rng, max_iters=200):
    """Hill climbing with single-step moves and incremental energy."""
    n = len(sequence)
    if n <= 2:
        return chain, occupied, current_energy
    
    no_improve = 0
    for _ in range(max_iters):
        if no_improve > 50:
            break
        
        i = rng.randint(1, n - 2)
        old_pos = chain[i]
        old_contacts = _count_h_contacts(i, chain, sequence, occupied)
        
        # Try 4 neighbors
        candidates = []
        for axis in range(2):
            for step in (-1, 1):
                new_pos = list(old_pos)
                new_pos[axis] += step
                new_pos = tuple(new_pos)
                
                # Check self-avoidance
                if new_pos in occupied:
                    if occupied[new_pos] != i:
                        continue
                else:
                    # Check connectivity
                    if abs(new_pos[0] - chain[i-1][0]) + abs(new_pos[1] - chain[i-1][1]) != 1:
                        continue
                    if abs(new_pos[0] - chain[i+1][0]) + abs(new_pos[1] - chain[i+1][1]) != 1:
                        continue
                
                if new_pos == old_pos:
                    continue
                
                candidates.append(new_pos)
        
        if not candidates:
            no_improve += 1
            continue
        
        new_pos = rng.choice(candidates)
        
        # Check if move is valid (already checked in candidates generation mostly, but re-verify connectivity if it was occupied by self)
        # Actually, if new_pos was occupied by i, it's the same pos, handled.
        # If new_pos was free, we checked connectivity.
        
        # Calculate new contacts
        # Temporarily update? Or just calc based on new_pos
        # New contacts for i at new_pos
        new_contacts = 0
        if sequence[i] == 'H':
            for axis in range(2):
                for step in (-1, 1):
                    q = list(new_pos)
                    q[axis] += step
                    q = tuple(q)
                    j = occupied.get(q)
                    if j is not None and j != i and abs(i - j) > 1 and sequence[j] == 'H':
                        new_contacts += 1
        
        delta_e = -(new_contacts - old_contacts)
        
        if delta_e < 0:
            # Accept
            del occupied[old_pos]
            occupied[new_pos] = i
            chain[i] = new_pos
            current_energy += delta_e
            no_improve = 0
        else:
            no_improve += 1
            
    return chain, occupied, current_energy

def _perturb(sequence, chain, occupied, rng, strength=5):
    """Apply random valid moves to jump basins."""
    n = len(sequence)
    for _ in range(strength):
        i = rng.randint(1, n - 2)
        old_pos = chain[i]
        
        candidates = []
        for axis in range(2):
            for step in (-1, 1):
                new_pos = list(old_pos)
                new_pos[axis] += step
                new_pos = tuple(new_pos)
                
                if new_pos in occupied:
                    if occupied[new_pos] != i:
                        continue
                else:
                    if abs(new_pos[0] - chain[i-1][0]) + abs(new_pos[1] - chain[i-1][1]) != 1:
                        continue
                    if abs(new_pos[0] - chain[i+1][0]) + abs(new_pos[1] - chain[i+1][1]) != 1:
                        continue
                
                if new_pos == old_pos:
                    continue
                candidates.append(new_pos)
                
        if candidates:
            new_pos = rng.choice(candidates)
            del occupied[old_pos]
            occupied[new_pos] = i
            chain[i] = new_pos

def fold(sequence, dimension):
    """Fold using Iterated Local Search."""
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    start_time = time.time()
    time_budget = 10.0 # Leave buffer for 10 sequences
    
    best_chain = None
    best_energy = 0
    best_occupied = None
    
    # Initialize
    init_chain = _initial_spiral(sequence, dimension)
    init_occupied = {p: i for i, p in enumerate(init_chain)}
    init_energy = _compute_energy_full(sequence, init_chain)
    
    best_chain = list(init_chain)
    best_energy = init_energy
    best_occupied = {p: i for i, p in enumerate(best_chain)}
    
    rng = random.Random(42 + len(sequence))
    
    while time.time() - start_time < time_budget:
        # Perturb current best (or a copy)
        current_chain = list(best_chain)
        current_occupied = dict(best_occupied)
        
        # Perturbation strength increases slightly or is fixed
        strength = rng.randint(3, 10)
        _perturb(sequence, current_chain, current_occupied, rng, strength)
        
        # Local Search
        current_energy = _compute_energy_full(sequence, current_chain) # Recalc to be safe after perturbation
        current_chain, current_occupied, current_energy = _local_search(
            sequence, current_chain, current_occupied, current_energy, rng, max_iters=100
        )
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_chain = current_chain
            best_occupied = current_occupied
    
    return [list(p) for p in best_chain]
