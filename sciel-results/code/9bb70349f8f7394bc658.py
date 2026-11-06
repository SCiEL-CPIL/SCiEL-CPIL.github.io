# EVOLVE-BLOCK-START
import time
import random
import math

_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 185.0  # Safety margin before 210s kill


def _time_left():
    return _GLOBAL_LIMIT - (time.time() - _GLOBAL_START)


def _neighbours_2d(x, y):
    """Yield the 4 neighbours of (x,y) on the square lattice."""
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _energy_2d(sequence, chain):
    """Compute energy: -number of non-adjacent H-H contacts."""
    n = len(chain)
    if n == 0:
        return 0
    where = {}
    for i, (x, y) in enumerate(chain):
        where[(x, y)] = i
    contacts = 0
    for i in range(n):
        if sequence[i] == 'H':
            x, y = chain[i]
            for nx, ny in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
                j = where.get((nx, ny))
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2  # each contact counted twice


def _grow_greedy(sequence, dimension, rng, bias=1.0):
    """Randomized greedy chain growth with contact-biased move selection."""
    n = len(sequence)
    if n == 0:
        return []
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        x, y = chain[-1]
        options = []
        for nx, ny in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
            if (nx, ny) in occupied:
                continue
            # Compute gain: new H-H contacts if we place residue i here
            gain = 0
            if sequence[i] == 'H':
                for mx, my in ((nx+1,ny), (nx-1,ny), (nx,ny+1), (nx,ny-1)):
                    j = occupied.get((mx, my))
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            # Free space heuristic: prefer moves that leave more room
            free = 0
            for mx, my in ((nx+1,ny), (nx-1,ny), (nx,ny+1), (nx,ny-1)):
                if (mx, my) not in occupied:
                    free += 1
            score = bias * gain + 0.1 * free + rng.random() * 0.5
            options.append((score, (nx, ny)))
        
        if not options:
            return None
        
        # Select best option
        best_site = max(options, key=lambda t: t[0])[1]
        occupied[best_site] = i
        chain.append(best_site)
    
    return chain


def _local_search(sequence, chain, rng, max_iters=5000, temperature=1.0):
    """Simulated annealing local search to improve conformation energy."""
    n = len(chain)
    if n < 4:
        return chain, _energy_2d(sequence, chain)
    
    best_chain = list(chain)
    best_energy = _energy_2d(sequence, chain)
    current_chain = list(chain)
    current_energy = best_energy
    
    # Track occupied positions for O(1) lookup
    visited = set(current_chain)
    
    # Precompute which residues are H
    is_h = [c == 'H' for c in sequence]
    
    start_temp = temperature
    end_temp = 0.01
    cooling = (end_temp / start_temp) ** (1.0 / max(1, max_iters))
    
    for iteration in range(max_iters):
        if _time_left() < 5.0:
            break
        
        temp = start_temp * (cooling ** iteration)
        
        # Pick a random internal residue to move
        i = rng.randint(1, n - 2)
        x, y = current_chain[i]
        old_pos = (x, y)
        
        # Try moving to one of the 4 neighbours
        moves = [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
        rng.shuffle(moves)
        
        moved = False
        for nx, ny in moves:
            new_pos = (nx, ny)
            
            # Check if new position is occupied by another part of the chain
            # It can be occupied by i-1 or i+1? No, self-avoiding means no two residues
            # can be at the same position.
            # However, we are moving residue i. So new_pos cannot be in visited
            # EXCEPT if we are "passing through" i-1 or i+1?
            # No, in a self-avoiding walk, every position is unique.
            # So new_pos must not be in visited.
            # BUT, if we move i to a position adjacent to i-1 or i+1, that's fine
            # as long as it's not the *same* position.
            # Wait, if new_pos == current_chain[i-1], that's a collision.
            # If new_pos == current_chain[i+1], that's a collision.
            # So we just check if new_pos is in visited.
            
            if new_pos in visited:
                continue
            
            # Perform the move temporarily
            visited.remove(old_pos)
            visited.add(new_pos)
            current_chain[i] = new_pos
            
            # Calculate energy
            new_energy = _energy_2d(sequence, current_chain)
            delta_e = new_energy - current_energy
            
            # Acceptance criterion
            if delta_e <= 0 or rng.random() < math.exp(-delta_e / max(temp, 1e-10)):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                moved = True
                break
            else:
                # Reject move: revert
                current_chain[i] = old_pos
                visited.remove(new_pos)
                visited.add(old_pos)
        
        if not moved:
            # If no move was accepted, state is unchanged
            pass
            
    return best_chain, best_energy


def _chain_to_lists(chain):
    """Convert chain of tuples to list of lists."""
    return [list(p) for p in chain]


def fold(sequence, dimension):
    """Fold a sequence into a low-energy self-avoiding conformation."""
    if dimension != 2:
        # Fallback for non-2D: straight line
        return [[i] + [0] * (dimension - 1) for i in range(len(sequence))]
    
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    if n == 2:
        return [[0, 0], [1, 0]]
    
    rng = random.Random(42 + n)
    
    best_chain = None
    best_energy = 0  # Energy is negative, so 0 is worst
    
    # Phase 1: Multiple greedy growths with different biases
    biases = [0.5, 1.0, 1.5, 2.0, 3.0]
    num_growths = 50
    
    for bias in biases:
        for _ in range(num_growths // len(biases)):
            if _time_left() < 10.0:
                break
            chain = _grow_greedy(sequence, 2, rng, bias=bias)
            if chain is None:
                continue
            energy = _energy_2d(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = chain
    
    if best_chain is None:
        # Fallback: straight line
        return [[i, 0] for i in range(n)]
    
    # Phase 2: Local search (simulated annealing) on the best chain
    if _time_left() > 15.0:
        improved_chain, improved_energy = _local_search(sequence, best_chain, rng, max_iters=3000, temperature=2.0)
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy
    
    # Phase 3: More local search iterations if time permits
    if _time_left() > 20.0:
        improved_chain, improved_energy = _local_search(sequence, best_chain, rng, max_iters=5000, temperature=1.0)
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy
    
    # Phase 4: Try a few more greedy growths and local search if time allows
    while _time_left() > 30.0:
        chain = _grow_greedy(sequence, 2, rng, bias=rng.uniform(0.5, 3.0))
        if chain is None:
            continue
        energy = _energy_2d(sequence, chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
            if _time_left() > 25.0:
                improved_chain, improved_energy = _local_search(sequence, best_chain, rng, max_iters=2000, temperature=1.5)
                if improved_energy < best_energy:
                    best_chain = improved_chain
                    best_energy = improved_energy
    
    return _chain_to_lists(best_chain)
# EVOLVE-BLOCK-END
