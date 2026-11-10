"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math


def _neighbours2d(point):
    """Yield the 4 unit-step neighbours of a 2D point (tuple)."""
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours2d(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours2d(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours2d(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours2d(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _incremental_energy_change(sequence, chain, occupied, i, old_pos, new_pos):
    """Compute energy delta when moving residue i from old_pos to new_pos."""
    n = len(chain)
    delta = 0
    # Remove old contacts involving residue i
    if sequence[i] == 'H':
        for q in _neighbours2d(old_pos):
            j = occupied.get(q)
            if j is not None and j != i and abs(j - i) > 1 and sequence[j] == 'H':
                delta += 1  # removing a contact increases energy (less negative)
    # Add new contacts involving residue i
    if sequence[i] == 'H':
        for q in _neighbours2d(new_pos):
            j = occupied.get(q)
            if j is not None and j != i and abs(j - i) > 1 and sequence[j] == 'H':
                delta -= 1  # adding a contact decreases energy (more negative)
    return delta


def _local_search(sequence, chain, max_iter=5000, rng=None):
    """Improve conformation via single-residue moves that preserve self-avoidance."""
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    if rng is None:
        rng = random.Random(42)
    
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    occupied = {p: i for i, p in enumerate(best_chain)}
    
    for _ in range(max_iter):
        improved = False
        # Try moving each residue (skip endpoints for simplicity, or try all)
        for i in range(1, n - 1):
            current_pos = best_chain[i]
            for new_pos in _neighbours2d(current_pos):
                # Check if new position is valid
                if new_pos in occupied:
                    continue
                # Check if moving would create chain adjacency issues
                valid = True
                for q in _neighbours2d(new_pos):
                    j = occupied.get(q)
                    if j is not None and abs(j - i) > 1:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Compute energy change incrementally
                delta = _incremental_energy_change(sequence, best_chain, occupied, i, current_pos, new_pos)
                new_energy = best_energy + delta
                
                if new_energy < best_energy:
                    # Accept the move
                    best_chain[i] = new_pos
                    occupied.pop(current_pos)
                    occupied[new_pos] = i
                    best_energy = new_energy
                    improved = True
                # else: revert (do nothing)
            
            if improved:
                break
        
        if not improved:
            break
    
    return best_chain, best_energy


def _simulated_annealing(sequence, chain, max_iter=10000, initial_temp=5.0, final_temp=0.1, rng=None):
    """Simulated annealing for HP lattice protein folding."""
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    if rng is None:
        rng = random.Random(42)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    cooling_rate = (final_temp / initial_temp) ** (1.0 / max(1, max_iter))
    
    for iteration in range(max_iter):
        temp = initial_temp * (cooling_rate ** iteration)
        i = rng.randint(1, n - 2)
        current_pos = current_chain[i]
        
        # Try each neighbor
        for new_pos in _neighbours2d(current_pos):
            if new_pos in occupied:
                continue
            
            # Check validity
            valid = True
            for q in _neighbours2d(new_pos):
                j = occupied.get(q)
                if j is not None and abs(j - i) > 1:
                    valid = False
                    break
            if not valid:
                continue
            
            # Compute energy change
            delta = _incremental_energy_change(sequence, current_chain, occupied, i, current_pos, new_pos)
            new_energy = current_energy + delta
            
            # Accept if better or with probability exp(-delta/temp)
            if delta < 0 or (temp > 0.01 and rng.random() < math.exp(-delta / temp)):
                # Accept move
                current_chain[i] = new_pos
                occupied.pop(current_pos)
                occupied[new_pos] = i
                current_energy = new_energy
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(current_chain)
                break  # Only accept one move per iteration
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using randomized greedy growths with local optimization and simulated annealing."""
    start_time = time.time()
    deadline = start_time + 175.0  # Safety margin before 210s kill
    
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 1
    
    n = len(sequence)
    # Adjust number of restarts based on sequence length
    if n <= 30:
        num_restarts = 500
    elif n <= 60:
        num_restarts = 300
    else:
        num_restarts = 150
    
    for attempt in range(num_restarts):
        if time.time() > deadline:
            break
        
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Apply local search to improve
        chain, energy = _local_search(sequence, chain, max_iter=min(5000, max(100, 10000 // n)), rng=rng)
        
        if energy < best_energy:
            best, best_energy = chain, energy
    
    # If we have time, apply simulated annealing to the best solution found
    if best is not None and time.time() < deadline - 5.0:
        sa_time_budget = min(30.0, deadline - time.time() - 2.0)
        sa_iter = int(min(50000, max(1000, sa_time_budget * 500)))
        best, best_energy = _simulated_annealing(
            sequence, best, max_iter=sa_iter, initial_temp=3.0, final_temp=0.05, rng=rng
        )
    
    if best is None:  # a straight line is always self-avoiding
        best = [(i, 0) for i in range(len(sequence))]
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
