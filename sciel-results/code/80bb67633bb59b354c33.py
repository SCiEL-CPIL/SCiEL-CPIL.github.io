"""
HP lattice protein folding on the cubic lattice.
Strategy: Monte Carlo Chain (Metropolis) with Pivot and Slide moves.
Starts from a greedy self-avoiding walk, then optimizes via local moves.
"""
import random
import time
import math

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # Safety margin before 330s kill

def _neighbours(point):
    """Yield the 6 face-neighbours of point in 3D."""
    x, y, z = point
    yield (x + 1, y, z)
    yield (x - 1, y, z)
    yield (x, y + 1, z)
    yield (x, y - 1, z)
    yield (x, y, z + 1)
    yield (x, y, z - 1)

def _energy(sequence, chain):
    """Calculate energy (negative number of H-H contacts)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _greedy_grow(sequence, dimension, rng):
    """Generate a self-avoiding walk using a greedy heuristic."""
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

def _mc_fold(sequence, dimension, rng, time_budget):
    """
    Perform Monte Carlo optimization starting from a greedy seed.
    Uses Pivot and Slide moves.
    """
    N = len(sequence)
    if N < 2:
        return [(0,0,0)] if N == 1 else []
    
    # 1. Generate initial seed
    chain = _greedy_grow(sequence, dimension, rng)
    if chain is None:
        # Fallback: straight line
        chain = [(i, 0, 0) for i in range(N)]
        
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    # Cooling schedule
    T = 1.0
    T_min = 0.01
    steps = 0
    max_steps = 50000
    
    while time.time() < time_budget and steps < max_steps:
        steps += 1
        move_type = rng.random()
        
        if move_type < 0.6:
            # Pivot move
            pivot = rng.randint(1, N - 2)
            # Generate a random rotation vector
            while True:
                v = (rng.randint(-2, 2), rng.randint(-2, 2), rng.randint(-2, 2))
                if v[0] != 0 or v[1] != 0 or v[2] != 0:
                    break
            
            # Check if the tail (pivot..N-1) can be rotated
            # We need to check if the new positions are free.
            # To do this efficiently, we would need to compute the new positions.
            # For simplicity and robustness, we do a "tentative" move:
            # 1. Remove tail from occupied
            # 2. Compute new tail positions
            # 3. Check collisions
            # 4. If collision, revert.
            
            tail = chain[pivot:]
            old_positions = list(tail)
            occupied_copy = occupied.copy()
            
            # Remove tail
            for p in tail:
                del occupied_copy[p]
            
            # Rotate
            # Simple rotation around origin for the vector, then translate back
            # Actually, pivot is at chain[pivot]. We rotate the vector from pivot to i.
            # Let's use a simpler rotation: pick a random axis and angle?
            # Or just try a few standard rotations (90, 180, 270) around x, y, z axes.
            # Let's try a random 90-degree rotation around a random axis through the pivot.
            
            axis = rng.randint(0, 2)
            # Direction of rotation
            sign = 1 if rng.random() < 0.5 else -1
            
            # Define rotation matrix for 90 deg around axis
            # Axis 0 (x): (x, y, z) -> (x, -z, y) or (x, z, -y)
            # Axis 1 (y): (x, y, z) -> (z, y, -x) or (-z, y, x)
            # Axis 2 (z): (x, y, z) -> (-y, x, z) or (y, -x, z)
            
            def rotate_point(p, pivot, axis, sign):
                px, py, pz = p
                vx, vy, vz = p[0]-pivot[0], p[1]-pivot[1], p[2]-pivot[2]
                if axis == 0:
                    if sign == 1:
                        nx, ny, nz = vx, -vz, vy
                    else:
                        nx, ny, nz = vx, vz, -vy
                elif axis == 1:
                    if sign == 1:
                        nx, ny, nz = vz, vy, -vx
                    else:
                        nx, ny, nz = -vz, vy, vx
                else:
                    if sign == 1:
                        nx, ny, nz = -vy, vx, vz
                    else:
                        nx, ny, nz = vy, -vx, vz
                return (pivot[0] + nx, pivot[1] + ny, pivot[2] + nz)
            
            new_tail = []
            valid = True
            for p in tail:
                new_p = rotate_point(p, chain[pivot], axis, sign)
                if new_p in occupied_copy:
                    valid = False
                    break
                occupied_copy[new_p] = len(new_tail) + pivot
                new_tail.append(new_p)
            
            if valid:
                new_chain = chain[:pivot] + new_tail
                new_energy = current_energy
                # Recalculate energy? 
                # Energy change only involves contacts involving the tail or the pivot.
                # For simplicity and correctness, recalc is safer but slower.
                # Given N is small (max 136), recalc is O(N).
                new_energy = _energy(sequence, new_chain)
                
                delta_e = new_energy - current_energy
                if delta_e <= 0 or rng.random() < math.exp(-delta_e / T):
                    chain = new_chain
                    occupied = occupied_copy
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
        else:
            # Slide move (regrow a small tail)
            # Pick a random cut point
            cut = rng.randint(max(1, N-10), N-1)
            # Remove tail
            tail = chain[cut:]
            for p in tail:
                del occupied[p]
            
            # Regrow tail greedily with some randomness
            new_tail = []
            last_pos = chain[cut-1]
            current_occupied = occupied
            for i in range(cut, N):
                options = []
                for site in _neighbours(last_pos):
                    if site in current_occupied:
                        continue
                    gain = 0
                    if sequence[i] == "H":
                        gain = sum(1 for q in _neighbours(site)
                                   if q in current_occupied and sequence[current_occupied[q]] == "H")
                    free = sum(1 for q in _neighbours(site) if q not in current_occupied)
                    options.append((gain + 0.1 * free + rng.random(), site))
                
                if not options:
                    # Failed to regrow, revert
                    for p in tail:
                        occupied[p] = chain.index(p) # This is inefficient, but rare
                    # Actually, we need to restore the original tail positions
                    # Let's just revert the chain and occupied
                    chain = best_chain if False else chain # Wait, we modified chain in place? No, chain is a list.
                    # We need to restore the original tail
                    # Let's do this more carefully.
                    pass
                    break
                
                site = max(options)[1]
                current_occupied[site] = i
                new_tail.append(site)
                last_pos = site
            
            if len(new_tail) == N - cut:
                new_chain = chain[:cut] + new_tail
                new_energy = _energy(sequence, new_chain)
                delta_e = new_energy - current_energy
                if delta_e <= 0 or rng.random() < math.exp(-delta_e / T):
                    chain = new_chain
                    occupied = {p: i for i, p in enumerate(chain)}
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
            else:
                # Revert
                for i, p in enumerate(tail):
                    occupied[p] = cut + i
                chain = chain[:cut] + tail
            
        # Cool down
        T = max(T_min, T * 0.999)
        
    return best_chain

def fold(sequence, dimension):
    """Best conformation using Monte Carlo optimization."""
    N = len(sequence)
    if N == 0:
        return []
    if N == 1:
        return [[0, 0, 0]]
    
    # Allocate time budget per sequence
    # Total budget 240s for 26 sequences ~ 9s per sequence.
    # But we have 330s wall clock. Let's use a global deadline.
    # We'll just run until the global deadline or until we've done enough steps.
    
    rng = random.Random(12345 + len(sequence))
    
    # Run MC
    best_chain = _mc_fold(sequence, dimension, rng, _DEADLINE)
    
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
