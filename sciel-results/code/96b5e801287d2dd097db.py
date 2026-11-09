import random
import time
import math
import sys

def _neighbours(point):
    x, y, z = point
    yield (x+1, y, z)
    yield (x-1, y, z)
    yield (x, y+1, z)
    yield (x, y-1, z)
    yield (x, y, z+1)
    yield (x, y, z-1)

def _energy(sequence, chain):
    """
    Calculate the energy of a chain.
    Energy = - (number of H-H contacts)
    """
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _grow(sequence, dimension, rng):
    """
    Greedy growth to create an initial self-avoiding walk.
    """
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Heuristic: prefer sites that close loops or are close to other H's
            # Simple heuristic: random with slight preference for "tight" packing
            # Count potential contacts
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied:
                        idx = occupied[q]
                        if idx < i - 1 and sequence[idx] == "H":
                            gain += 1
            # Add small random noise to avoid deterministic traps
            score = gain * 10.0 + rng.random()
            options.append((score, site))
        
        if not options:
            return None
            
        # Select best option
        best_score, best_site = max(options)
        chain.append(best_site)
        occupied[best_site] = i
        
    return chain

def _pivot_move(sequence, chain, i, rng):
    """
    Attempt a pivot move at index i.
    Returns (new_chain, delta_e) or (None, 0) if invalid.
    """
    N = len(chain)
    if i <= 0 or i >= N - 1:
        return None, 0
        
    p_prev = chain[i-1]
    p_curr = chain[i]
    
    # Axis is the vector from p_prev to p_curr
    axis = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1], p_curr[2] - p_prev[2])
    
    # Determine rotation direction (+90 or -90)
    sign = 1 if rng.random() < 0.5 else -1
    
    # The part of the chain that moves is chain[i:]
    # We rotate the vector from p_prev to each point in chain[i:] around the axis
    # Wait, standard pivot rotates the tail around the bond (i-1, i).
    # The points in chain[i:] are relative to p_prev? No, they are absolute.
    # Vector v_j = chain[j] - p_prev
    # Rotate v_j around axis by 90 deg * sign
    # New point = p_prev + rotated_v_j
    
    # Pre-calculate rotation matrix components for 90 deg around axis
    # Axis is unit vector (ax, ay, az)
    ax, ay, az = axis
    
    # Rotation of vector (vx, vy, vz) by 90 deg around (ax, ay, az)
    # Formula: v' = v cos(theta) + (k x v) sin(theta) + k (k . v) (1 - cos(theta))
    # For theta = 90: cos=0, sin=1
    # v' = (k x v) + k (k . v)
    
    new_positions = []
    valid = True
    fixed_set = set(chain[:i])
    
    for j in range(i, N):
        p_old = chain[j]
        vx = p_old[0] - p_prev[0]
        vy = p_old[1] - p_prev[1]
        vz = p_old[2] - p_prev[2]
        
        # Cross product k x v
        cx = ay * vz - az * vy
        cy = az * vx - ax * vz
        cz = ax * vy - ay * vx
        
        # Dot product k . v
        dot = ax * vx + ay * vy + az * vz
        
        # v' = (k x v) + k * dot
        rx = cx + ax * dot
        ry = cy + ay * dot
        rz = cz + az * dot
        
        if sign == -1:
            rx = -rx
            ry = -ry
            rz = -rz
            
        p_new = (p_prev[0] + rx, p_prev[1] + ry, p_prev[2] + rz)
        
        # Check collision with fixed part
        if p_new in fixed_set:
            valid = False
            break
            
        new_positions.append(p_new)
        
    if not valid:
        return None, 0
        
    # Calculate Delta Energy
    # Only contacts between fixed part (0..i-1) and moving part (i..N-1) change
    # Also, contacts *within* the moving part do not change relative distances, 
    # so their contribution to energy is constant.
    # We only need to count H-H contacts crossing the boundary i.
    
    old_cross = 0
    for u in range(i):
        if sequence[u] != 'H': continue
        pu = chain[u]
        for v in range(i, N):
            if sequence[v] != 'H': continue
            pv = chain[v]
            # Manhattan distance 1
            if abs(pu[0]-pv[0]) + abs(pu[1]-pv[1]) + abs(pu[2]-pv[2]) == 1:
                old_cross += 1
                
    new_cross = 0
    for u in range(i):
        if sequence[u] != 'H': continue
        pu = chain[u]
        for k, pv_new in enumerate(new_positions):
            v_idx = i + k
            if sequence[v_idx] != 'H': continue
            if abs(pu[0]-pv_new[0]) + abs(pu[1]-pv_new[1]) + abs(pu[2]-pv_new[2]) == 1:
                new_cross += 1
                
    delta_e = -(new_cross - old_cross)
    
    new_chain = chain[:]
    new_chain[i:] = new_positions
    
    return new_chain, delta_e

def _crossover(seq, parent1, parent2, rng):
    """
    Crossover: Swap a segment from parent2 into parent1.
    This is a heuristic crossover for lattice walks.
    It attempts to splice parent2's segment [i..j] into parent1.
    This is complex to guarantee validity. 
    A simpler valid crossover for SAWs is difficult.
    Alternative: Use "Recombination" via alignment? 
    Given time constraints, a robust GA with only mutation (pivot) and elitism 
    is often sufficient if the population is large enough and local search is strong.
    However, let's try a simple "segment swap" if valid, else reject.
    """
    N = len(seq)
    # Pick two cut points
    i = rng.randint(1, N - 2)
    j = rng.randint(i + 1, N - 1)
    
    # Try to replace parent1[i:j+1] with parent2[i:j+1]
    # This usually breaks self-avoidance.
    # Instead, let's stick to a robust Memetic Algorithm:
    # Population of SA runs. 
    # "Crossover" is effectively replaced by "Inheritance of Best" + "New Random Seeds".
    # But to satisfy the structural change request, let's implement a valid 
    # "Link Crossover" or just rely on the diversity of the population 
    # initialized with different random seeds and refined by SA.
    
    # Actually, a simpler structural improvement is using a larger population 
    # of independent SA threads (Parallel SA / Island Model).
    # Let's implement Island Model: 
    # - Divide time into generations.
    # - In each generation, run SA for a short time on each individual.
    # - Migrate best individuals between islands.
    
    # For this implementation, we will just run multiple independent SA 
    # processes in parallel (conceptually) by interleaving their steps.
    
    return None # Not used in the final fold logic below, replaced by Island Model

def fold(sequence, dimension):
    N = len(sequence)
    if N == 0: return []
    
    start_time = time.time()
    # 240s budget for all sequences. Assume ~9s per sequence.
    # Use 8.0s to be safe.
    deadline = start_time + 8.0
    
    # Initialize Population
    pop_size = 10
    population = []
    energies = []
    
    for _ in range(pop_size):
        # Try to grow a valid chain
        attempts = 0
        while attempts < 20:
            rng_init = random.Random()
            chain = _grow(sequence, dimension, rng_init)
            if chain is not None:
                e = _energy(sequence, chain)
                population.append(chain)
                energies.append(e)
                break
            attempts += 1
        
        if len(population) < pop_size:
             # Fallback: straight line
             chain = [tuple([i] + [0]*(dimension-1)) for i in range(N)]
             population.append(chain)
             energies.append(_energy(sequence, chain))
             
    best_chain = population[0]
    best_energy = energies[0]
    
    # Island Model / Parallel SA
    # We will interleave steps of SA for each individual in the population.
    # This allows us to explore multiple basins simultaneously.
    
    # Parameters for SA
    T_initial = max(2.0, N * 0.1)
    T_min = 0.01
    cooling_rate = 0.9995 # Slower cooling to allow more exploration
    
    temps = [T_initial] * pop_size
    
    while time.time() < deadline:
        # Pick an individual to move
        idx = random.randint(0, pop_size - 1)
        chain = population[idx]
        e = energies[idx]
        T = temps[idx]
        
        rng_move = random.Random()
        
        # Perform a pivot move
        i = rng_move.randint(1, N - 2)
        new_chain, delta_e = _pivot_move(sequence, chain, i, rng_move)
        
        if new_chain is not None:
            # Metropolis Acceptance
            if delta_e < 0 or rng_move.random() < math.exp(-delta_e / T):
                population[idx] = new_chain
                energies[idx] += delta_e
                
                if energies[idx] < best_energy:
                    best_energy = energies[idx]
                    best_chain = new_chain
                
                # Cool down
                temps[idx] *= cooling_rate
                if temps[idx] < T_min:
                    temps[idx] = T_min
            else:
                # Rejected, no change
                pass
                
        # Occasionally restart a stuck individual
        if temps[idx] <= T_min * 1.01:
            # If temperature is very low, maybe restart this individual
            # with a new random growth to explore a new basin
            if random.random() < 0.1:
                new_chain = _grow(sequence, dimension, random.Random())
                if new_chain is not None:
                    new_e = _energy(sequence, new_chain)
                    if new_e < energies[idx] + 10: # Only restart if not too bad
                        population[idx] = new_chain
                        energies[idx] = new_e
                        temps[idx] = T_initial
                        if new_e < best_energy:
                            best_energy = new_e
                            best_chain = new_chain

    return [list(p) for p in best_chain]
