"""HP lattice protein folding on the cubic lattice.
Uses Global Regrowth Simulated Annealing to allow topological changes 
across the entire chain, not just the tail.
"""
import random
import time
import math

_START_TIME = time.time()
_DEADLINE = _START_TIME + 318.0  # Safety margin before 330s kill


def _neighbours(point):
    """Yield the 6 face-neighbours of point in 3D (fast path)."""
    x, y, z = point
    yield (x + 1, y, z)
    yield (x - 1, y, z)
    yield (x, y + 1, z)
    yield (x, y - 1, z)
    yield (x, y, z + 1)
    yield (x, y, z - 1)


def _energy(sequence, chain):
    """Compute total energy: negative count of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            x, y, z = p
            for q in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _grow(sequence, dimension, rng, temperature=1.0):
    """One randomized greedy chain growth with temperature control.
    Used for initialization.
    """
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        x, y, z = chain[-1]
        for site in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                sx, sy, sz = site
                for q in ((sx+1,sy,sz),(sx-1,sy,sz),(sx,sy+1,sz),(sx,sy-1,sz),(sx,sy,sz+1),(sx,sy,sz-1)):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            # Count free neighbors for entropy bonus
            free = 0
            sx, sy, sz = site
            for q in ((sx+1,sy,sz),(sx-1,sy,sz),(sx,sy+1,sz),(sx,sy-1,sz),(sx,sy,sz+1),(sx,sy,sz-1)):
                if q not in occupied:
                    free += 1
            score = gain + 0.1 * free + rng.random() * temperature
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _regrow_from(sequence, chain, start_idx, rng, temperature=1.0):
    """
    Regrow the chain from start_idx to the end.
    Uses a stochastic greedy approach: at each step, choose among valid sites
    based on a score that includes energy gain and noise.
    Returns a new full chain if successful, None if stuck.
    """
    if start_idx >= len(chain) - 1:
        return list(chain)
    
    n = len(sequence)
    # Copy the prefix
    prefix = chain[:start_idx + 1]
    occupied = {p: i for i, p in enumerate(prefix)}
    new_chain = list(prefix)
    
    for i in range(start_idx + 1, n):
        x, y, z = new_chain[-1]
        options = []
        for site in ((x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)):
            if site in occupied:
                continue
            
            gain = 0
            if sequence[i] == "H":
                sx, sy, sz = site
                for q in ((sx+1,sy,sz),(sx-1,sy,sz),(sx,sy+1,sz),(sx,sy-1,sz),(sx,sy,sz+1),(sx,sy,sz-1)):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            
            # Free neighbors bonus to avoid dead ends
            free = 0
            sx, sy, sz = site
            for q in ((sx+1,sy,sz),(sx-1,sy,sz),(sx,sy+1,sz),(sx,sy-1,sz),(sx,sy,sz+1),(sx,sy,sz-1)):
                if q not in occupied:
                    free += 1
            
            # Score: prioritize contacts, avoid dead ends, add noise
            score = gain * 2.0 + free * 0.1 + rng.random() * temperature
            options.append((score, site))
        
        if not options:
            return None
        
        # Select best option
        # For SA, we might want to select probabilistically, but for regrowth,
        # picking the best stochastic option is a common heuristic for speed.
        # A true SA regrowth would sample from the distribution.
        # Here we pick the max score, which includes noise.
        site = max(options)[1]
        occupied[site] = i
        new_chain.append(site)
        
    return new_chain


def fold(sequence, dimension):
    """
    Best conformation using Global Regrowth Simulated Annealing.
    """
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0
    
    n = len(sequence)
    remaining = _DEADLINE - time.time()
    
    # Phase 1: Initialization
    # Generate several initial chains
    num_initial = 20
    if n > 100:
        num_initial = 10
        
    for trial in range(num_initial):
        if time.time() > _DEADLINE - 120:
            break
        temp = 1.0 + rng.random()
        chain = _grow(sequence, dimension, rng, temperature=temp)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if value < best_energy:
            best_chain = chain
            best_energy = value
    
    if best_chain is None:
        # Fallback
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
        return [list(p) for p in best_chain]
    
    # Phase 2: Global Regrowth Simulated Annealing
    # We allow regrowth from ANY index, including 0 (full re-sampling)
    # This allows the N-terminus to change, escaping deep local minima.
    
    current_chain = list(best_chain)
    current_energy = best_energy
    
    # SA Parameters
    # Initial temperature high enough to accept bad moves
    T = 5.0 
    T_min = 0.1
    cooling_rate = 0.9995  # Slow cooling for long runs
    steps_per_check = 1000 # Check time every N steps
    
    start_time = time.time()
    steps = 0
    no_improve_steps = 0
    max_no_improve = 5000 # Restart if stuck?
    
    while time.time() < _DEADLINE - 10:
        # Pick a random start index for regrowth
        # Bias towards later indices for speed? Or uniform for global exploration?
        # Uniform allows global changes.
        start_idx = rng.randint(0, n - 2)
        
        # Regrow
        new_chain = _regrow_from(sequence, current_chain, start_idx, rng, temperature=T)
        
        if new_chain is None:
            # Failed to complete, cool slightly faster?
            T = max(T_min, T * 0.99)
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        # Metropolis Acceptance
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_chain = list(current_chain)
                best_energy = current_energy
                no_improve_steps = 0
            else:
                no_improve_steps += 1
                
        else:
            no_improve_steps += 1
            
        # Cooling
        T = max(T_min, T * cooling_rate)
        
        steps += 1
        if steps % steps_per_check == 0:
            # Optional: If stuck for a long time, increase T or reset?
            # For now, just continue.
            pass
            
    return [list(p) for p in best_chain]
