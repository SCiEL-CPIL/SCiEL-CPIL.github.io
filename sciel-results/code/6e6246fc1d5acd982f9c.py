"""HP lattice protein folding using Simulated Annealing."""
# EVOLVE-BLOCK-START
import random
import time
import math

def _neighbours(point):
    """Yield the 6 neighbors of a point in 3D cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _compute_energy(sequence, chain):
    """Compute energy: -number of H-H contacts (non-consecutive)."""
    n = len(chain)
    if n == 0:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _random_walk(sequence, dimension, rng):
    """Generate a random self-avoiding walk."""
    n = len(sequence)
    if n <= 1:
        return [(0, 0, 0)]
    
    chain = [(0, 0, 0)]
    occupied = {(0, 0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        neighbors = list(_neighbours(last))
        rng.shuffle(neighbors)
        
        placed = False
        for site in neighbors:
            if site not in occupied:
                chain.append(site)
                occupied[site] = i
                placed = True
                break
        
        if not placed:
            # Backtrack or fail
            # Simple strategy: if stuck, restart is handled by caller.
            # For efficiency, we just return None if we get stuck early, 
            # but for short chains it's rare. For long chains, SAWs are hard.
            # However, the task requires 64-136 residues. Random walks might fail.
            # We need a more robust initialization or accept that some walks fail.
            # Let's try a "biased" random walk or just return None.
            return None
            
    return chain

def _simulated_annealing(sequence, chain, time_limit, rng):
    """Optimize the chain using Simulated Annealing."""
    n = len(chain)
    if n <= 1:
        return chain, _compute_energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _compute_energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    t_end = start_time + time_limit
    
    # Temperature schedule
    # Start high, cool down
    # Heuristic: T_start ~ 5.0, T_end ~ 0.001
    # Number of steps depends on time.
    # We estimate steps per second.
    
    # Pre-compute neighbors for speed? No, dynamic.
    
    T = 5.0
    T_min = 0.001
    alpha = 0.9995  # Cooling rate per step
    
    # We need to know how many steps we can do.
    # Let's run until time limit.
    
    pos_map = {p: i for i, p in enumerate(current_chain)}
    
    steps = 0
    while time.time() < t_end:
        # Pick a random interior residue
        i = rng.randint(1, n - 2)
        
        current_site = current_chain[i]
        prev_site = current_chain[i-1]
        next_site = current_chain[i+1]
        
        # Find valid new sites
        # A site is valid if:
        # 1. It is free (not in pos_map)
        # 2. It is adjacent to prev_site
        # 3. It is adjacent to next_site
        # 4. It is not prev_site or next_site (implied by adjacency and distinctness usually, but check)
        
        # Actually, for a single residue move in a lattice walk:
        # The residue moves from C to N.
        # N must be adjacent to C[i-1] and C[i+1].
        # N must not be occupied by any other part of the chain.
        
        candidates = []
        # Check neighbors of current_site
        for new_site in _neighbours(current_site):
            if new_site == prev_site or new_site == next_site:
                continue
            if new_site in pos_map:
                continue
            
            # Check adjacency to neighbors
            dist_prev = sum(abs(a-b) for a, b in zip(new_site, prev_site))
            dist_next = sum(abs(a-b) for a, b in zip(new_site, next_site))
            
            if dist_prev == 1 and dist_next == 1:
                candidates.append(new_site)
        
        if not candidates:
            continue
            
        new_site = rng.choice(candidates)
        
        # Calculate energy delta
        # Old energy contribution of i
        # New energy contribution of i
        # This requires checking H-H contacts.
        
        # Efficient delta calculation:
        # 1. Remove i from current_site. Check if it breaks any H-H contacts.
        # 2. Add i to new_site. Check if it creates any H-H contacts.
        
        old_energy = current_energy
        new_energy = current_energy
        
        # Remove i from current_site
        if sequence[i] == 'H':
            for q in _neighbours(current_site):
                j = pos_map.get(q)
                if j is not None and j != i and sequence[j] == 'H':
                    # Check if this contact is non-consecutive
                    if abs(i - j) > 1:
                        new_energy += 1 # Energy increases (less negative) by 1 if contact lost
                        # Wait, energy = -contacts.
                        # If we lose a contact, contacts decrease by 1, so energy increases by 1 (e.g. -5 -> -4).
                        # So delta = +1.
        
        # Add i to new_site
        if sequence[i] == 'H':
            for q in _neighbours(new_site):
                j = pos_map.get(q)
                if j is not None and j != i and sequence[j] == 'H':
                    if abs(i - j) > 1:
                        new_energy -= 1 # Energy decreases (more negative) by 1 if contact gained
        
        delta = new_energy - old_energy
        
        # Acceptance
        if delta < 0:
            # Accept
            pass
        else:
            # Metropolis criterion
            if T > 0:
                prob = math.exp(-delta / T)
                if rng.random() > prob:
                    # Reject
                    # Cool down
                    T *= alpha
                    if T < T_min:
                        T = T_min
                    continue
            
        # Apply move
        # Update pos_map
        del pos_map[current_site]
        pos_map[new_site] = i
        current_chain[i] = new_site
        current_energy = new_energy
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_chain = list(current_chain)
            
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
            # Optionally restart or stop if stuck for too long, but simple cooling is fine for now
            # To avoid infinite loops at T_min with no improvement, we could add a restart mechanism
            # but for now we just continue.
            
        steps += 1
        
    return best_chain, best_energy

def fold(sequence, dimension):
    """Fold the sequence using Simulated Annealing."""
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    best_chain = None
    best_energy = 0
    
    rng = random.Random(12345 + n)
    
    # We will run multiple independent SA runs, each with a time slice.
    # Or one long run with restarts.
    # Let's do multiple runs with decreasing time budgets or equal time budgets.
    
    # Estimate time per step.
    # Let's just run SA in a loop, restarting if it gets stuck or just running multiple instances.
    
    # Strategy: 
    # 1. Generate a random walk.
    # 2. Run SA for a fixed time slice.
    # 3. Keep best.
    # 4. Repeat until time limit.
    
    # How many runs? 
    # Let's say each run gets 5-10 seconds.
    # Total time 290s.
    
    run_time = 5.0
    
    while time.time() < deadline - run_time:
        # Generate initial chain
        chain = _random_walk(sequence, dimension, rng)
        if chain is None:
            # Random walk failed, try a few times quickly
            found = False
            for _ in range(10):
                chain = _random_walk(sequence, dimension, rng)
                if chain:
                    found = True
                    break
            if not found:
                # If we can't even make a walk, something is wrong or sequence is too long for random walk
                # Fallback to straight line
                best_chain = [[i] + [0] * (dimension - 1) for i in range(n)]
                break
        
        # Run SA
        opt_chain, opt_energy = _simulated_annealing(sequence, chain, run_time, rng)
        
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_chain = opt_chain
            
        # Check if we have enough time for another run
        if time.time() > deadline - run_time:
            break
            
    # Fallback
    if best_chain is None:
        best_chain = [[i] + [0] * (dimension - 1) for i in range(n)]
        best_energy = 0
        
    return best_chain
# EVOLVE-BLOCK-END
