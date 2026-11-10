"""HP lattice protein folding on the cubic lattice - simulated annealing approach."""
import random
import math


def _neighbours(point):
    """Yield the 6 neighbours of a point on the cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute the HP energy (negative number of H-H non-consecutive contacts)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _make_initial_chain(sequence, dimension, rng):
    """Create a random self-avoiding walk as initial chain."""
    n = len(sequence)
    if n == 1:
        return [(0,) * dimension]
    
    # Try random walks until we get a valid self-avoiding walk
    for _ in range(1000):
        chain = [(0,) * dimension]
        occupied = {chain[0]}
        success = True
        for i in range(1, n):
            neighbours = list(_neighbours(chain[-1]))
            rng.shuffle(neighbours)
            placed = False
            for site in neighbours:
                if site not in occupied:
                    chain.append(site)
                    occupied.add(site)
                    placed = True
                    break
            if not placed:
                success = False
                break
        if success:
            return chain
    
    # Fallback: straight line
    return [(i,) + (0,) * (dimension - 1) for i in range(n)]


def _simulated_annealing(sequence, dimension, rng, time_budget=220.0):
    """Simulated annealing to find low-energy conformations."""
    import time
    
    start_time = time.time()
    n = len(sequence)
    
    # Start with a random self-avoiding walk
    chain = _make_initial_chain(sequence, dimension, rng)
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    # Annealing parameters
    initial_temp = 10.0
    final_temp = 0.01
    cooling_rate = 0.999
    
    temp = initial_temp
    iteration = 0
    
    while temp > final_temp:
        # Check time budget
        if time.time() - start_time > time_budget:
            break
        
        # Pick a random residue to move (not the first one)
        i = rng.randint(1, n - 1)
        
        # Try moving residue i to a random unoccupied neighbour of its current position
        # We need to be careful: moving a middle residue breaks the chain
        # Instead, let's try a different move: pick two consecutive residues and try
        # to insert a new site between them, or use a "pivot" move
        
        # Actually, for HP folding, a common move is to pick a random internal bond
        # and try to rotate it. But simpler: let's try to move the endpoint or
        # use a "regrow" strategy.
        
        # Let's use a different approach: pick a random position i (1 to n-1) and
        # try to replace chain[i] with a random neighbour of chain[i-1] that is
        # unoccupied and doesn't create conflicts with chain[i+1] if i < n-1.
        
        # This is tricky. Let's use a simpler move: pick a random index i from 1 to n-1,
        # and try to move chain[i] to a random neighbour of chain[i-1] that is free.
        # But we must ensure chain[i] != chain[i+1] if i < n-1, and the new position
        # doesn't conflict with any other occupied site.
        
        old_pos = chain[i]
        neighbours = list(_neighbours(chain[i - 1]))
        rng.shuffle(neighbours)
        
        moved = False
        for new_pos in neighbours:
            if new_pos == old_pos:
                continue
            # Check if new_pos is occupied by any other residue
            conflict = False
            for j in range(n):
                if j != i and chain[j] == new_pos:
                    conflict = True
                    break
            if conflict:
                continue
            # If i < n-1, check that new_pos != chain[i+1]
            if i < n - 1 and new_pos == chain[i + 1]:
                continue
            # This move is valid
            chain[i] = new_pos
            new_energy = _energy(sequence, chain)
            delta = new_energy - current_energy
            
            if delta <= 0 or rng.random() < math.exp(-delta / temp):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
                moved = True
                break
        
        if not moved:
            # Try a different move: pick a random endpoint and regrow a few steps
            # Or just skip this iteration
            pass
        
        temp *= cooling_rate
        iteration += 1
    
    return best_chain, best_energy


def _local_search(sequence, chain, dimension, rng, max_iter=5000):
    """Local search: try single-residue moves to improve energy."""
    n = len(sequence)
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    for _ in range(max_iter):
        improved = False
        for i in range(1, n):
            old_pos = chain[i]
            neighbours = list(_neighbours(chain[i - 1]))
            rng.shuffle(neighbours)
            
            for new_pos in neighbours:
                if new_pos == old_pos:
                    continue
                # Check conflicts
                conflict = False
                for j in range(n):
                    if j != i and chain[j] == new_pos:
                        conflict = True
                        break
                if conflict:
                    continue
                if i < n - 1 and new_pos == chain[i + 1]:
                    continue
                
                chain[i] = new_pos
                new_energy = _energy(sequence, chain)
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(chain)
                    current_energy = new_energy
                    improved = True
                else:
                    chain[i] = old_pos
        
        if not improved:
            break
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Find a low-energy conformation using simulated annealing and local search."""
    rng = random.Random(12345 + len(sequence))
    n = len(sequence)
    
    if n <= 1:
        return [[0] * dimension]
    
    best_chain = None
    best_energy = float('inf')
    
    # Run multiple SA trials with different random seeds
    import time
    start_time = time.time()
    time_per_trial = 200.0 / 5  # 5 trials, 200s total budget
    
    for trial in range(5):
        trial_start = time.time()
        trial_budget = time_per_trial - (trial_start - start_time)
        if trial_budget < 5:
            break
        
        chain, energy = _simulated_annealing(sequence, dimension, rng, time_budget=trial_budget)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
        
        # Local search refinement
        chain2, energy2 = _local_search(sequence, best_chain, dimension, rng, max_iter=2000)
        if energy2 < best_energy:
            best_energy = energy2
            best_chain = chain2
        
        if time.time() - start_time > 210:
            break
    
    if best_chain is None:
        # Fallback
        best_chain = [(i,) + (0,) * (dimension - 1) for i in range(n)]
    
    return [list(p) for p in best_chain]
