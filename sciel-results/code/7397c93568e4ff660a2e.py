"""HP lattice protein folding with deadline-aware local search."""
# EVOLVE-BLOCK-START
import random
import time

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _grow(sequence, dimension, rng):
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

def _local_search(sequence, chain, rng, deadline, max_steps=2000):
    n = len(chain)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    # Pre-compute contact counts for speedup? 
    # For n up to 136, _energy is O(n*6). 136*6 = 816 operations. 
    # 2000 steps * 816 = 1.6M ops. This is very fast in Python if optimized.
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        # Pick a random pivot index (not 0 or n-1 usually, but any 1..n-2)
        i = rng.randint(1, n - 2)
        
        # Try to rotate the segment [i, i+1] around chain[i-1]
        # Actually, standard move: move residue i to a new valid position.
        # To maintain connectivity, we can try moving residue i to any of the 6 neighbors of its current position, 
        # provided the new position is not occupied by any other residue (excluding i-1 and i+1).
        
        old_pos = chain[i]
        old_res = sequence[i]
        
        # Remove old_pos from occupancy (temporarily)
        # We need to check neighbors of old_pos.
        # Valid new positions are neighbors of old_pos that are:
        # 1. Not occupied by any residue j where j != i-1 and j != i+1
        # 2. Not equal to old_pos (trivial)
        
        candidates = []
        for q in _neighbours(old_pos):
            # Check if q is occupied
            # We need a quick way to check occupancy. 
            # Building a set every step is O(n). 
            # Let's maintain an occupancy set outside the loop? 
            # No, the chain changes. 
            # For n=136, building a set is cheap.
            
            pass
        
        # Let's use a set for occupancy for this step
        occ = set(chain)
        occ.discard(old_pos)
        
        # Also, we must ensure the new position doesn't collide with i-1 or i+1
        # i-1 and i+1 are in occ. So if we just check 'q not in occ', it handles it.
        
        valid_moves = []
        for q in _neighbours(old_pos):
            if q not in occ:
                valid_moves.append(q)
        
        if not valid_moves:
            continue
            
        new_pos = rng.choice(valid_moves)
        
        # Apply move
        chain[i] = new_pos
        
        # Calculate new energy
        new_energy = _energy(sequence, chain)
        
        if new_energy < current_energy:
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(chain)
        else:
            # Revert (simple hill climbing)
            chain[i] = old_pos
            
    return best_chain, best_energy

def fold(sequence, dimension):
    """Best conformation using growth and local search."""
    start_time = time.time()
    # 330s wall clock limit. Leave 10s margin.
    deadline = start_time + 320.0
    
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0  # 0 is unfolded (0 contacts). Energy is negative.
    
    # Estimate how many iterations we can afford.
    # Each growth is O(n^2) roughly due to neighbor checks.
    # Each local search step is O(n) for set building + O(1) for move + O(n) for energy.
    
    # Strategy: Run a fixed number of full cycles (growth + LS) or until deadline.
    # Let's try 100 restarts.
    # If n=136, growth is fast. LS is 2000 steps.
    # 100 * (growth + 2000 LS steps) should be well within 320s.
    
    num_restarts = 100
    
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
            
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
            
        # Initial energy
        init_energy = _energy(sequence, chain)
        
        # Refine with local search
        refined_chain, refined_energy = _local_search(sequence, chain, rng, deadline)
        
        if refined_energy < best_energy:
            best_energy = refined_energy
            best_chain = refined_chain
            
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
