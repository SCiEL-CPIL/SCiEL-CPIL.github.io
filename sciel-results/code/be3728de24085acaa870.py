# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_TOTAL_BUDGET = 195.0  # Safety margin before 210s kill

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Compute HP contact energy: -1 for each H-H pair on neighbouring lattice sites not adjacent in chain."""
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
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
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

def _try_pivot_move(sequence, chain, k, rot_choice, occupied_set):
    """
    Attempt a pivot move at index k with a specific rotation.
    Returns (new_chain, new_energy_delta) if valid, else (None, None).
    """
    n = len(chain)
    px, py = chain[k]
    
    # Determine rotation
    v_x = chain[k+1][0] - px
    v_y = chain[k+1][1] - py
    
    if rot_choice == 1: # 90 deg
        def rot(x, y): return (-y, x)
    elif rot_choice == 2: # 180 deg
        def rot(x, y): return (-x, -y)
    else: # 270 deg
        def rot(x, y): return (y, -x)
        
    prefix_occupied = set(chain[:k])
    
    new_chain = list(chain[:k+1])
    
    # Track energy change
    # Energy = -contacts
    # Delta Energy = -(new_contacts - old_contacts) = old_contacts - new_contacts
    # We need to count contacts involving the moving suffix (indices > k)
    # and the fixed prefix (indices < k) and internal suffix contacts.
    # Actually, simpler: 
    # Old contacts:
    # 1. Internal to suffix: unchanged (rigid body)
    # 2. Between suffix and prefix: removed
    # 3. Internal to prefix: unchanged
    # New contacts:
    # 1. Internal to suffix: unchanged
    # 2. Between new suffix and prefix: added
    # So Delta = (Added Contacts) - (Removed Contacts)
    # Energy Delta = -Delta_Contacts
    
    delta_contacts = 0
    
    # 1. Count removed contacts (old suffix vs prefix)
    # For each residue i in [k+1, n-1], check neighbors in prefix [0, k-1]
    # Note: index k is the pivot, it stays fixed. 
    # Contacts between k and k+1 are chain bonds, not HP contacts.
    # Contacts between i and j where |i-j|>1 are HP contacts.
    
    # We need to be careful: 
    # Old state: chain
    # New state: new_chain
    
    # Removed contacts:
    # For i in range(k+1, n):
    #   pos = chain[i]
    #   for q in _neighbours(pos):
    #      j = where_old.get(q)
    #      if j is not None and j < k and abs(i-j) > 1 and sequence[i]=='H' and sequence[j]=='H':
    #          delta_contacts -= 1
    
    # Added contacts:
    # For i in range(k+1, n):
    #   pos = new_chain[i]
    #   for q in _neighbours(pos):
    #      j = where_new.get(q) # which is same as where_old for prefix
    #      if j is not None and j < k and abs(i-j) > 1 and sequence[i]=='H' and sequence[j]=='H':
    #          delta_contacts += 1
      
    # Optimization: 
    # The prefix is fixed. 
    # We can pre-calculate "contact potential" for prefix H's.
    # But let's just iterate.
    
    # To speed up, we only check H residues in the suffix.
    
    # Build a set of H indices in prefix? No, just check sequence[j]=='H'
    
    # Let's build a map of prefix positions to indices for fast lookup?
    # We already have occupied_set for validity, but we need index mapping for contact logic.
    # The `where` dict is O(1) lookup. Let's pass it or build it?
    # Building `where` for the whole chain is O(N). 
    # In the main loop, we maintain `where`.
    
    # However, `_try_pivot_move` is called frequently. 
    # If we pass `where` (dict of pos->idx), we can use it.
    # Note: `where` for the prefix is unchanged.
    
    # Let's assume `where` is passed.
    
    # Wait, the current signature doesn't pass `where`. 
    # I will modify the main loop to maintain `where` and pass it.
    
    # For now, let's stick to the logic inside.
    
    valid = True
    # We need to check collisions with prefix [0, k-1]
    # And update new_chain
    
    # We also need to check self-collision within the new suffix? 
    # No, rigid rotation preserves self-avoidance of the suffix.
    # Only collision with prefix matters.
    
    # Let's iterate and build new_chain while checking validity and counting contacts.
    
    # We need `where_old` to count removed contacts? 
    # Actually, we can just count contacts with the prefix set.
    # Prefix H's positions are fixed.
    
    # Let's create a set of prefix H positions for O(1) check?
    # prefix_h_set = {chain[j] for j in range(k) if sequence[j]=='H'}
    # This is O(k). 
    # Then for each suffix H at pos, check if any neighbor is in prefix_h_set.
    
    prefix_h_set = set()
    for j in range(k):
        if sequence[j] == 'H':
            prefix_h_set.add(chain[j])
            
    # Removed contacts
    for i in range(k+1, n):
        if sequence[i] == 'H':
            pos = chain[i]
            for q in _neighbours(pos):
                if q in prefix_h_set:
                    # Need to check if j (index of q) is not k (pivot) or i-1, i+1
                    # But q is in prefix [0, k-1], so index < k.
                    # i >= k+1. So |i - j| >= 1.
                    # Is it possible that j = i-1? No, j < k < i.
                    # Is it possible that j = i+1? No.
                    # So any contact with prefix H is a valid HP contact.
                    delta_contacts -= 1
                    
    # Added contacts & Validity check
    for i in range(k+1, n):
        ox = chain[i][0] - px
        oy = chain[i][1] - py
        
        nx, ny = rot(ox, oy)
        new_pos = (px + nx, py + ny)
        
        if new_pos in prefix_occupied:
            valid = False
            break
            
        new_chain.append(new_pos)
        
        if sequence[i] == 'H':
            for q in _neighbours(new_pos):
                if q in prefix_h_set:
                    delta_contacts += 1
                    
    if not valid:
        return None, None
        
    return new_chain, -delta_contacts # Energy is -contacts, so delta E = -delta C

def _pivot_search(sequence, chain, time_limit, rng):
    """
    Monte Carlo search using pivot moves with simulated annealing.
    Optimized with delta energy updates.
    """
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Maintain occupied set and position-to-index map
    occupied_set = set(current_chain)
    where = {p: i for i, p in enumerate(current_chain)}
    
    # Simulated Annealing parameters
    T_start = 1.0
    T_end = 0.05
    start_time = time.time()
    
    # Estimate iterations
    # With delta updates, we can do much more.
    # Let's aim for ~1ms per iter? No, much faster.
    # Let's just run until time limit.
    
    cooling_factor = 0.99995 # Slow cooling
    T = T_start
    
    while time.time() - start_time < time_limit:
        
        # Pick a pivot point
        k = rng.randint(0, n - 2)
        rot_choice = rng.randint(1, 3)
        
        new_chain, delta_e = _try_pivot_move(sequence, current_chain, k, rot_choice, occupied_set)
        
        if new_chain is None:
            continue
            
        new_energy = current_energy + delta_e
        
        # Acceptance
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            # Update state
            # We need to update `where` and `occupied_set`?
            # `occupied_set` doesn't change (same set of positions, just rearranged? No, positions change!)
            # Wait, `occupied_set` contains the set of all positions in the chain.
            # The set of positions is invariant under pivot moves? 
            # YES. The chain is a self-avoiding walk. The set of occupied sites is the same.
            # So `occupied_set` is constant!
            
            # But `where` changes for indices > k.
            # Updating `where` is O(N-k).
            # For N=100, this is fast enough.
            
            for i in range(k+1, n):
                old_pos = current_chain[i]
                new_pos = new_chain[i]
                where[old_pos] = -1 # Mark as removed? Or just delete?
                # Actually, we can just rebuild `where` for the suffix?
                # Or update:
                del where[old_pos]
                where[new_pos] = i
                
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                
        T *= cooling_factor
        if T < T_end:
            T = T_end
            
    return best_chain, best_energy

def _greedy_polish(sequence, chain, time_limit, rng):
    """
    Local search that only accepts energy-improving moves.
    """
    n = len(chain)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    where = {p: i for i, p in enumerate(current_chain)}
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        k = rng.randint(0, n - 2)
        rot_choice = rng.randint(1, 3)
        
        new_chain, delta_e = _try_pivot_move(sequence, current_chain, k, rot_choice, where)
        
        if new_chain is None:
            continue
            
        new_energy = current_energy + delta_e
        
        if delta_e < 0:
            # Update where
            for i in range(k+1, n):
                old_pos = current_chain[i]
                new_pos = new_chain[i]
                del where[old_pos]
                where[new_pos] = i
                
            current_chain = new_chain
            current_energy = new_energy
            best_chain = list(current_chain)
            best_energy = new_energy
        # If delta_e >= 0, reject (greedy)
            
    return best_chain, best_energy

def fold(sequence, dimension):
    """Fold the sequence into a low-energy conformation."""
    n = len(sequence)
    if n == 0:
        return []
    
    # Check global time
    elapsed = time.time() - _GLOBAL_START
    if elapsed > _TOTAL_BUDGET:
        return [[i, 0] for i in range(n)]
    
    # Allocate remaining time
    remaining = _TOTAL_BUDGET - elapsed
    
    if remaining <= 1.0:
        rng = random.Random(12345 + n)
        initial = _greedy_grow(sequence, dimension, rng)
        if initial:
            return [list(p) for p in initial]
        return [[i, 0] for i in range(n)]
        
    rng = random.Random(12345 + n)
    
    best_chain = None
    best_energy = 0
    
    # Strategy: 
    # 1. Generate initial chain
    # 2. Run SA Pivot Search
    # 3. Run Greedy Polish
    
    # We can do multiple restarts if time permits.
    # Let's try 2 restarts for larger sequences, 1 for smaller?
    # Or just 1 long run?
    # Given the speedup, 2 restarts might be good.
    
    num_restarts = 2
    time_per_restart = remaining / num_restarts
    
    for k in range(num_restarts):
        if time.time() - _GLOBAL_START > _TOTAL_BUDGET:
            break
            
        init_rng = random.Random(12345 + n + k * 100)
        initial = _greedy_grow(sequence, dimension, init_rng)
        
        if initial is None:
            initial = [(i, 0) for i in range(n)]
            
        # Refine
        # Split time: 70% SA, 30% Polish
        t_sa = time_per_restart * 0.7
        t_polish = time_per_restart * 0.3
        
        sa_chain, sa_energy = _pivot_search(sequence, initial, t_sa, rng)
        final_chain, final_energy = _greedy_polish(sequence, sa_chain, t_polish, rng)
        
        if final_chain is not None:
            if best_chain is None or final_energy < best_energy:
                best_chain = final_chain
                best_energy = final_energy
                
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
