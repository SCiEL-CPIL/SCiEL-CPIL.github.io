"""Improved HP lattice protein folding with local search refinement."""
# EVOLVE-BLOCK-START
import random
import time

_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    _init_time()
    # Budget is 120s for CPU, but wall clock might be tighter or looser.
    # The prompt says "120 s CPU budget for all sequences together".
    # It also says "Each evaluation must finish within 180 s".
    # And "KILLS the program after 210s wall-clock".
    # Let's aim for 110s to be safe.
    return 110.0 - (time.time() - _START_TIME)

def _neighbours(point):
    """Yield 4-neighbours on square lattice."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _grow(sequence, dimension, rng):
    """
    One randomized greedy chain growth: prefer sites with new H-H contacts, then free space.
    """
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        current = chain[-1]
        options = []
        
        for site in _neighbours(current):
            if site in occupied:
                continue
            
            # Count immediate H-H contacts this placement would make
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            
            # Count free neighbours (for future growth flexibility)
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            # Score: prioritize contacts, then free space, then random tie-breaker
            # Weights can be tuned. Contacts are most important.
            score = gain * 10.0 + free * 0.1 + rng.random()
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick the best option
        options.sort(key=lambda x: x[0], reverse=True)
        site = options[0][1]
        occupied[site] = i
        chain.append(site)
    
    return chain

def _energy(sequence, chain):
    """Compute HP contact energy: -1 for each H-H pair on neighbouring sites not adjacent in chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _local_search(sequence, chain, rng, max_steps=5000, time_limit=5.0):
    """
    Refine a conformation using local search moves (single residue slides).
    Returns the improved chain and its energy.
    """
    n = len(chain)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    
    # Pre-compute H positions for faster contact checks?
    # Or just use the sequence string.
    
    start_time = time.time()
    no_improve_count = 0
    best_chain = list(chain)
    best_energy = current_energy
    
    while max_steps > 0 and (time.time() - start_time) < time_limit:
        max_steps -= 1
        if _time_remaining() < 2.0:
            break
            
        i = rng.randint(1, n - 2)
        prev_pos = chain[i-1]
        curr_pos = chain[i]
        next_pos = chain[i+1]
        
        # Find potential new positions for chain[i]
        # Must be neighbor of prev_pos AND neighbor of next_pos
        # And not occupied by any other residue
        candidates = set(_neighbours(prev_pos)) & set(_neighbours(next_pos))
        candidates.discard(curr_pos)
        
        moved = False
        for new_pos in candidates:
            if new_pos in occupied:
                continue
            
            # Check if this move is valid (it is, by construction of candidates and occupied check)
            # Calculate delta energy
            # Remove old contacts involving i
            # Add new contacts involving i
            
            # For simplicity and speed, let's just do a full energy recalculation if we find a candidate?
            # No, that's too slow. Let's calculate delta.
            
            # Contacts lost:
            # i was at curr_pos. It had contacts with any H at neighbors of curr_pos (excluding i-1, i+1).
            # Contacts gained:
            # i will be at new_pos. It will have contacts with any H at neighbors of new_pos (excluding i-1, i+1).
            
            # Note: The adjacency to i-1 and i+1 doesn't change the "contact" definition (which excludes adjacent chain links).
            # So we only care about non-adjacent H neighbors.
            
            delta = 0
            if sequence[i] == 'H':
                # Lost contacts
                for q in _neighbours(curr_pos):
                    j = occupied.get(q)
                    if j is not None and j != i-1 and j != i+1 and sequence[j] == 'H':
                        # This contact is lost
                        delta += 1 
                
                # Gained contacts
                for q in _neighbours(new_pos):
                    j = occupied.get(q)
                    if j is not None and j != i-1 and j != i+1 and sequence[j] == 'H':
                        # This contact is gained
                        # Note: If j was already in contact with i via curr_pos, we already counted it as lost?
                        # No, curr_pos and new_pos are different.
                        # A neighbor q could be a neighbor of both curr_pos and new_pos?
                        # Yes, if curr_pos and new_pos are distance 2 apart.
                        # But we are moving i. The neighbors of i change.
                        # If q is a neighbor of both, and q is H, then:
                        # Old: i-H contact existed.
                        # New: i-H contact exists.
                        # So delta should be 0 for that specific q.
                        # My simple add/subtract logic above might double count or miss things if q is neighbor to both.
                        
                        # Let's refine:
                        # Lost: H neighbors of curr_pos that are NOT neighbors of new_pos?
                        # Gained: H neighbors of new_pos that are NOT neighbors of curr_pos?
                        # This is getting complicated.
                        
                        # Given N <= 100, full recalculation is ~100*4 = 400 ops.
                        # If we do 5000 steps, that's 2M ops. Very fast in Python? Maybe 0.1s.
                        # Let's stick to full recalculation for correctness and simplicity, as N is small.
                        pass
            
            # Actually, let's just do the move and recalculate energy. It's safer.
            # The move is:
            del occupied[curr_pos]
            occupied[new_pos] = i
            chain[i] = new_pos
            
            new_energy = _energy(sequence, chain)
            
            if new_energy < current_energy:
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
                no_improve_count = 0
            else:
                # Revert
                del occupied[new_pos]
                occupied[curr_pos] = i
                chain[i] = curr_pos
                no_improve_count += 1
            
            moved = True
            break # Only try one candidate per step? Or try all? 
            # Trying all candidates for a specific i might be better.
            # But let's keep it simple: pick a random i, try to move it to a random valid candidate.
        
        if no_improve_count > 1000:
            break
            
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Fold an HP sequence on the square lattice.
    Uses randomized greedy growth followed by local search refinement.
    """
    _init_time()
    
    n = len(sequence)
    if n == 0:
        return []
    
    # Emergency fallback if no time left
    if _time_remaining() < 1.0:
        return [[i, 0] for i in range(n)]
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0 # Energy is negative, so we look for lower (more negative) values
    
    iterations = 0
    
    while _time_remaining() > 5.0:
        iterations += 1
        
        # 1. Grow
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # 2. Refine
        # Use a smaller time limit for local search to allow many restarts
        # Or use a fixed number of steps.
        # Let's use a fixed number of steps to ensure progress.
        chain, energy = _local_search(sequence, chain, rng, max_steps=2000, time_limit=2.0)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
        
        # Early termination if we hit a very low energy?
        # Hard to know the optimum.
        
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
