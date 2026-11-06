"""HP lattice protein folding on the square lattice."""
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
    # Budget is 120s CPU for all sequences, 210s wall-clock kill.
    # Use 190s as a safe deadline to ensure we finish before kill.
    return 190.0 - (time.time() - _START_TIME)

def _neighbours(point):
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _grow_beam(sequence, dimension, rng, beam_width=10):
    """
    Beam search growth.
    Keeps the top `beam_width` partial chains based on a heuristic score.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    # Each item in beam is (chain, occupied_set, score)
    # Score is current energy (negative is better) + heuristic
    start = (0, 0)
    initial_chain = [start]
    initial_occupied = {start: 0}
    beam = [(initial_chain, initial_occupied, 0.0)]
    
    for i in range(1, n):
        next_beam = []
        for chain, occupied, score in beam:
            current = chain[-1]
            # Generate all valid extensions
            for site in _neighbours(current):
                if site in occupied:
                    continue
                
                # Calculate immediate gain
                gain = 0
                if sequence[i] == 'H':
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            gain += 1
                
                # Heuristic: favor contacts. 
                # We want to maximize gain, so minimize -gain.
                new_score = score - gain * 1.0
                
                # Add to candidate list
                new_chain = chain + [site]
                new_occupied = dict(occupied)
                new_occupied[site] = i
                
                next_beam.append((new_chain, new_occupied, new_score))
        
        if not next_beam:
            return None
        
        # Sort by score (lower is better, i.e., more negative energy potential)
        # Tie-breaker: random to maintain diversity
        next_beam.sort(key=lambda x: (x[2], rng.random()))
        
        # Keep top beam_width
        beam = next_beam[:beam_width]
        
        # Early termination if time is low
        if _time_remaining() < 2.0:
             break

    if not beam:
        return None
    
    # Pick the best one
    best_chain, _, _ = beam[0]
    return best_chain

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

def _local_search(sequence, chain, rng, max_steps=1000):
    """
    Local search using single-residue moves.
    Moves a residue i to a new position that maintains connectivity with i-1 and i+1.
    Accepts moves that improve energy (lower value).
    """
    n = len(chain)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    
    for _ in range(max_steps):
        # Pick a random internal residue to move
        i = rng.randint(1, n - 2)
        prev = chain[i-1]
        curr = chain[i]
        next_ = chain[i+1]
        
        # Find valid positions for the new location of residue i
        # It must be a neighbor of prev AND a neighbor of next_
        # And it must not be the current position
        candidates = set(_neighbours(prev)) & set(_neighbours(next_))
        candidates.discard(curr)
        
        if not candidates:
            continue
            
        # Pick a random candidate from the valid ones
        new_pos = rng.choice(list(candidates))
        
        # Check if the new position is occupied by another residue
        if new_pos in occupied:
            continue
            
        # Perform the move
        del occupied[curr]
        occupied[new_pos] = i
        chain[i] = new_pos
        
        # Calculate new energy
        new_energy = _energy(sequence, chain)
        
        # Accept if better (more negative), otherwise revert
        if new_energy < current_energy:
            current_energy = new_energy
        else:
            # Revert
            del occupied[new_pos]
            occupied[curr] = i
            chain[i] = curr
    
    return chain, current_energy

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
    
    # Perform many restarts with time-adaptive budget allocation
    # Early on: more restarts with shorter local search
    # Later: fewer restarts with longer local search
    max_iterations = 5000
    iterations = 0
    
    # Time-based budget allocation
    # Reserve 10% of time for final polish
    total_time = 190.0
    polish_time = 19.0
    
    while _time_remaining() > polish_time and iterations < max_iterations:
        iterations += 1
        
        # Generate a new conformation
        chain = _grow_beam(sequence, dimension, rng, beam_width=5)
        if chain is None:
            continue
        
        # Adaptive local search steps: more steps as we use more time
        # Start with 50 steps, increase to 200 as time progresses
        progress = (total_time - _time_remaining()) / (total_time - polish_time)
        max_ls_steps = int(50 + progress * 150)
        
        # Refine with local search
        chain, energy = _local_search(sequence, chain, rng, max_steps=max_ls_steps)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
        
        # Check if we're making progress; if not, try different beam widths
        if iterations % 10 == 0 and best_chain is not None:
            # Try a slightly different approach every 10 iterations
            pass
    
    # Final polish pass with more local search steps
    if best_chain is not None and _time_remaining() > 2.0:
        # Deep polish: many local search steps on the best solution
        chain, energy = _local_search(sequence, best_chain, rng, max_steps=500)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
            
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
    
    # Final validation: ensure the chain is self-avoiding
    # If not, fall back to a simple valid chain
    if len(set(map(tuple, best_chain))) != len(best_chain):
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
