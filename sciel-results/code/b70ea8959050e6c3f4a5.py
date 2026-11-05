import random
import time

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Calculate energy of a full chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _beam_search(sequence, dimension, beam_width):
    """
    Beam search to grow the chain.
    Returns the best chain found.
    """
    start = (0,) * dimension
    # Each beam entry: (chain_list, occupied_set, pos_to_idx_dict, score)
    # score is number of H-H contacts formed so far
    beam = [([start], {start}, {start: 0}, 0)]
    
    for i in range(1, len(sequence)):
        new_beams = []
        current_residue = sequence[i]
        
        for chain, occupied, pos_to_idx, score in beam:
            last = chain[-1]
            for site in _neighbours(last):
                if site in occupied:
                    continue
                
                # Calculate new score
                new_score = score
                if current_residue == "H":
                    for q in _neighbours(site):
                        if q in occupied:
                            j = pos_to_idx.get(q)
                            if j is not None and j < i - 1 and sequence[j] == "H":
                                new_score += 1
                
                # Create new state
                # Copying is expensive, but necessary for branching
                new_chain = chain + [site]
                new_occupied = occupied | {site}
                new_pos_to_idx = dict(pos_to_idx)
                new_pos_to_idx[site] = i
                
                # Add small random factor for tie-breaking
                new_beams.append((new_chain, new_occupied, new_pos_to_idx, new_score + random.random() * 0.01))
                
        if not new_beams:
            return None
            
        # Sort by score (descending) and keep top beam_width
        new_beams.sort(key=lambda x: x[3], reverse=True)
        beam = new_beams[:beam_width]
        
    if not beam:
        return None
        
    return beam[0][0]

def _local_search(sequence, dimension, initial_chain, initial_energy, time_budget):
    """
    Monte Carlo local search to improve the chain.
    """
    chain = list(initial_chain)
    current_energy = initial_energy
    where = {p: i for i, p in enumerate(chain)}
    
    def try_move(chain, i, new_pos, where):
        """Try moving residue i to new_pos. Returns (new_chain, new_energy, new_where) or None."""
        if new_pos in where:
            return None
        old_pos = chain[i]
        
        # Check connectivity
        if i > 0:
            if abs(new_pos[0] - chain[i-1][0]) + abs(new_pos[1] - chain[i-1][1]) + abs(new_pos[2] - chain[i-1][2]) != 1:
                return None
        if i < len(chain) - 1:
            if abs(new_pos[0] - chain[i+1][0]) + abs(new_pos[1] - chain[i+1][1]) + abs(new_pos[2] - chain[i+1][2]) != 1:
                return None
            
        # Update chain
        new_chain = list(chain)
        new_chain[i] = new_pos
        
        # Update where
        new_where = dict(where)
        del new_where[old_pos]
        new_where[new_pos] = i
        
        # Recalculate energy
        new_energy = _energy(sequence, new_chain)
        
        return new_chain, new_energy, new_where

    start_time = time.time()
    while time.time() - start_time < time_budget:
        # Pick random residue (interior)
        i = random.randint(1, len(chain) - 2)
        old_pos = chain[i]
        
        # Try all 6 neighbors
        moved = False
        for new_pos in _neighbours(old_pos):
            res = try_move(chain, i, new_pos, where)
            if res is None:
                continue
            new_chain, new_energy, new_where = res
            
            if new_energy < current_energy:
                chain = new_chain
                current_energy = new_energy
                where = new_where
                moved = True
                break
            elif new_energy == current_energy and random.random() < 0.1:
                chain = new_chain
                current_energy = new_energy
                where = new_where
                moved = True
                break
        
        if not moved:
            # If no move, try a random jump? Or just continue.
            pass
            
    return chain

def fold(sequence, dimension):
    """
    Fold the sequence using Beam Search and Local Search polish.
    """
    n = len(sequence)
    
    # Determine beam width based on sequence length to manage time
    if n <= 50:
        width = 1000
    elif n <= 100:
        width = 300
    else:
        width = 100
        
    best_chain = None
    best_energy = 0
    
    start_time = time.time()
    # Allow up to 10 seconds per fold call to be safe (26 * 10 = 260s < 300s)
    deadline = start_time + 10.0 
    
    # Run Beam Search
    chain = _beam_search(sequence, dimension, width)
    if chain:
        best_chain = chain
        best_energy = _energy(sequence, chain)
        
    # Local Search Polish
    if best_chain and time.time() < deadline:
        time_left = deadline - time.time()
        # Use at least 0.5s for local search if available
        if time_left > 0.5:
            best_chain = _local_search(sequence, dimension, best_chain, best_energy, time_budget=time_left)
        
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        
    return [list(p) for p in best_chain]
