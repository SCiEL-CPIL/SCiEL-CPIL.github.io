import random
import time
import sys

# EVOLVE-BLOCK-START

def _neighbours(point):
    """Yield the 6 neighbors of a point in 3D cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _calc_energy(sequence, chain, where):
    """Calculate the energy of a conformation.
    Energy = - (number of H-H contacts)
    """
    contacts = 0
    n = len(chain)
    for i in range(n):
        if sequence[i] == 'H':
            p = chain[i]
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _local_search(sequence, chain, where, rng, max_steps=1000, time_limit=None, start_time=None):
    """Perform local search to minimize energy.
    Returns the best chain found and its energy.
    """
    current_chain = list(chain)
    current_where = dict(where)
    current_energy = _calc_energy(sequence, current_chain, current_where)
    
    best_chain = list(current_chain)
    best_where = dict(current_where)
    best_energy = current_energy
    
    n = len(sequence)
    steps = 0
    
    while steps < max_steps:
        if time_limit and start_time:
            if time.time() - start_time > time_limit:
                break
        
        # Try to move a random residue
        i = rng.randint(1, n - 2)  # Avoid ends for now, simpler
        p = current_chain[i]
        
        # Try a random neighbor
        neighbors = list(_neighbours(p))
        rng.shuffle(neighbors)
        
        moved = False
        for new_p in neighbors:
            if new_p in current_where:
                continue
            
            # Check if the move is valid
            # The residue at i moves from p to new_p.
            # We need to ensure that the new position is not occupied by any other residue.
            # Since we are moving a single residue, we just need to check if new_p is free.
            # However, we must also ensure that the chain remains connected.
            # Moving a single residue in the middle of the chain breaks the chain connectivity 
            # unless we consider it as a "flip" or "move" that preserves the sequence order.
            # Actually, standard local search for self-avoiding walks often involves 
            # moving a segment or using specific moves like "pivots" or "crankshaft".
            # Moving a single internal residue is not a valid self-avoiding walk move 
            # because it breaks the bond between i-1 and i, and i and i+1.
            
            # Let's use a simpler local search: try to move the end of the chain? 
            # No, that's just regrowth.
            
            # Better: Try to swap two non-adjacent residues? No, that changes sequence.
            
            # Let's stick to a simpler heuristic: 
            # Try to move a small segment (e.g., 1-3 residues) to a new location.
            # This is complex.
            
            # Alternative: Use a "pivot" move.
            # Pick a random pivot point i (1 to n-1).
            # Rotate the segment from i+1 to n-1 around the bond (i-1, i) or (i, i+1)?
            # This is also complex to implement correctly for self-avoidance.
            
            # Let's try a simpler approach: 
            # Randomly select a residue i (1 to n-2).
            # Try to move it to a new position such that it is still adjacent to i-1 and i+1?
            # That's impossible in 3D unless it's a specific move.
            
            # Let's use a different local search: 
            # Try to move the entire tail of the chain?
            # Pick a random index k. Move the chain from k to n-1 to a new location.
            # This is a "tail move".
            
            k = rng.randint(1, n - 1)
            tail = current_chain[k:]
            pivot = current_chain[k-1]
            
            # Generate new positions for the tail
            # This is essentially regrowing the tail.
            # We can do a quick randomized regrowth of the tail.
            
            new_tail = [pivot]
            new_occupied = set(current_where.keys())
            new_occupied.remove(current_chain[k-1]) # Remove pivot from occupied for regrowth? No, pivot is fixed.
            # Actually, we need to remove the old tail from occupied.
            for p in tail:
                new_occupied.remove(p)
            
            # Regrow the tail
            success = True
            for j in range(k, n):
                options = []
                for site in _neighbours(new_tail[-1]):
                    if site in new_occupied:
                        continue
                    # Score: energy gain
                    gain = 0
                    if sequence[j] == 'H':
                        for q in _neighbours(site):
                            if q in new_occupied:
                                idx = current_where.get(q)
                                if idx is not None and idx < k - 1 and sequence[idx] == 'H':
                                    gain += 1
                    options.append((gain + rng.random() * 0.1, site))
                
                if not options:
                    success = False
                    break
                
                # Pick best or random
                # For local search, we might want to be greedy or random
                # Let's pick the best with some randomness
                options.sort(reverse=True)
                # Pick from top 3 if available
                top_k = min(3, len(options))
                chosen = rng.choice(options[:top_k])
                site = chosen[1]
                new_tail.append(site)
                new_occupied.add(site)
            
            if success and len(new_tail) == n - k + 1:
                # Construct new chain
                new_chain = current_chain[:k] + new_tail[1:]
                # Check self-avoidance (should be guaranteed by regrowth)
                # Calculate energy
                new_where = {p: i for i, p in enumerate(new_chain)}
                new_energy = _calc_energy(sequence, new_chain, new_where)
                
                if new_energy < current_energy:
                    current_chain = new_chain
                    current_where = new_where
                    current_energy = new_energy
                    
                    if current_energy < best_energy:
                        best_chain = list(current_chain)
                        best_where = dict(current_where)
                        best_energy = current_energy
                
                # Even if not better, we might accept it with some probability (simulated annealing)?
                # For now, let's stick to hill climbing.
        
        steps += 1
    
    return best_chain, best_energy, best_where

def _beam_search(sequence, dimension, rng, beam_width=50, time_limit=None, start_time=None):
    """Perform beam search to generate a full conformation.
    Returns the best chain found and its energy.
    """
    n = len(sequence)
    if n == 0:
        return [], 0, {}
    
    # Initial state
    start = (0, 0, 0)
    chains = [ [start] ]
    wheres = [ {start: 0} ]
    
    for i in range(1, n):
        if time_limit and start_time:
            if time.time() - start_time > time_limit:
                # Return the best chain so far (extend randomly?)
                # For simplicity, just return the best from previous step if possible
                # Or break and return best found
                pass
        
        new_chains = []
        new_wheres = []
        
        for chain, where in zip(chains, wheres):
            last = chain[-1]
            options = []
            for site in _neighbours(last):
                if site in where:
                    continue
                # Score: energy gain + randomness
                gain = 0
                if sequence[i] == 'H':
                    for q in _neighbours(site):
                        if q in where:
                            idx = where.get(q)
                            if idx is not None and idx < i - 1 and sequence[idx] == 'H':
                                gain += 1
                score = gain + rng.random() * 0.5
                options.append((score, site))
            
            if not options:
                continue
            
            # Sort by score
            options.sort(reverse=True)
            # Keep top beam_width candidates from this chain
            # Actually, we want to keep top beam_width globally
            # So we add all to a global list
            for score, site in options:
                new_chain = chain + [site]
                new_where = dict(where)
                new_where[site] = i
                new_chains.append((score, new_chain))
                new_wheres.append(new_where)
        
        if not new_chains:
            # Dead end, return best from previous step?
            # Or fail
            return None, 0, {}
        
        # Sort all new chains by score
        combined = list(zip(new_chains, new_wheres))
        combined.sort(key=lambda x: x[0][0], reverse=True)
        
        # Keep top beam_width
        best_combined = combined[:beam_width]
        
        chains = [c[1] for c in best_combined]
        wheres = [w for c, w in best_combined]
        
        # If we are at the last step, we have full chains
        if i == n - 1:
            # Evaluate all full chains
            best_chain = None
            best_energy = 0
            best_where = None
            for chain, where in zip(chains, wheres):
                energy = _calc_energy(sequence, chain, where)
                if best_chain is None or energy < best_energy:
                    best_chain = chain
                    best_energy = energy
                    best_where = where
            return best_chain, best_energy, best_where
    
    # If loop finishes without returning (shouldn't happen if n>1)
    return chains[0], _calc_energy(sequence, chains[0], wheres[0]), wheres[0]

def fold(sequence, dimension):
    """Fold the sequence on the lattice."""
    start_time = time.time()
    deadline = 240  # Safety margin for 330s limit
    
    n = len(sequence)
    if n == 0:
        return []
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0
    best_where = None
    
    # Number of restarts
    # We will do multiple beam search + local search cycles
    # Time budget per cycle: ~ (deadline - elapsed) / remaining_cycles
    # Let's just run until time is up.
    
    cycle = 0
    while time.time() - start_time < deadline:
        cycle += 1
        # Beam search
        chain, energy, where = _beam_search(sequence, dimension, rng, beam_width=50, 
                                            time_limit=deadline - (time.time() - start_time), 
                                            start_time=start_time)
        if chain is None:
            continue
        
        # Local search
        # Time limit for local search: remaining time / 2
        local_time_limit = max(1, (deadline - (time.time() - start_time)) / 2)
        ls_chain, ls_energy, ls_where = _local_search(sequence, chain, where, rng, 
                                                      max_steps=1000, 
                                                      time_limit=local_time_limit, 
                                                      start_time=start_time)
        
        if ls_energy < best_energy:
            best_chain = ls_chain
            best_energy = ls_energy
            best_where = ls_where
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
