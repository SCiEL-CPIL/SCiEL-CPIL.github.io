"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Compute the energy (negative number of H-H contacts)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _greedy_grow_with_lookahead(sequence, dimension, rng):
    """
    Greedy growth with a lookahead heuristic.
    Scores candidates based on immediate H-H contacts and potential future contacts.
    """
    n = len(sequence)
    if n <= 1:
        return [(0,) * dimension]

    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        candidates = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            
            # Score 1: Immediate H-H contacts
            score = 0.0
            if sequence[i] == 'H':
                for nb in _neighbours(site):
                    if nb in occupied:
                        idx = occupied[nb]
                        if idx < i - 1 and sequence[idx] == 'H':
                            score += 1.0
            
            # Score 2: Lookahead - count H residues in the 2-neighborhood
            # This encourages placing H in areas where other H's might end up.
            if sequence[i] == 'H':
                for nb1 in _neighbours(site):
                    if nb1 not in occupied:
                        for nb2 in _neighbours(nb1):
                            if nb2 in occupied:
                                idx = occupied[nb2]
                                if idx < i - 1 and sequence[idx] == 'H':
                                    score += 0.1
            else:
                # For P residues, prefer areas with more free space to avoid trapping
                free_space = sum(1 for nb in _neighbours(site) if nb not in occupied)
                score += 0.05 * free_space
            
            # Add random noise to break ties and explore different paths
            score += rng.random() * 0.5
            
            candidates.append((score, site))
        
        if not candidates:
            return None
        
        # Pick the best candidate
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_site = candidates[0][1]
        
        occupied[best_site] = i
        chain.append(best_site)
        
    return chain


def _local_search(sequence, chain, max_iterations=1000):
    """
    Local search to improve the energy of a given chain.
    Tries to move individual residues to adjacent empty spots.
    """
    n = len(sequence)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    improved = True
    iterations = 0
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Try to move each residue (except the first and last to keep ends fixed? 
        # Or allow all moves? Moving ends can be tricky. Let's try all.)
        # To maintain self-avoidance, we must check if the new position is free
        # and if the move doesn't create a collision.
        
        for i in range(n):
            current_pos = current_chain[i]
            
            # Try moving to each neighbor
            for nb in _neighbours(current_pos):
                # Check if nb is occupied by another residue
                if nb in occupied:
                    # If it's occupied by itself, skip (shouldn't happen)
                    # If occupied by another, skip
                    continue
                
                # Potential new chain
                # We need to check if moving i to nb creates a collision.
                # Since nb is not in occupied, the only collision risk is if
                # the move causes the chain to intersect itself elsewhere?
                # No, moving a single point to an empty spot only affects
                # the local neighborhood. The rest of the chain is unchanged.
                # The only check needed is that nb is not occupied.
                
                # However, we must ensure the chain remains a valid walk.
                # Moving residue i to nb might break the connectivity if
                # nb is not adjacent to i-1 or i+1.
                # Wait, the chain is a path. Residue i is connected to i-1 and i+1.
                # If we move i to nb, the new position must be adjacent to
                # the positions of i-1 and i+1? No, that's not required for
                # a self-avoiding walk. The walk is defined by the sequence of
                # points. If we change point i, the steps (i-1 -> i) and (i -> i+1)
                # change. They must still be unit steps.
                
                # So, for a move to be valid:
                # 1. nb is not occupied.
                # 2. If i > 0, nb must be adjacent to current_chain[i-1].
                # 3. If i < n-1, nb must be adjacent to current_chain[i+1].
                
                valid = True
                if i > 0:
                    prev_pos = current_chain[i-1]
                    if abs(nb[0] - prev_pos[0]) + abs(nb[1] - prev_pos[1]) != 1:
                        valid = False
                if valid and i < n - 1:
                    next_pos = current_chain[i+1]
                    if abs(nb[0] - next_pos[0]) + abs(nb[1] - next_pos[1]) != 1:
                        valid = False
                
                if not valid:
                    continue
                
                # Calculate new energy
                # This is expensive if we recalculate from scratch.
                # But for simplicity, let's do it.
                new_chain = list(current_chain)
                new_chain[i] = nb
                new_energy = _energy(sequence, new_chain)
                
                if new_energy < current_energy:
                    # Accept the move
                    current_chain = new_chain
                    current_energy = new_energy
                    # Update occupied
                    occupied = {p: idx for idx, p in enumerate(current_chain)}
                    improved = True
                    break # Restart the loop over residues? Or continue?
                    # Breaking and restarting ensures we always check from the beginning.
            
    return current_chain, current_energy


def fold(sequence, dimension):
    """
    Find the best conformation using randomized greedy growth with lookahead
    and local search optimization.
    """
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0 # Energy is negative, so we want more negative
    
    num_restarts = 500
    for _ in range(num_restarts):
        chain = _greedy_grow_with_lookahead(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Apply local search
        optimized_chain, energy = _local_search(sequence, chain)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = optimized_chain
    
    if best_chain is None:
        # Fallback to a straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
