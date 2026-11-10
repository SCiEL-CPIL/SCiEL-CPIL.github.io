"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time


def _neighbours2d(point):
    """Yield the 4 unit-step neighbours of a 2D point (tuple)."""
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _energy(sequence, chain):
    """Compute the hydrophobic contact energy of a conformation."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours2d(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _greedy_grow(sequence, dimension, rng, temperature=0.0):
    """
    One randomized greedy chain growth.
    Prefer sites that create new H-H contacts, then free space.
    """
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours2d(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                gain = sum(1 for q in _neighbours2d(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == 'H')
            free = sum(1 for q in _neighbours2d(site) if q not in occupied)
            # Score: gain is primary, free space is secondary, randomness breaks ties
            score = gain * 10.0 + free * 0.5 + rng.random() * temperature
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick the best option
        best_score, best_site = max(options)
        occupied[best_site] = i
        chain.append(best_site)
    
    return chain


def _local_search(sequence, chain, max_iters=5000, rng=None):
    """
    Local search using single-residue moves (kinks and loops).
    Returns the best chain found.
    """
    if rng is None:
        rng = random.Random()
    
    best_chain = list(chain)
    best_energy = _energy(sequence, best_chain)
    current_chain = list(chain)
    current_energy = best_energy
    
    n = len(sequence)
    where = {p: i for i, p in enumerate(current_chain)}
    
    for iteration in range(max_iters):
        # Pick a random residue to move (not the first or last)
        i = rng.randint(1, n - 2)
        old_pos = current_chain[i]
        
        # Try moving to each neighbour of old_pos
        moved = False
        for new_pos in _neighbours2d(old_pos):
            if new_pos == current_chain[i - 1] or new_pos == current_chain[i + 1]:
                continue
            if new_pos in where:
                # Check if the occupant is the residue we're moving (it shouldn't be, but just in case)
                if where[new_pos] == i:
                    continue
                # Check if new_pos is adjacent to any other part of the chain that would cause a collision
                # Actually, we just need to check if new_pos is free
                # But we also need to ensure that removing old_pos and adding new_pos doesn't create a self-intersection
                # The main check is: is new_pos in where? If so, skip.
                # Also, we must ensure that the path from i-1 to i+1 via new_pos is valid (no immediate backtracking)
                # We already checked new_pos != chain[i-1] and new_pos != chain[i+1]
                
                # Check if new_pos is occupied by another residue
                if where.get(new_pos, -1) != -1:
                    continue
                
                # Tentative move
                # Remove old_pos from where
                del where[old_pos]
                # Check if new_pos is free (it should be, since we checked above)
                # But wait, we need to make sure that the move doesn't cause the chain to intersect itself elsewhere.
                # Since we are only moving one residue, the only potential intersections are with the immediate neighbours
                # which we already checked. The rest of the chain is unchanged.
                # So the move is valid if new_pos is not in where (after removing old_pos).
                
                where[new_pos] = i
                current_chain[i] = new_pos
                
                new_energy = _energy(sequence, current_chain)
                
                if new_energy < current_energy:
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(current_chain)
                    moved = True
                    break
                else:
                    # Revert
                    del where[new_pos]
                    where[old_pos] = i
                    current_chain[i] = old_pos
        
        if not moved:
            # Try a random kink move (move residue i to a random free neighbour)
            # This is essentially the same as above, but with a different selection strategy
            # For now, if no improvement, we can accept a random move with some probability (simulated annealing)
            # But for simplicity, we'll just continue to the next iteration
            pass
    
    return best_chain


def _regrow_tail(sequence, chain, start_index, rng):
    """
    Regrow the tail of the chain from start_index onwards.
    """
    if start_index >= len(sequence):
        return list(chain)
    
    # Keep the prefix
    prefix = chain[:start_index]
    occupied = {p: i for i, p in enumerate(prefix)}
    
    # Regrow from the last residue of the prefix
    chain_new = list(prefix)
    
    for i in range(start_index, len(sequence)):
        options = []
        for site in _neighbours2d(chain_new[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                gain = sum(1 for q in _neighbours2d(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == 'H')
            free = sum(1 for q in _neighbours2d(site) if q not in occupied)
            score = gain * 10.0 + free * 0.5 + rng.random() * 0.1
            options.append((score, site))
        
        if not options:
            return None
        
        best_score, best_site = max(options)
        occupied[best_site] = i
        chain_new.append(best_site)
    
    return chain_new


def fold(sequence, dimension):
    """
    Fold a sequence into a low-energy conformation.
    """
    if dimension != 2:
        raise ValueError("Only dimension 2 is supported")
    
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 110 seconds, leaving 10 seconds margin
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = float('inf')
    
    # Phase 1: Multiple greedy growths with different temperatures
    temperatures = [0.0, 0.1, 0.5, 1.0, 2.0]
    
    for temp in temperatures:
        for _ in range(20):
            if time.time() > deadline:
                break
            chain = _greedy_grow(sequence, dimension, rng, temperature=temp)
            if chain is None:
                continue
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = chain
        
        if time.time() > deadline:
            break
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [[i, 0] for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    # Phase 2: Local search on the best chain
    if time.time() < deadline:
        improved_chain = _local_search(sequence, best_chain, max_iters=10000, rng=rng)
        improved_energy = _energy(sequence, improved_chain)
        if improved_energy < best_energy:
            best_energy = improved_energy
            best_chain = improved_chain
    
    # Phase 3: Regrow tails from different points
    if time.time() < deadline:
        # Try regrowing from the middle
        mid = n // 2
        for _ in range(10):
            if time.time() > deadline:
                break
            new_chain = _regrow_tail(sequence, best_chain, mid, rng)
            if new_chain is not None:
                energy = _energy(sequence, new_chain)
                if energy < best_energy:
                    best_energy = energy
                    best_chain = new_chain
                # Local search on the new chain
                if time.time() < deadline:
                    improved = _local_search(sequence, new_chain, max_iters=2000, rng=rng)
                    improved_energy = _energy(sequence, improved)
                    if improved_energy < best_energy:
                        best_energy = improved_energy
                        best_chain = improved
    
    # Phase 4: More greedy growths with local search
    if time.time() < deadline:
        for _ in range(50):
            if time.time() > deadline:
                break
            chain = _greedy_grow(sequence, dimension, rng, temperature=0.5)
            if chain is None:
                continue
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = chain
            # Local search
            if time.time() < deadline:
                improved = _local_search(sequence, chain, max_iters=3000, rng=rng)
                improved_energy = _energy(sequence, improved)
                if improved_energy < best_energy:
                    best_energy = improved_energy
                    best_chain = improved
    
    # Convert to list of lists
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
