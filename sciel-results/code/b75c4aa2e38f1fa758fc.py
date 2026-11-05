"""HP lattice protein folding using O(1)-delta heuristic local search."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Global deadline guard
_DEADLINE = None
_BUDGET_FRACTION = 0.95  # Use 95% of the 330s budget to be safe

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 330 * _BUDGET_FRACTION
    return _DEADLINE

def _is_deadline_exceeded():
    return time.time() > _get_deadline()

def _neighbours(point):
    """Yield the 6 neighbours of a point in 3D."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Compute energy as -number of H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(sequence)
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _initial_grow(sequence, dimension, rng):
    """Generate an initial self-avoiding walk using randomized greedy growth."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    
    # Start at origin
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Score: prefer positions with more H-H contacts and more free space
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == "H":
                            gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 2.0 + free * 0.1 + rng.random()
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick best option
        options.sort(reverse=True)
        site = options[0][1]
        chain.append(site)
        occupied[site] = i
    
    return chain

def _local_search(sequence, chain, dimension, rng, time_limit=1.0):
    """
    Improve a chain using local search with O(1) delta energy computation.
    Uses a contact map to track H-H interactions.
    """
    n = len(sequence)
    if n <= 2:
        return chain
    
    # Build position map and contact map
    pos_map = {p: i for i, p in enumerate(chain)}
    is_h = [1 if s == 'H' else 0 for s in sequence]
    
    # contact_map[i] is a set of j > i+1 such that i and j are H-H contacts
    contact_map = [set() for _ in range(n)]
    current_energy = 0
    
    for i in range(n):
        if is_h[i]:
            for q in _neighbours(chain[i]):
                j = pos_map.get(q)
                if j is not None and j > i + 1 and is_h[j]:
                    contact_map[i].add(j)
                    contact_map[j].add(i)
                    current_energy -= 1
    
    best_energy = current_energy
    best_chain = list(chain)
    
    start_time = time.time()
    no_improvement_count = 0
    
    while time.time() - start_time < time_limit and not _is_deadline_exceeded():
        # Pick a random internal residue
        i = rng.randint(1, n - 2)
        
        p_i = chain[i]
        
        # Try all 6 neighbours
        best_move_site = None
        best_delta = 0 # We want to minimize energy (make it more negative), so delta < 0 is good
        
        for site in _neighbours(p_i):
            # Check if site is valid
            if site in pos_map:
                continue
            if i > 0 and site == chain[i-1]:
                continue
            if i < n-1 and site == chain[i+1]:
                continue
            
            # Check if moving p_i to site is valid
            # Must be adjacent to chain[i-1] and chain[i+1]
            valid = True
            if i > 0:
                if abs(site[0]-chain[i-1][0]) + abs(site[1]-chain[i-1][1]) + abs(site[2]-chain[i-1][2]) != 1:
                    valid = False
            if valid and i < n-1:
                if abs(site[0]-chain[i+1][0]) + abs(site[1]-chain[i+1][1]) + abs(site[2]-chain[i+1][2]) != 1:
                    valid = False
            
            if not valid:
                continue
            
            # Calculate delta energy
            # Remove contacts involving i
            delta = 0
            if is_h[i]:
                for j in contact_map[i]:
                    delta += 1 # Removing a contact increases energy (less negative)
            
            # Add new contacts involving i
            if is_h[i]:
                for q in _neighbours(site):
                    j = pos_map.get(q)
                    if j is not None and j > i + 1 and is_h[j]:
                        # Check if this contact already exists? No, because site is not occupied.
                        # But wait, could it be a contact that was just removed?
                        # We removed contacts with old p_i. site != p_i.
                        # So these are new contacts.
                        delta -= 1 # Adding a contact decreases energy (more negative)
            
            # If delta is better (more negative) than current best, track it
            if delta < best_delta:
                best_delta = delta
                best_move_site = site
        
        if best_move_site is not None:
            # Apply the best move
            old_site = chain[i]
            chain[i] = best_move_site
            del pos_map[old_site]
            pos_map[best_move_site] = i
            
            # Update contact map
            if is_h[i]:
                # Remove old contacts
                for j in list(contact_map[i]):
                    contact_map[i].discard(j)
                    contact_map[j].discard(i)
                
                # Add new contacts
                for q in _neighbours(best_move_site):
                    j = pos_map.get(q)
                    if j is not None and j > i + 1 and is_h[j]:
                        contact_map[i].add(j)
                        contact_map[j].add(i)
            
            current_energy += best_delta
            no_improvement_count = 0
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        else:
            no_improvement_count += 1
            # If stuck for too long, maybe do a random move or break?
            # For simplicity, we just continue. The random pick of i will eventually find a move.
            # If we want to escape local minima, we could implement a simulated annealing acceptance here.
            # But for now, let's keep it greedy/hill-climbing.
    
    return best_chain

def fold(sequence, dimension):
    """Best conformation using multi-start greedy growth and local search."""
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 1
    
    n = len(sequence)
    # Allocate time based on chain length?
    # Longer chains take more time per iteration.
    # Let's just use a fixed time limit per restart and let the global deadline handle it.
    time_per_restart = 2.0
    
    # Multiple restarts
    # We can increase restarts if we are fast.
    for restart in range(50):
        if _is_deadline_exceeded():
            break
        
        chain = _initial_grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Apply local search to improve
        chain = _local_search(sequence, chain, dimension, rng, time_limit=time_per_restart)
        
        value = _energy(sequence, chain)
        if value < best_energy:
            best, best_energy = chain, value
            
        # If we found a very good solution, we can stop early?
        # For now, keep going to find better ones.
    
    if best is None:  # a straight line is always self-avoiding
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
