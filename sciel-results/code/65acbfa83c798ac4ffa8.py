"""HP lattice protein folding using CSP with backjumping."""
# EVOLVE-BLOCK-START
import time
import sys
import random
from collections import defaultdict

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _compute_energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _fold_csp(sequence, dimension, time_limit=10.0, seed=42):
    n = len(sequence)
    start_time = time.time()
    random.seed(seed)
    
    best_chain = None
    best_energy = 0  # We want most negative (most contacts)
    
    # Symmetry breaking: 
    # Residue 0 at (0,0)
    # Residue 1 at (1,0) (if n > 1)
    
    def _search(chain, occupied, current_energy):
        nonlocal best_chain, best_energy, start_time
        
        if time.time() - start_time > time_limit:
            return
        
        i = len(chain)
        if i == n:
            # Complete chain found
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
            return
            
        last_pos = chain[-1]
        current_residue = sequence[i]
        
        # Generate candidates
        candidates = []
        for site in _neighbours(last_pos):
            if site in occupied:
                continue
            
            # Forward Checking: Does the next residue have a valid move?
            # If i == n-1, we don't need to check further.
            if i < n - 1:
                next_has_move = False
                for next_site in _neighbours(site):
                    if next_site not in occupied and next_site != last_pos:
                        next_has_move = True
                        break
                if not next_has_move:
                    continue

            # Heuristic Scoring
            score = 0.0
            
            # 1. Immediate Contact Gain
            if current_residue == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if sequence[j] == 'H':
                            score += 20.0
            
            # 2. Free Space / Trap Avoidance
            free_count = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free_count += 1
            
            # If placing here leaves 0 free neighbors for the next residue (and i < n-1), it's a dead end
            if free_count == 0 and i < n - 1:
                continue # Prune immediately
            
            # Prefer sites with more free space to avoid trapping
            score += free_count * 2.0
            
            # 3. Penalty for low free space if H
            if current_residue == 'H' and free_count <= 1:
                score -= 15.0
                
            # 4. Compactness: Prefer moves that keep the chain close to the origin
            # This encourages compact folding
            dist = abs(site[0]) + abs(site[1])
            score -= dist * 0.5
            
            # 5. Randomness for diversity in restarts
            score += random.random() * 5.0 # Increased noise for better exploration
            
            candidates.append((score, site))
            
        if not candidates:
            return
            
        # Sort by score descending
        candidates.sort(key=lambda x: -x[0])
        
        for score, site in candidates:
            # Try placing at site
            chain.append(site)
            occupied[site] = i
            
            # Calculate new energy increment
            delta_energy = 0
            if current_residue == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if sequence[j] == 'H':
                            delta_energy -= 1 # Energy decreases (improves) by 1 per contact
            
            _search(chain, occupied, current_energy + delta_energy)
            
            # Backtrack
            chain.pop()
            del occupied[site]
            
            if time.time() - start_time > time_limit:
                return

    # Run search
    init_chain = [(0, 0)]
    init_occupied = {(0, 0): 0}
    
    if n > 1:
        init_chain.append((1, 0))
        init_occupied[(1, 0)] = 1
        _search(init_chain, init_occupied, 0)
    else:
        best_chain = init_chain[:]
        best_energy = 0

    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = 0
    else:
        best_energy = _compute_energy(sequence, best_chain)

    return best_chain, best_energy

def fold(sequence, dimension):
    """Fold an HP sequence using CSP-inspired search with local refinement."""
    n = len(sequence)
    
    # Time budget per sequence
    if n <= 20:
        time_limit = 10.0
    elif n <= 40:
        time_limit = 8.0
    elif n <= 60:
        time_limit = 6.0
    else:
        time_limit = 4.0
        
    # Multi-restart strategy to explore different parts of the search space
    num_restarts = 5 if n > 40 else 10
    time_per_restart = time_limit / num_restarts
    
    best_chain = None
    best_energy = 0
    global_start = time.time()
    
    for i in range(num_restarts):
        seed = 42 + i
        chain, energy = _fold_csp(sequence, dimension, time_limit=time_per_restart, seed=seed)
        
        if chain is not None and energy < best_energy:
            best_energy = energy
            best_chain = chain
            
        # Check if we are running out of total time
        if time.time() - global_start > time_limit * 0.9:
            break
            
    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = 0
    else:
        best_energy = _compute_energy(sequence, best_chain)

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
