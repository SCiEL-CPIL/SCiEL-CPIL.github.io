"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import sys

# Global deadline management
_DEADLINE = None
_TIME_BUDGET = 280.0  # Leave 50s margin for 330s limit

def _init_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _TIME_BUDGET

def _time_exceeded():
    return _DEADLINE is not None and time.time() >= _DEADLINE

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth with lookahead."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    n = len(sequence)
    
    for i in range(1, n):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            # Lookahead: check if this move traps the next residue
            trap_penalty = 0
            if i < n - 1:
                free_next = 0
                for q in _neighbours(site):
                    if q not in occupied:
                        free_next += 1
                if free_next == 0:
                    trap_penalty = 1000
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain + 0.1 * free - trap_penalty + rng.random()
            options.append((score, site))
        
        if not options:
            return None
        
        # Sort by score descending and pick from top candidates with some randomness
        options.sort(key=lambda x: -x[0])
        if len(options) > 1 and rng.random() < 0.3:
            # Pick from top 3 with weighted probability
            k = min(3, len(options))
            idx = rng.choices(range(k), weights=[1.0, 0.5, 0.25][:k])[0]
            site = options[idx][1]
        else:
            site = options[0][1]
        
        occupied[site] = i
        chain.append(site)
    
    return chain

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

def _local_search(sequence, chain, dimension, rng, max_moves=50):
    """Simple local search: try reversing segments or moving single residues."""
    n = len(chain)
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    occupied = {p: i for i, p in enumerate(chain)}
    
    for move_iter in range(max_moves):
        if _time_exceeded():
            break
        
        # Try a random move: pick a residue and try to relocate it
        if n < 4:
            break
            
        i = rng.randint(1, n - 2)  # Don't move ends
        orig_pos = chain[i]
        
        # Try moving residue i to a neighboring free site
        candidates = []
        for site in _neighbours(orig_pos):
            if site == chain[i-1] or site == chain[i+1]:
                continue
            if site in occupied:
                # Only allow if it's the original position
                if site != orig_pos:
                    continue
            
            # Check if we can place here
            new_occupied = occupied.copy()
            del new_occupied[orig_pos]
            if site in new_occupied:
                continue
            new_occupied[site] = i
            
            # Check self-avoidance for the moved residue
            ok = True
            for j in range(n):
                if j == i:
                    continue
                if chain[j] == site:
                    ok = False
                    break
            if not ok:
                continue
            
            # Check that the chain remains connected (unit steps)
            if abs(chain[i-1][0] - site[0]) + abs(chain[i-1][1] - site[1]) + abs(chain[i-1][2] - site[2]) != 1:
                continue
            if abs(chain[i+1][0] - site[0]) + abs(chain[i+1][1] - site[1]) + abs(chain[i+1][2] - site[2]) != 1:
                continue
            
            candidates.append(site)
        
        if not candidates:
            continue
        
        new_site = rng.choice(candidates)
        new_chain = list(chain)
        new_chain[i] = new_site
        
        # Verify self-avoidance
        seen = set()
        valid = True
        for p in new_chain:
            if p in seen:
                valid = False
                break
            seen.add(p)
        if not valid:
            continue
        
        # Verify unit steps
        for k in range(n-1):
            diff = sum(abs(new_chain[k][d] - new_chain[k+1][d]) for d in range(dimension))
            if diff != 1:
                valid = False
                break
        if not valid:
            continue
        
        new_energy = _energy(sequence, new_chain)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_chain = new_chain
            chain = new_chain
            occupied = {p: idx for idx, p in enumerate(chain)}
        else:
            # Accept with some probability for exploration
            if rng.random() < 0.1:
                chain = new_chain
                occupied = {p: idx for idx, p in enumerate(chain)}
    
    return best_chain, best_energy

def fold(sequence, dimension):
    """Best conformation using multiple randomized greedy growths plus local search."""
    _init_deadeline()
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 0
    
    n = len(sequence)
    
    # Determine number of restarts based on time budget
    # Estimate time per growth
    start_time = time.time()
    
    # First, do a few quick greedy runs to estimate
    num_greedy = 0
    for attempt in range(200):
        if _time_exceeded():
            break
        if time.time() - start_time > _TIME_BUDGET * 0.4:
            break
            
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        num_greedy += 1
        value = _energy(sequence, chain)
        if value < best_energy:
            best, best_energy = chain, value
    
    # Now use remaining time for local search on best candidates
    if best is not None and not _time_exceeded():
        search_chain, search_energy = _local_search(sequence, best, dimension, rng, max_moves=100)
        if search_energy < best_energy:
            best, best_energy = search_chain, search_energy
    
    if best is None:
        # Fallback: straight line
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _energy(sequence, best)
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
