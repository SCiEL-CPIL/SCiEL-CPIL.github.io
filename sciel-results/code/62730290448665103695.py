import time
import random
import math

_START_TIME = time.time()
_TOTAL_BUDGET = 240.0
_WALL_CLOCK_LIMIT = 330.0
# Safety margin: return before 330s wall clock kill.

def _neighbours(point):
    x, y, z = point
    return [(x+1, y, z), (x-1, y, z), (x, y+1, z), (x, y-1, z), (x, y, z+1), (x, y, z-1)]

def _energy(sequence, chain):
    """
    Calculate the energy of a chain.
    Energy = - (number of H-H contacts)
    """
    if not chain:
        return 0
    N = len(chain)
    where = [None] * (N * N * N)  # Too large, use dict instead
    # Use dict for sparse lookup
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    
    h_set = set(i for i, c in enumerate(sequence) if c == 'H')
    contacts = 0
    for i in h_set:
        p = chain[i]
        x, y, z = p
        for dx, dy, dz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
            q = (x+dx, y+dy, z+dz)
            j = where.get(q)
            if j is not None and j > i + 1 and j in h_set:
                contacts += 1
    return -contacts

def _greedy_grow(sequence, rng, bias=1.0):
    """
    Greedy growth to create an initial self-avoiding walk.
    Incorporates free-space heuristic to avoid traps.
    """
    N = len(sequence)
    if N == 0:
        return []
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    h_set = set(i for i, c in enumerate(sequence) if c == 'H')
    
    for i in range(1, N):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            gain = 0
            free = 0
            if i in h_set:
                for q in _neighbours(site):
                    if q in occupied:
                        idx = occupied[q]
                        if idx < i - 1 and idx in h_set:
                            gain += 1
                    else:
                        free += 1
            else:
                # Even for P, prefer some free space to avoid dead ends
                for q in _neighbours(site):
                    if q not in occupied:
                        free += 1

            # Score: maximize gain, then free space, then noise
            score = gain * 10.0 + free * 0.1 + rng.random()
            options.append((score, site))
        
        if not options:
            return None
            
        best_score, best_site = max(options)
        chain.append(best_site)
        occupied[best_site] = i
        
    return chain

def _local_optimize(sequence, chain, rng, deadline):
    """
    Local optimization using pivot moves for quick polishing.
    """
    N = len(chain)
    if N < 10:
        return chain, _energy(sequence, chain)
    
    current_chain = chain[:]
    current_energy = _energy(sequence, chain)
    best_chain = current_chain[:]
    best_energy = current_energy
    
    h_set = set(i for i, c in enumerate(sequence) if c == 'H')
    
    while time.time() < deadline:
        # Try a pivot move: cut at position i, reverse the segment, and try to regrow
        if N > 6:
            i = rng.randint(1, N - 3)
            # Reverse the tail from position i
            head = current_chain[:i]
            tail = current_chain[i:]
            
            # Try regrowing the tail with the reversed direction
            new_tail = []
            occupied = set(head)
            current_pos = head[-1]
            valid = True
            
            for k in range(i, N):
                options = []
                for site in _neighbours(current_pos):
                    if site in occupied:
                        continue
                    gain = 0
                    free = 0
                    if k in h_set:
                        for q in _neighbours(site):
                            if q in occupied:
                                idx = None
                                # Check if this position is in head
                                for idx_h, pos_h in enumerate(head):
                                    if pos_h == q:
                                        idx = idx_h
                                        break
                                if idx is not None and idx in h_set:
                                    gain += 1
                            else:
                                free += 1
                    else:
                        for q in _neighbours(site):
                            if q not in occupied:
                                free += 1
                    score = gain * 10.0 + free * 0.1 + rng.random()
                    options.append((score, site))
                
                if not options:
                    valid = False
                    break
                
                best_score, best_site = max(options)
                new_tail.append(best_site)
                occupied.add(best_site)
                current_pos = best_site
            
            if valid:
                new_chain = head + new_tail
                new_energy = _energy(sequence, new_chain)
                if new_energy < current_energy:
                    current_chain = new_chain
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = new_chain[:]
    
    return best_chain, best_energy

def fold(sequence, dimension):
    N = len(sequence)
    if N == 0:
        return []
    
    # Global time guard: ensure we don't exceed wall clock limit
    elapsed = time.time() - _START_TIME
    remaining = _WALL_CLOCK_LIMIT - elapsed - 1.0  # 1s safety margin
    if remaining < 1.0:
        # Not enough time, return a simple fallback
        return [[i, 0, 0] for i in range(N)]
    
    # Per-sequence time budget: use remaining time divided by estimated remaining sequences
    # We don't know how many sequences are left, so use a conservative estimate
    # Assume at most 26 total, use ~9s per sequence as baseline
    per_seq_budget = min(9.0, remaining)
    deadline = time.time() + per_seq_budget
    
    rng = random.Random(42)
    
    # Initialize with best of a few greedy chains
    best_chain = None
    best_energy = 0
    
    # Generate multiple diverse seeds
    num_seeds = 20
    for _ in range(num_seeds):
        if time.time() > deadline - 0.5:
            break
        chain = _greedy_grow(sequence, rng)
        if chain:
            e = _energy(sequence, chain)
            if best_chain is None or e < best_energy:
                best_chain = chain
                best_energy = e
                
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0, 0) for i in range(N)]
        best_energy = _energy(sequence, best_chain)
        
    current_chain = best_chain[:]
    current_energy = best_energy
    
    # Tail Regrowth Search with Simulated Annealing
    T = 2.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    # Precompute H indices for faster checks
    h_set = set(i for i, c in enumerate(sequence) if c == 'H')
    
    # Add a "kick" mechanism to escape local minima
    no_improvement_count = 0
    max_no_improvement = 1000
    
    while time.time() < deadline:
        # Pick a cut point
        # For longer chains, bias towards the tail to allow more frequent updates
        if N > 50:
            i = rng.randint(N // 2, N - 2)
        else:
            i = rng.randint(1, N - 2)
        
        head = current_chain[:i]
        
        # Prepare data structures for regrowth
        head_map = {p: idx for idx, p in enumerate(head)}
        occupied = set(head_map.keys())
        
        new_tail = []
        tail_map = {} # pos -> global index
        current_pos = head[-1]
        
        valid = True
        
        for k in range(i, N):
            options = []
            for site in _neighbours(current_pos):
                if site in occupied:
                    continue
                
                gain = 0
                free = 0
                if k in h_set:
                    for q in _neighbours(site):
                        if q in head_map:
                            idx_h = head_map[q]
                            if idx_h in h_set:
                                gain += 1
                        elif q in tail_map:
                            idx_t = tail_map[q]
                            if idx_t in h_set:
                                gain += 1
                        else:
                            free += 1
                else:
                    for q in _neighbours(site):
                        if q not in occupied:
                            free += 1

                # Score: maximize gain, then free space, then noise
                score = gain * 10.0 + free * 0.1 + rng.random()
                options.append((score, site))
                
            if not options:
                valid = False
                break
                
            best_score, best_site = max(options)
            new_tail.append(best_site)
            tail_map[best_site] = k
            occupied.add(best_site)
            current_pos = best_site
            
        if not valid:
            continue
            
        new_chain = head + new_tail
        new_energy = _energy(sequence, new_chain)
        
        # Metropolis Acceptance
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            current_chain = new_chain
            current_energy = new_energy
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = new_chain[:]
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min
        else:
            # Rejected
            no_improvement_count += 1
        
        if no_improvement_count > max_no_improvement:
            # Kick: restart temperature
            T = 2.0
            no_improvement_count = 0
            
    return [list(p) for p in best_chain]
