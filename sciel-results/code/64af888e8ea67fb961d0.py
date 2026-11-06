# EVOLVE-BLOCK-START
import random
import time
import sys

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

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

def _grow(sequence, dimension, rng):
    """
    Randomized greedy growth.
    Returns a list of points (tuples) or None if stuck.
    """
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    n = len(sequence)
    
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            # Strong bias for contacts, weak bias for free space
            score = gain * 10.0 + free * 0.1 + rng.random()
            options.append((score, site))
        if not options:
            return None
        # Pick best
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _local_optimize(sequence, chain, dimension, rng, deadline):
    """
    Optimize chain using pivot moves (tail rotation).
    Returns the best chain found.
    """
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    n = len(chain)
    
    # Precompute H positions for faster energy calculation? 
    # For now, full recalculation is O(N*6) which is fast enough for N~100-136 in Python if done carefully.
    # But we can optimize _energy by passing the where dict.
    
    # Let's optimize _energy to take a where dict to avoid rebuilding it every time.
    # However, the where dict changes with every pivot. 
    # Rebuilding where dict is O(N). 
    # Checking contacts is O(N * 6). 
    # Total O(N) per move. 
    # For N=136, this is ~800 operations. 
    # 10k moves = 8M operations. In Python, ~1-2 seconds. 
    # So we can do many moves.
    
    while time.time() < deadline:
        if n < 4:
            break
            
        # Pivot move
        pivot_idx = rng.randint(1, n - 2)
        p_prev = chain[pivot_idx - 1]
        p_pivot = chain[pivot_idx]
        
        v = [p_pivot[i] - p_prev[i] for i in range(dimension)]
        bond_axis = -1
        for i in range(dimension):
            if v[i] != 0:
                bond_axis = i
                break
        if bond_axis == -1: continue
                
        rot_axes = [i for i in range(dimension) if i != bond_axis]
        if not rot_axes: continue
        
        r_axis = rng.choice(rot_axes)
        angle = rng.choice([1, -1])
        
        # Rotate tail [pivot_idx+1 .. N-1]
        head_set = set(chain[:pivot_idx])
        new_tail = []
        valid = True
        occupied_tail = set()
        
        for i in range(pivot_idx + 1, n):
            orig = chain[i]
            dx = orig[0] - p_pivot[0]
            dy = orig[1] - p_pivot[1]
            dz = orig[2] - p_pivot[2]
            
            if r_axis == 0: # Rotate around X
                if angle == 1:
                    nx, ny, nz = dx, -dz, dy
                else:
                    nx, ny, nz = dx, dz, -dy
            elif r_axis == 1: # Rotate around Y
                if angle == 1:
                    nx, ny, nz = dz, dy, -dx
                else:
                    nx, ny, nz = -dz, dy, dx
            else: # Rotate around Z
                if angle == 1:
                    nx, ny, nz = -dy, dx, dz
                else:
                    nx, ny, nz = dy, -dx, dz
            
            new_point = (p_pivot[0] + nx, p_pivot[1] + ny, p_pivot[2] + nz)
            
            if new_point in head_set:
                valid = False
                break
            if new_point in occupied_tail:
                valid = False
                break
            
            if new_tail:
                prev = new_tail[-1]
                if abs(new_point[0]-prev[0]) + abs(new_point[1]-prev[1]) + abs(new_point[2]-prev[2]) != 1:
                    valid = False
                    break
            
            occupied_tail.add(new_point)
            new_tail.append(new_point)
            
        if not valid:
            continue
        
        candidate_chain = chain[:pivot_idx + 1] + new_tail
        new_energy = _energy(sequence, candidate_chain)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_chain = candidate_chain
            chain = candidate_chain

    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Uses a global time budget strategy.
    1. Generate multiple initial chains via randomized growth.
    2. Optimize each one for a fixed amount of time.
    3. Keep the best result.
    
    Total time limit: 330s wall-clock for the whole program.
    Since fold is called once per sequence, and there are 26 sequences,
    we should reserve a global deadline.
    However, the contract says "240 s CPU budget for all sequences together" and "300s wall-clock".
    Let's assume the 240s is the hard limit for the sum of all calls.
    To be safe, let's track the time used by previous calls? 
    No, the subprocess is killed after 330s wall-clock.
    If we don't know the order, we can't easily share state.
    BUT, we can use a heuristic: 
    If we assume the calls are sequential, we can estimate the remaining time.
    However, a safer approach for a single call is to use a fixed budget that sums to < 240s.
    240 / 26 ≈ 9.2 seconds.
    Let's use 9.0 seconds per call to be safe.
    """
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0
    start_time = time.time()
    
    # Limit each call to 9.0 seconds to fit 26 calls in 240s (with margin).
    call_deadline = time.time() + 9.0
    
    n_restarts = 10
    # For long chains, maybe fewer restarts but longer optimization?
    # Or just fixed time per restart?
    # Let's give each restart a slice of the time budget.
    
    for i in range(n_restarts):
        if time.time() > call_deadline - 1.0: # Leave 1s for last one to finish
            break
            
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Optimization budget for this restart
        # We want to spend the remaining time on this restart if it's the last one,
        # or a fair share if not.
        # Simple strategy: each restart gets (call_deadline - start_time) / n_restarts
        # But if we are late, give more to the current one?
        # Let's just give a fixed time slice, say 0.8s per restart? 10 * 0.8 = 8s.
        # This leaves 1s margin.
        
        opt_deadline = time.time() + 0.8
        
        optimized_chain, opt_energy = _local_optimize(sequence, chain, dimension, rng, opt_deadline)
        
        if optimized_chain is not None:
            if best_chain is None or opt_energy < best_energy:
                best_chain = optimized_chain
                best_energy = opt_energy

    if best_chain is None:
        # Fallback: linear chain
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
