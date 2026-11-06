"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import sys

START_TIME = time.time()
TIME_LIMIT = 300.0  # Safety margin before 330s kill

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _compute_energy(sequence, chain):
    """Compute energy of a full chain."""
    if not chain:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _mccg_growth(sequence, dimension, temperature=1.0):
    """
    Monte Carlo Chain Growth:
    Grows the chain step-by-step. At each step, it samples valid moves
    with probabilities proportional to exp(-delta_E / T) where delta_E is the
    immediate energy change (contact gain).
    This creates a 'potential field' effect where the chain is drawn towards
    existing H residues.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    # Pre-compute H positions for quick check? No, just check sequence.
    
    for i in range(1, n):
        last_pos = chain[-1]
        valid_moves = []
        
        for site in _neighbours(last_pos):
            if site in occupied:
                continue
            
            # Calculate immediate energy change (delta E)
            # If sequence[i] is H, it gains contacts with existing H's
            # If sequence[i] is P, it gains 0
            delta_e = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            delta_e -= 1 # Energy decreases (good)
            
            valid_moves.append((delta_e, site))
        
        if not valid_moves:
            # Dead end. Return None to indicate failure
            return None
        
        # Sample move based on Boltzmann distribution
        # P(move) = exp(-delta_e / T) / sum(exp(-delta_e' / T))
        # Note: delta_e is negative for good moves. So -delta_e is positive.
        # We want to favor negative delta_e (energy gain).
        # So weight = exp(delta_e / T) ? No.
        # Energy E_new = E_old + delta_e.
        # We want to minimize E.
        # Standard Metropolis: accept if E_new < E_old always.
        # For sampling: P ~ exp(-E_new/kT).
        # Relative weight for move m: exp(-(E_old + delta_m)/kT) = exp(-E_old/kT) * exp(-delta_m/kT).
        # Since E_old is constant for this step, weight ~ exp(-delta_m/kT).
        # If delta_m is negative (gain), -delta_m is positive, weight > 1. Correct.
        
        weights = []
        max_log_w = -1000
        for delta_e, site in valid_moves:
            log_w = -delta_e / temperature
            if log_w > max_log_w:
                max_log_w = log_w
            weights.append((log_w, site))
        
        # Normalize weights (log-sum-exp trick to avoid underflow/overflow)
        sum_exp = 0.0
        exp_weights = []
        for log_w, site in weights:
            w = math.exp(log_w - max_log_w)
            exp_weights.append((w, site))
            sum_exp += w
        
        r = random.random() * sum_exp
        cumulative = 0.0
        chosen_site = exp_weights[-1][1]
        for w, site in exp_weights:
            cumulative += w
            if r <= cumulative:
                chosen_site = site
                break
        
        chain.append(chosen_site)
        occupied[chosen_site] = i
        
    return chain

def _local_search(sequence, chain, dimension, time_budget=1.0):
    """
    Monte Carlo Local Search using pivot moves.
    A pivot move takes a subchain and rotates it 90, 180, or 270 degrees
    around the pivot point.
    """
    if not chain:
        return chain, 0
    
    n = len(chain)
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    start_time = time.time()
    end_time = start_time + time_budget
    
    # Temperature for Metropolis
    T = 1.0
    
    # Pre-allocate for speed?
    # We need to check self-avoidance for the moved segment.
    
    while time.time() < end_time:
        # Choose pivot point
        # Pivot can be at any position i. The segment from i+1 to n-1 is rotated.
        # To ensure connectivity, we rotate around chain[i].
        # The new positions of j > i must not collide with any k <= i or other j' > i.
        
        if n < 2:
            break
            
        i = random.randint(0, n - 2)
        
        # Get the segment to rotate
        pivot = chain[i]
        segment = chain[i+1:]
        
        # Generate 3 possible rotations (90, 180, 270 deg)
        # In 3D, we can rotate around any axis.
        # Simple approach: rotate the vector (p - pivot) by 90 deg around x, y, or z.
        # Actually, standard pivot move in 3D:
        # The segment is a set of vectors relative to pivot.
        # We can apply rotation matrices.
        
        rotations = []
        
        # Rotation around X axis: (x, y, z) -> (x, -z, y) or (x, z, -y)
        # Let's just try all 24 orientations? No, that's expensive.
        # Let's try the 3 simple 90-degree rotations around the 3 axes.
        # Actually, a pivot move usually rotates the tail around the bond.
        # Let's stick to rotating the tail around the pivot point by 90 deg around one of the 3 axes.
        
        for axis in range(3):
            for sign in (-1, 1):
                # Rotate segment by 90 deg around 'axis' passing through 'pivot'
                # Vector v = p - pivot
                # New v:
                if axis == 0: # x
                    # y, z -> -z, y (sign 1) or z, -y (sign -1)
                    # Let's define rotation matrices properly or just swap coords
                    # Rot 90 around X: (x, y, z) -> (x, -z, y)
                    # Rot -90 around X: (x, y, z) -> (x, z, -y)
                    new_segment = []
                    for p in segment:
                        v = [p[0]-pivot[0], p[1]-pivot[1], p[2]-pivot[2]]
                        if sign == 1:
                            nv = [v[0], -v[2], v[1]]
                        else:
                            nv = [v[0], v[2], -v[1]]
                        new_p = [pivot[0]+nv[0], pivot[1]+nv[1], pivot[2]+nv[2]]
                        new_segment.append(tuple(new_p))
                elif axis == 1: # y
                    # x, z -> z, -x (sign 1) or -z, x (sign -1)
                    new_segment = []
                    for p in segment:
                        v = [p[0]-pivot[0], p[1]-pivot[1], p[2]-pivot[2]]
                        if sign == 1:
                            nv = [v[2], v[1], -v[0]]
                        else:
                            nv = [-v[2], v[1], v[0]]
                        new_p = [pivot[0]+nv[0], pivot[1]+nv[1], pivot[2]+nv[2]]
                        new_segment.append(tuple(new_p))
                else: # z
                    # x, y -> -y, x (sign 1) or y, -x (sign -1)
                    new_segment = []
                    for p in segment:
                        v = [p[0]-pivot[0], p[1]-pivot[1], p[2]-pivot[2]]
                        if sign == 1:
                            nv = [-v[1], v[0], v[2]]
                        else:
                            nv = [v[1], -v[0], v[2]]
                        new_p = [pivot[0]+nv[0], pivot[1]+nv[1], pivot[2]+nv[2]]
                        new_segment.append(tuple(new_p))
                
                # Check self-avoidance
                # The new chain is chain[:i+1] + new_segment
                # We need to ensure new_segment doesn't intersect chain[:i+1] (except pivot)
                # and is self-avoiding.
                
                # Build set of fixed part
                fixed_set = set(chain[:i+1])
                # Remove pivot? No, pivot is in fixed part. The segment starts at i+1.
                # The first element of new_segment is chain[i+1] rotated.
                # It must not be in fixed_set.
                
                valid = True
                seen_in_seg = set()
                for p in new_segment:
                    if p in fixed_set:
                        valid = False
                        break
                    if p in seen_in_seg:
                        valid = False
                        break
                    seen_in_seg.add(p)
                
                if valid:
                    new_chain = chain[:i+1] + new_segment
                    # Calculate energy
                    # Optimization: Only recalculate energy if valid.
                    e = _compute_energy(sequence, new_chain)
                    rotations.append((e, new_chain))
        
        if not rotations:
            continue
        
        # Choose the best move or Metropolis accept
        # For simplicity and speed, let's take the best move (Greedy Local Search)
        # or Metropolis. Metropolis allows escaping local minima.
        # Given the short time budget per sequence, Greedy is safer?
        # Let's use Metropolis with T=1.0
        
        # Pick a random candidate from rotations? Or the best?
        # Let's pick the best one to ensure progress, but with a chance to accept worse?
        # Actually, standard MC: pick a random valid move from the 6 possibilities.
        # If none valid, skip.
        
        # Let's pick the one with lowest energy among valid ones?
        # No, that's steepest descent.
        # Let's pick a random valid move.
        
        e_new, chain_new = random.choice(rotations)
        
        # Metropolis criterion
        delta_e = e_new - current_energy
        if delta_e <= 0 or random.random() < math.exp(-delta_e / T):
            current_energy = e_new
            chain = chain_new
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        
        # Cool down slightly?
        # T = max(0.1, T * 0.999)
        
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Best conformation using MCCG + Local Search.
    """
    start_time = time.time()
    remaining_time = TIME_LIMIT - (time.time() - START_TIME)
    
    if remaining_time < 1.0:
        # Emergency fallback
        return [list(p) for p in [tuple([i] + [0]*(dimension-1)) for i in range(len(sequence))]]
    
    n = len(sequence)
    
    # Allocate time:
    # 60% for multiple MCCG runs, 40% for Local Search on the best one.
    # Or, run MCCG once, then Local Search.
    # Given the stochastic nature, multiple MCCG runs with different temps/seeds are good.
    
    # Let's try a few MCCG runs.
    # Time per MCCG run is O(N^2) roughly due to energy check? No, O(N) per step, O(N^2) total.
    # For N=100, it's fast.
    
    num_growth_runs = 5
    time_per_growth = remaining_time * 0.4 / num_growth_runs
    time_for_local = remaining_time * 0.6
    
    best_chain = None
    best_energy = float('inf')
    
    # Run MCCG multiple times
    for run_idx in range(num_growth_runs):
        # Vary temperature
        T = 1.0 + run_idx * 0.5
        
        # Set random seed for reproducibility within run? No, let it be random.
        # But we should ensure we don't waste time.
        
        # Time check inside growth?
        # Growth is fast.
        
        chain = _mccg_growth(sequence, dimension, temperature=T)
        
        if chain is None or len(chain) != n:
            continue
        
        e = _compute_energy(sequence, chain)
        
        if e < best_energy:
            best_energy = e
            best_chain = chain
    
    if best_chain is None:
        # Fallback to straight line
        best_chain = [tuple([i] + [0]*(dimension-1)) for i in range(n)]
        best_energy = _compute_energy(sequence, best_chain)
    
    # Local Search on the best chain
    if time.time() - START_TIME < TIME_LIMIT - 1.0:
        best_chain, best_energy = _local_search(sequence, best_chain, dimension, time_budget=time_for_local)
    
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
