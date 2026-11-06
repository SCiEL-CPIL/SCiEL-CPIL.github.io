import random
import time

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _grow(sequence, dimension, rng):
    """Fast greedy growth to initialize."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
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

def fold(sequence, dimension):
    """
    Simulated Annealing with Pivots.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0]*dimension]
    
    rng = random.Random(12345 + len(sequence))
    
    # Global time budget per call (approx 11s to be safe for 26 calls in 300s total, 
    # though "subprocess" might mean isolated. Let's assume we have ~10-12s).
    # If it's a single process for all, we need to be smarter. 
    # The prompt says "fold is called once per benchmark sequence in a sandboxed subprocess".
    # This strongly implies isolation. So I will use a local time limit of 11 seconds.
    
    start_time = time.time()
    max_time = 11.0
    
    best_chain = None
    best_energy = 0  # 0 is unfolded (no contacts), energy is negative
    
    # Try multiple restarts if time permits
    max_restarts = 1000
    
    for restart in range(max_restarts):
        if time.time() - start_time > max_time * 0.9:
            break
            
        # 1. Initialize
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            # Fallback: simple random walk or straight line
            # Straight line is always valid but energy 0
            chain = [tuple([i] + [0]*(dimension-1)) for i in range(n)]
        
        occupied = {p: i for i, p in enumerate(chain)}
        current_energy = _energy(sequence, chain)
        
        if best_chain is None or current_energy < best_energy:
            best_chain = chain.copy()
            best_energy = current_energy
            
        # 2. Simulated Annealing
        # T starts high, cools down
        # Max steps per restart depends on time left
        # Let's run for a fixed number of steps or until time slice ends
        # ~10000 steps per restart? 
        # If we have 11s, and 1000 steps/sec, that's 11k steps total.
        # So maybe 1 restart with many steps is better? 
        # Or 10 restarts with 1000 steps?
        # SA benefits from long runs. Let's do 1 long run per restart if possible.
        
        # Estimate steps possible: 5000 steps/sec is optimistic for Python with O(N) moves.
        # Let's say 2000 steps/sec.
        # 11s * 2000 = 22000 steps.
        
        t_start = time.time()
        # Temperature schedule
        # E_range is roughly -N/2 to 0. 
        # Let's set T_max = 5.0, T_min = 0.01
        T_max = 5.0
        T_min = 0.01
        # Number of steps for this restart
        # If first restart, use most of the time.
        if restart == 0:
            steps = 20000
        else:
            steps = 2000
            
        # Precompute H positions for faster energy delta?
        # We need to know if a site is H.
        is_h = [1 if s == 'H' else 0 for s in sequence]
        
        for step in range(steps):
            if time.time() - t_start > (max_time * 0.9) / max(1, restart+1) + 1.0:
                break
                
            # Choose move: Pivot
            # Pivot index k: 1 to n-1
            # We rotate segment k..n-1 around chain[k-1]
            k = rng.randint(1, n-1)
            pivot_site = chain[k-1]
            
            # Extract segment
            segment = chain[k:]
            seg_len = len(segment)
            
            # Remove segment from occupied
            # We need to restore it later if rejected
            old_positions = segment
            for p in old_positions:
                del occupied[p]
            
            # Choose rotation
            # 6 possible rotations around pivot_site
            # Axis: 0, 1, 2. Step: +1, -1 (90 deg)
            axis = rng.randint(0, dimension-1)
            direction = rng.choice([-1, 1])
            
            # Rotation logic:
            # Vector v = p - pivot_site
            # Rotate v by 90 deg around axis 'axis'
            # If axis=0: (x,y,z) -> (x, -y*z_dir, y*z_dir)? No.
            # Standard 90 deg rotation around Z (axis 2): (x,y,z) -> (-y, x, z) or (y, -x, z)
            # Around Y (axis 1): (x,y,z) -> (z, y, -x) or (-z, y, x)
            # Around X (axis 0): (x,y,z) -> (x, -z, y) or (x, z, -y)
            
            new_segment = []
            valid = True
            for p in old_positions:
                v = [p[i] - pivot_site[i] for i in range(dimension)]
                # Rotate v
                if axis == 0:
                    if direction == 1:
                        # x, -z, y
                        v_new = [v[0], -v[2], v[1]]
                    else:
                        # x, z, -y
                        v_new = [v[0], v[2], -v[1]]
                elif axis == 1:
                    if direction == 1:
                        # z, y, -x
                        v_new = [v[2], v[1], -v[0]]
                    else:
                        # -z, y, x
                        v_new = [-v[2], v[1], v[0]]
                else: # axis 2
                    if direction == 1:
                        # -y, x, z
                        v_new = [-v[1], v[0], v[2]]
                    else:
                        # y, -x, z
                        v_new = [v[1], -v[0], v[2]]
                
                new_p = tuple(pivot_site[i] + v_new[i] for i in range(dimension))
                new_segment.append(new_p)
                
                # Check collision with occupied (which is R)
                if new_p in occupied:
                    valid = False
                    break
            
            if not valid:
                # Restore
                for p, idx in zip(old_positions, range(k, n)):
                    occupied[p] = idx
                continue
            
            # Calculate Energy Delta
            # Old energy contribution of segment with R
            # New energy contribution of segment with R
            # Internal energy is invariant.
            
            # Helper to get energy between two sets of indices/positions
            # We need to check contacts between new_segment and R (indices 0..k-1)
            
            def calc_boundary_energy(seg_positions, seg_start_idx, other_indices):
                # other_indices is a set of indices in R
                # We check neighbors of each pos in seg_positions
                # If neighbor is in occupied (which is R), check if both H
                e = 0
                # Map position to index for R is in 'occupied'
                # But we removed seg from occupied. So occupied is exactly R.
                for i_offset, pos in enumerate(seg_positions):
                    idx = seg_start_idx + i_offset
                    if is_h[idx] == 0:
                        continue
                    for nb in _neighbours(pos):
                        if nb in occupied:
                            nb_idx = occupied[nb]
                            if is_h[nb_idx] == 1:
                                # Avoid double counting? 
                                # We are counting pairs (seg, R). 
                                # Since seg and R are disjoint, each pair is counted once 
                                # if we only look from seg side.
                                e += 1
                return e
            
            # Note: 'occupied' currently contains R.
            old_e_boundary = calc_boundary_energy(old_positions, k, set(range(k))) # dummy arg
            new_e_boundary = calc_boundary_energy(new_segment, k, set(range(k)))
            
            delta_e = new_e_boundary - old_e_boundary
            
            # Note: Energy is negative contacts. 
            # My _energy returns -contacts.
            # So if contacts increase, energy decreases (becomes more negative).
            # calc_boundary_energy returns count of contacts (positive).
            # So delta_energy = -delta_contacts.
            delta_energy = -delta_e
            
            # SA Acceptance
            if delta_energy < 0:
                accept = True
            else:
                # T is needed. 
                # We need a temperature schedule.
                # Let's use linear cooling.
                # T = T_max * (1 - step/steps)
                T = T_max * (1.0 - step / steps)
                if T < T_min: T = T_min
                prob = 1.0 / (1.0 + delta_energy / T) # Logistic? Or Exp?
                # Standard Metropolis: exp(-delta_E / T)
                # delta_E is positive (energy increase).
                # If delta_E is large, exp is small.
                import math
                try:
                    prob = math.exp(-delta_energy / T)
                except OverflowError:
                    prob = 0.0
                
                accept = rng.random() < prob
            
            if accept:
                # Update chain and occupied
                # Remove old, add new
                # We already removed old.
                for p, idx in zip(new_segment, range(k, n)):
                    occupied[p] = idx
                    chain[idx] = p
                
                current_energy += delta_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain.copy()
            else:
                # Restore old
                for p, idx in zip(old_positions, range(k, n)):
                    occupied[p] = idx
                # chain is already correct for old positions? 
                # We didn't change chain for old. 
                # But we changed chain for new in the accept block. 
                # In reject, chain is still old.
                # Wait, I modified chain only in accept. 
                # So in reject, chain is intact.
                pass

    if best_chain is None:
        # Fallback
        best_chain = [tuple([i] + [0]*(dimension-1)) for i in range(n)]
        
    return [list(p) for p in best_chain]
