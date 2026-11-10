"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
import sys

# Global deadline guard: must finish well before 330s wall-clock.
_TOTAL_START = time.time()
_SAFETY_MARGIN = 15.0  # seconds to leave before hard kill

def _remaining_time():
    """Return remaining seconds before our safety deadline."""
    return _TOTAL_START + 300.0 - _SAFETY_MARGIN - time.time()

def _neighbours(point):
    """Yield the 6 neighbors of a point in 3D cubic lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Calculate the energy of a chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _build_h_positions(sequence):
    """Precompute set of H residue indices for fast lookup."""
    return set(i for i, c in enumerate(sequence) if c == "H")

def _contact_delta(sequence, h_pos, i, old_site, new_site, occupied):
    """
    Compute energy delta for moving residue i from old_site to new_site.
    Returns: negative delta means energy improves (more contacts).
    """
    if i not in h_pos:
        return 0
    
    delta = 0
    # Lose contacts at old_site
    for q in _neighbours(old_site):
        j = occupied.get(q)
        if j is not None and j != i and j in h_pos and abs(i - j) > 1:
            delta += 1
    # Gain contacts at new_site
    for q in _neighbours(new_site):
        j = occupied.get(q)
        if j is not None and j != i and j in h_pos and abs(i - j) > 1:
            delta -= 1
    return delta

def _beam_search_fold(sequence, dimension, beam_width, rng, time_limit):
    """
    Beam search for HP lattice folding.
    Returns the best chain found or None.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0,) * dimension]
        
    start_point = (0,) * dimension
    start_chain = [start_point]
    start_occupied = {start_point: 0}
    start_energy = 0
    start_score = 0
    
    beam = [(start_score, start_energy, start_chain, start_occupied, start_point)]
    
    h_pos = _build_h_positions(sequence)
    
    for i in range(1, n):
        res = sequence[i]
        new_beam_candidates = []
        
        if time.time() > time_limit:
            return None
            
        for score, energy_so_far, chain, occupied_dict, last_point in beam:
            for nb in _neighbours(last_point):
                if nb in occupied_dict:
                    continue
                
                delta_e = 0
                if res == "H":
                    for q in _neighbours(nb):
                        if q in occupied_dict:
                            j = occupied_dict[q]
                            if j != i - 1:
                                if sequence[j] == "H":
                                    delta_e += 1
                new_energy = energy_so_far - delta_e
                
                free_count = 0
                for q in _neighbours(nb):
                    if q not in occupied_dict:
                        free_count += 1
                
                # Heuristic: Prefer compact structures (fewer free neighbors)
                # and lower energy.
                new_score = new_energy + 0.15 * free_count
                
                new_chain = chain + [nb]
                new_occupied = occupied_dict.copy()
                new_occupied[nb] = i
                
                new_beam_candidates.append((new_score, new_energy, new_chain, new_occupied, nb))
        
        if not new_beam_candidates:
            return None
            
        new_beam_candidates.sort(key=lambda x: x[0])
        beam = new_beam_candidates[:beam_width]
        
        if not beam:
            return None
            
    best_state = min(beam, key=lambda x: x[1])
    return best_state[2]

def _local_search(sequence, chain, max_iters, rng, time_limit):
    """
    Local search with Simulated Annealing.
    Includes Single Moves and Pivot Moves.
    """
    n = len(sequence)
    if n < 3:
        return chain
    
    h_pos = _build_h_positions(sequence)
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    occupied = {p: i for i, p in enumerate(chain)}
    
    T_start = 1.5
    T_end = 0.05
    cooling_rate = (T_end / T_start) ** (1.0 / max(1, max_iters))
    
    iters = 0
    no_improve_count = 0
    patience = 1000
    
    while iters < max_iters:
        if time.time() > time_limit:
            break
        iters += 1
        
        T = T_start * (cooling_rate ** iters)
        
        move_type = rng.random()
        improved = False
        
        if move_type < 0.7:
            # Single Move
            i = rng.randrange(1, n - 1)
            old_site = chain[i]
            
            for new_site in _neighbours(old_site):
                if new_site in occupied:
                    continue
                
                # Check adjacency
                if i > 0:
                    prev = chain[i-1]
                    if abs(new_site[0]-prev[0]) + abs(new_site[1]-prev[1]) + abs(new_site[2]-prev[2]) != 1:
                        continue
                if i < n - 1:
                    nxt = chain[i+1]
                    if abs(new_site[0]-nxt[0]) + abs(new_site[1]-nxt[1]) + abs(new_site[2]-nxt[2]) != 1:
                        continue
                
                delta = _contact_delta(sequence, h_pos, i, old_site, new_site, occupied)
                new_energy = current_energy + delta
                
                accept = False
                if new_energy < current_energy:
                    accept = True
                else:
                    prob = math.exp(-delta / T)
                    if rng.random() < prob:
                        accept = True
                
                if accept:
                    occupied.pop(old_site)
                    occupied[new_site] = i
                    chain[i] = new_site
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_chain = list(chain)
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    improved = True
                    break
        else:
            # Pivot Move
            # Pick a segment [i, j] to rotate around i-1
            if n > 4:
                i = rng.randrange(1, n - 1)
                max_len = min(10, n - i - 1)
                if max_len > 1:
                    seg_len = rng.randint(2, max_len)
                    j = i + seg_len - 1
                    
                    # Store original segment
                    orig_seg = chain[i:j+1]
                    pivot = chain[i-1]
                    
                    # Try rotating the first vector of the segment
                    # The vector from pivot to chain[i]
                    v = (chain[i][0]-pivot[0], chain[i][1]-pivot[1], chain[i][2]-pivot[2])
                    
                    # Find valid new vectors
                    valid_dirs = []
                    for d in _neighbours(pivot):
                        nd = (d[0]-pivot[0], d[1]-pivot[1], d[2]-pivot[2])
                        if nd != v and nd != (-v[0], -v[1], -v[2]):
                            valid_dirs.append(nd)
                    
                    if valid_dirs:
                        new_v = rng.choice(valid_dirs)
                        diff = (new_v[0]-v[0], new_v[1]-v[1], new_v[2]-v[2])
                        
                        # Calculate new positions for the segment
                        new_seg = []
                        valid = True
                        # We need to check if new positions collide with static parts
                        # Static parts: [0, i-1] and [j+1, n-1]
                        static_occupied = set()
                        for k in range(0, i):
                            static_occupied.add(chain[k])
                        for k in range(j+1, n):
                            static_occupied.add(chain[k])
                        
                        # Note: chain[i] moves to pivot + new_v
                        # chain[k] moves to chain[k] + diff
                        for k in range(i, j+1):
                            np = (chain[k][0]+diff[0], chain[k][1]+diff[1], chain[k][2]+diff[2])
                            # Check collision with static parts
                            if np in static_occupied:
                                valid = False
                                break
                            # Check collision with other moving parts? 
                            # Since it's a rigid rotation, internal collisions are preserved 
                            # if the original was valid. But we must ensure 
                            # the new segment doesn't intersect itself? 
                            # Rigid motion preserves internal distances, so if it was 
                            # self-avoiding before, it is self-avoiding after.
                            # However, it might collide with the *tail* of the moving segment 
                            # if the rotation causes overlap? No, rigid body motion.
                            new_seg.append(np)
                        
                        if valid:
                            # Calculate energy delta
                            # Remove old segment from occupied
                            for k in range(i, j+1):
                                del occupied[chain[k]]
                            
                            # Add new segment to occupied (tentatively)
                            for k in range(i, j+1):
                                occupied[new_seg[k-i]] = k
                            
                            # Calculate new energy
                            new_energy = _energy(sequence, chain[:i] + new_seg + chain[j+1:])
                            delta = new_energy - current_energy
                            
                            accept = False
                            if new_energy < current_energy:
                                accept = True
                            else:
                                prob = math.exp(-delta / T)
                                if rng.random() < prob:
                                    accept = True
                            
                            if accept:
                                # Apply move
                                for k in range(i, j+1):
                                    chain[k] = new_seg[k-i]
                                current_energy = new_energy
                                if new_energy < best_energy:
                                    best_energy = new_energy
                                    best_chain = list(chain)
                                    no_improve_count = 0
                                else:
                                    no_improve_count += 1
                                improved = True
                            else:
                                # Revert occupied set
                                for k in range(i, j+1):
                                    del occupied[new_seg[k-i]]
                                    occupied[orig_seg[k-i]] = k
                            
                    if not improved:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= patience:
                break
                
    return best_chain

def fold(sequence, dimension):
    """
    Fold the sequence using Beam Search for construction and SA for refinement.
    """
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = float('inf')
    
    n = len(sequence)
    
    # Determine beam width based on chain length
    if n <= 48:
        beam_width = 300
        num_runs = 6
    elif n <= 80:
        beam_width = 200
        num_runs = 4
    else:
        beam_width = 100
        num_runs = 3
        
    total_time_available = _remaining_time()
    time_per_run = total_time_available / num_runs
    
    for run in range(num_runs):
        if _remaining_time() < 5.0:
            break
            
        # Time allocation per run: 30% Beam, 70% SA
        bs_time = time_per_run * 0.3
        sa_time = time_per_run * 0.7
        
        bs_deadline = time.time() + bs_time
        sa_deadline = time.time() + time_per_run - 1.0
        
        # Run Beam Search
        chain = _beam_search_fold(sequence, dimension, beam_width, rng, bs_deadline)
        
        if chain is None:
            # Fallback: Greedy
            start = (0,) * dimension
            chain, occupied = [start], {start: 0}
            for i in range(1, len(sequence)):
                options = []
                for site in _neighbours(chain[-1]):
                    if site not in occupied:
                        options.append(site)
                if not options:
                    break
                site = rng.choice(options)
                occupied[site] = i
                chain.append(site)
            if len(chain) < len(sequence):
                chain = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
        
        # Run SA
        max_ls_iters = max(2000, min(50000, 200 * len(sequence)))
        improved_chain = _local_search(
            sequence, 
            chain, 
            max_ls_iters, 
            rng, 
            sa_deadline
        )
        
        energy = _energy(sequence, improved_chain)
        if energy < best_energy:
            best_energy = energy
            best_chain = improved_chain
            
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
