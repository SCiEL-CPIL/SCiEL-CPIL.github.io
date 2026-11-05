"""HP lattice protein folding on the cubic lattice - Beam Search Init + Pivot SA."""
import time
import random
import math

# EVOLVE-BLOCK-START
import time as _time

_NEIGHBOR_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
]

def _energy(sequence, chain):
    """Calculate energy of a chain. O(N) with set lookup."""
    n = len(chain)
    if n == 0:
        return 0
    pos_to_idx = {chain[i]: i for i in range(n)}
    energy = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        pi = chain[i]
        for d in _NEIGHBOR_OFFSETS:
            neighbor = (pi[0] + d[0], pi[1] + d[1], pi[2] + d[2])
            j = pos_to_idx.get(neighbor)
            if j is not None and j > i + 1:
                energy -= 1
    return energy

def _beam_search_init(sequence, beam_width=200, time_limit=10.0):
    """
    Initialize a chain using Beam Search.
    Returns a list of points (full or partial).
    """
    n = len(sequence)
    if n == 0:
        return []
    
    start_time = _time.time()
    
    # State: (energy, free_space, chain, occupied_set)
    # We prioritize low energy, then high free space (to avoid trapping)
    
    start = (0, 0, 0)
    start_occ = {start}
    # Initial free space: 6 neighbors are free
    start_free = 6
    start_energy = 0
    
    # Beam list of tuples: (score, chain, occ)
    # Score = energy - alpha * free_space. Lower is better.
    # We use a small penalty for low free space to encourage spreading.
    beam = [(0, [start], start_occ)]
    
    for i in range(1, n):
        if _time.time() - start_time > time_limit:
            break
            
        candidates = []
        current_residue = sequence[i]
        
        for score, chain, occ in beam:
            last = chain[-1]
            lx, ly, lz = last
            
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                new_point = (lx+dx, ly+dy, lz+dz)
                
                if new_point in occ:
                    continue
                
                # Calculate delta energy
                delta_e = 0
                if current_residue == 'H':
                    for odx, ody, odz in _NEIGHBOR_OFFSETS:
                        neighbor = (new_point[0]+odx, new_point[1]+ody, new_point[2]+odz)
                        if neighbor in occ:
                            # Check if neighbor is H and not adjacent in chain
                            # We don't have pos_to_idx here easily without building it, 
                            # but we can check if the point is in the chain and its index is < i-1
                            # For efficiency in beam search, we approximate or just check H count in occ?
                            # Actually, we need to know if the neighbor is H.
                            # We can store the sequence in the state? No, that's heavy.
                            # Let's just check if neighbor is in occ. If it is, is it H?
                            # We can't know if it's H without the index.
                            # However, we can precompute a set of H positions? No, they change.
                            # Alternative: Just check if neighbor is in occ. 
                            # If we assume random H/P, this is an approximation.
                            # Better: Store pos_to_idx in the beam state? That doubles memory.
                            # Let's stick to a simpler heuristic: 
                            # If neighbor is in occ, it's a contact. 
                            # We need to know if both are H.
                            # To do this properly, we need to know the index of the neighbor.
                            # Let's build pos_to_idx for the beam states? 
                            # Beam width 200, N up to 136. 200 * 136 dict entries. Manageable.
                            pass # Placeholder for proper energy calc
                
                # For the sake of speed in Beam Search, let's use a proxy score:
                # Energy is hard to compute incrementally without pos_to_idx.
                # Let's just use Free Space as the primary heuristic for Beam Search,
                # and rely on the subsequent SA to fix the energy.
                # Or, we can compute energy exactly by maintaining pos_to_idx.
                
                # Let's maintain pos_to_idx.
                # State: (score, chain, occ, pos_to_idx)
                # This is heavier but more accurate.
                
                new_pos_idx = dict(occ.keys()) # This is O(N) copy. Too slow for beam search.
                
        # Optimization: We will use a simpler heuristic for Beam Search:
        # Maximize Free Space. This avoids trapping. 
        # The energy will be optimized in the SA phase.
        # This is a structural change: Beam Search for topology (non-trapping), SA for energy.
        
        # Re-implementing Beam Search with Free Space heuristic for speed.
        new_beam = []
        for score, chain, occ in beam:
            last = chain[-1]
            lx, ly, lz = last
            for dx, dy, dz in _NEIGHBOR_OFFSETS:
                new_point = (lx+dx, ly+dy, lz+dz)
                if new_point in occ:
                    continue
                new_occ = occ | {new_point}
                new_chain = chain + [new_point]
                
                # Calculate free space of new_point
                free = 0
                for odx, ody, odz in _NEIGHBOR_OFFSETS:
                    nb = (new_point[0]+odx, new_point[1]+ody, new_point[2]+odz)
                    if nb not in new_occ:
                        free += 1
                            
                # Score: Lower is better. 
                # We want to maximize free space, so minimize -free.
                # Add small noise to break ties.
                noise = random.random() * 0.01
                new_score = -free + noise
                new_beam.append((new_score, new_chain, new_occ))
                
        if not new_beam:
            # Trapped. Return what we have.
            return beam[0][1] if beam else [start]
            
        new_beam.sort(key=lambda x: x[0])
        beam = new_beam[:beam_width]
        
    # If we finished the loop, return the best chain
    if beam:
        return beam[0][1]
    return [start]

def _complete_chain(partial_chain, sequence, rng):
    """Complete a partial chain with a random walk."""
    if not partial_chain:
        return [(0,0,0)]
    
    n_target = len(sequence)
    current_len = len(partial_chain)
    if current_len >= n_target:
        return partial_chain[:n_target]
        
    chain = partial_chain[:]
    occ = set(chain)
    
    for i in range(current_len, n_target):
        last = chain[-1]
        moves = _NEIGHBOR_OFFSETS[:]
        rng.shuffle(moves)
        placed = False
        for d in moves:
            new_pt = (last[0] + d[0], last[1] + d[1], last[2] + d[2])
            if new_pt not in occ:
                chain.append(new_pt)
                occ.add(new_pt)
                placed = True
                break
        if not placed:
            # Trapped. Try backtracking? For simplicity, just fail and return what we have?
            # No, must return full length. 
            # If trapped, we have to restart or use a different init.
            # In this function, if trapped, we return a linear chain as fallback.
            return [(i, 0, 0) for i in range(n_target)]
            
    return chain

def _pivot_sa_fold(sequence, initial_chain, time_limit, rng):
    """Simulated Annealing with Pivot moves."""
    n = len(sequence)
    if n <= 2:
        return initial_chain
    
    chain = initial_chain[:]
    current_energy = _energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    start_time = _time.time()
    
    # Temperature schedule
    T_start = max(1.0, abs(current_energy) * 0.5)
    T_end = 0.05
    # Estimate steps based on time. ~1ms per step? Let's aim for 5000 steps per second.
    # But we check time every 100 steps.
    
    step_count = 0
    while _time.time() - start_time < time_limit:
        T = T_start * (T_end / T_start) ** (step_count / 5000.0)
        if T < T_end:
            T = T_end
            
        k = rng.randint(1, n - 2)
        rot_type = rng.randint(0, 5)
        
        pivot_point = chain[k]
        old_tail = chain[k+1:]
        new_tail = []
        valid = True
        head_set = set(chain[:k+1])
        
        # Apply rotation
        for pt in old_tail:
            rx, ry, rz = pt[0] - pivot_point[0], pt[1] - pivot_point[1], pt[2] - pivot_point[2]
            if rot_type == 0:
                nx, ny, nz = rx, -rz, ry
            elif rot_type == 1:
                nx, ny, nz = rx, rz, -ry
            elif rot_type == 2:
                nx, ny, nz = rz, ry, -rx
            elif rot_type == 3:
                nx, ny, nz = -rz, ry, rx
            elif rot_type == 4:
                nx, ny, nz = -ry, rx, rz
            else:
                nx, ny, nz = ry, -rx, rz
                
            new_pt = (pivot_point[0] + nx, pivot_point[1] + ny, pivot_point[2] + nz)
            if new_pt in head_set:
                valid = False
                break
            new_tail.append(new_pt)
        
        if valid:
            tail_set = set()
            for pt in new_tail:
                if pt in tail_set:
                    valid = False
                    break
                tail_set.add(pt)
        
        if not valid:
            step_count += 1
            continue
            
        new_chain = chain[:k+1] + new_tail
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta <= 0 or (T > 0 and rng.random() < math.exp(-delta / T)):
            chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
                
        step_count += 1
        if step_count % 100 == 0 and _time.time() - start_time > time_limit:
            break
            
    return best_chain

def fold(sequence, dimension):
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    # Global time management
    if not hasattr(fold, '_start_time'):
        fold._start_time = _time.time()
        
    elapsed = _time.time() - fold._start_time
    # Reserve 10s for safety. Total budget 300s.
    remaining = max(1.0, 300.0 - elapsed - 10.0)
    
    # Allocate 30% to Beam Search, 70% to SA
    beam_time = remaining * 0.3
    sa_time = remaining * 0.7
    
    rng = random.Random(42 + n * 7)
    
    # 1. Beam Search Initialization
    initial_chain = _beam_search_init(sequence, beam_width=100, time_limit=beam_time)
    
    # Ensure it's full length
    if len(initial_chain) < n:
        initial_chain = _complete_chain(initial_chain, sequence, rng)
        
    # Fallback if completion failed (returned linear chain)
    # Check if it's the fallback linear chain
    if all(initial_chain[i] == (i, 0, 0) for i in range(min(5, n))):
        # Try a random walk init as backup if beam search failed badly
        pass # The linear chain is valid, just bad. SA will try to fix it.
        
    # 2. Pivot SA Refinement
    best_chain = _pivot_sa_fold(sequence, initial_chain, sa_time, rng)
    
    # Validation
    if len(set(best_chain)) != n:
        best_chain = [(i, 0, 0) for i in range(n)]
        
    for i in range(1, n):
        d = abs(best_chain[i][0] - best_chain[i-1][0]) + \
            abs(best_chain[i][1] - best_chain[i-1][1]) + \
            abs(best_chain[i][2] - best_chain[i-1][2])
        if d != 1:
            best_chain = [(i, 0, 0) for i in range(n)]
            break
            
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
