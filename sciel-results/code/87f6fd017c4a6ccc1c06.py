import random
import time
import math

# Precompute 90-degree rotation matrices for 3D
_ROT_MATRICES = [
    [[1,0,0],[0,0,-1],[0,1,0]],
    [[1,0,0],[0,0,1],[0,-1,0]],
    [[0,0,1],[0,1,0],[-1,0,0]],
    [[0,0,-1],[0,1,0],[1,0,0]],
    [[0,-1,0],[1,0,0],[0,0,1]],
    [[0,1,0],[-1,0,0],[0,0,1]],
]

_NB_DIRS = [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]

def _neighbours(point):
    for dx, dy, dz in _NB_DIRS:
        yield (point[0]+dx, point[1]+dy, point[2]+dz)

def _neighbours_list(point):
    return [(point[0]+dx, point[1]+dy, point[2]+dz) for dx, dy, dz in _NB_DIRS]

def _beam_search_growth(sequence, dimension, rng, beam_width=50):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0,0,0)]
    
    start = (0, 0, 0)
    # Beam items: (score, energy, chain, occupied)
    beams = [(-0.3 * 6, 0, [(0,0,0)], {(0,0,0): 0})]
    is_h = [1 if s == 'H' else 0 for s in sequence]
    
    for i in range(1, n):
        candidates = []
        for chain, energy, occupied in [(b[2], b[1], b[3]) for b in beams]:
            last_pos = chain[-1]
            for site in _neighbours(last_pos):
                if site in occupied:
                    continue
                
                new_energy = energy
                if is_h[i]:
                    for q in _neighbours(site):
                        if q in occupied:
                            j = occupied[q]
                            if is_h[j]:
                                new_energy -= 1
                
                # Heuristic: Prefer lower energy, then more free space (expansion)
                # We don't need to compute free_count exactly if we just use a small penalty for "crowding"
                # But free_count is a good proxy for future flexibility.
                # To save time, we can approximate or just rely on energy + small random noise.
                # The original code calculated free_count. Let's keep it but optimize if possible.
                # Actually, free_count calculation is O(1) per neighbor, so 6 checks.
                free_count = 0
                for q in _neighbours(site):
                    if q not in occupied:
                        free_count += 1
                
                # Score: We want to minimize this value for sorting (lower is better in this context if we sort ascending)
                # Original: new_energy - 0.3 * free_count + noise
                # Let's increase the weight on energy to ensure we don't sacrifice too much energy for space.
                score = new_energy - 0.2 * free_count + rng.random() * 0.1
                
                new_chain = chain + [site]
                new_occupied = occupied.copy()
                new_occupied[site] = i
                candidates.append((score, new_energy, new_chain, new_occupied))
        
        if not candidates:
            break
            
        # Sort by score (ascending)
        candidates.sort(key=lambda x: x[0])
        # Keep top beam_width
        beams = [(c[0], c[1], c[2], c[3]) for c in candidates[:beam_width]]
        
    if not beams:
        return [(i, 0, 0) for i in range(n)]
        
    # Return the one with the lowest energy among the best scored ones?
    # Or just the one with the best score? The score includes energy.
    # Let's pick the one with the lowest energy from the top candidates.
    # The first one in the sorted list has the best score.
    best = min(beams, key=lambda x: x[1])
    return best[2]

def _compute_energy(sequence, chain):
    n = len(chain)
    if n < 2:
        return 0
    is_h = [1 if s == 'H' else 0 for s in sequence]
    occupied = {p: i for i, p in enumerate(chain)}
    e = 0
    for i, p in enumerate(chain):
        if is_h[i]:
            for q in _neighbours(p):
                j = occupied.get(q)
                if j is not None and j > i + 1 and is_h[j]:
                    e -= 1
    return e

def _local_search_steepest(sequence, chain, dimension, rng, time_limit):
    """
    Steepest Descent Local Search with Random Kicks.
    1. Try to find a move (pivot of length 1-3) that improves energy.
    2. If found, apply it.
    3. If stuck (no improvement in K tries), apply a random valid move (kick) to escape.
    """
    n = len(chain)
    if n < 3:
        return chain
    
    is_h = [1 if s == 'H' else 0 for s in sequence]
    current_chain = chain.copy()
    occupied = {p: i for i, p in enumerate(current_chain)}
    current_energy = _compute_energy(sequence, current_chain)
    best_chain = current_chain.copy()
    best_energy = current_energy
    start_time = time.time()
    
    # Parameters
    MAX_STUCK = 50  # Number of failed improvement attempts before a kick
    stuck_count = 0
    
    while time.time() - start_time < time_limit:
        improved = False
        kick_needed = False
        stuck_count += 1
        
        if stuck_count > MAX_STUCK:
            kick_needed = True
            stuck_count = 0
            
        # Try a random move
        # Choose pivot length: 1 (end), 2 (kink), 3 (crank/pivot)
        # Weighted towards shorter moves for speed? Or random?
        # Let's pick a random segment length 1 to min(4, n-1)
        max_seg_len = min(4, n - 1)
        seg_len = rng.randint(1, max_seg_len)
        k = rng.randint(1, n - seg_len)
        
        pivot_site = current_chain[k-1]
        
        # Choose a rotation
        axis = rng.randint(0, 2)
        direction = rng.choice([-1, 1])
        mat = _ROT_MATRICES[axis*2 + (0 if direction == 1 else 1)]
        
        # Remove segment from occupied
        old_segment = current_chain[k:k+seg_len]
        for p in old_segment:
            del occupied[p]
            
        # Apply rotation
        new_segment = []
        valid = True
        for i in range(seg_len):
            p = old_segment[i]
            v = [p[0]-pivot_site[0], p[1]-pivot_site[1], p[2]-pivot_site[2]]
            nv0 = mat[0][0]*v[0] + mat[0][1]*v[1] + mat[0][2]*v[2]
            nv1 = mat[1][0]*v[0] + mat[1][1]*v[1] + mat[1][2]*v[2]
            nv2 = mat[2][0]*v[0] + mat[2][1]*v[1] + mat[2][2]*v[2]
            np_ = (pivot_site[0]+nv0, pivot_site[1]+nv1, pivot_site[2]+nv2)
            if np_ in occupied:
                valid = False
                break
            new_segment.append(np_)
            
        if not valid:
            # Restore
            for i, p in enumerate(old_segment):
                occupied[p] = k + i
            continue
            
        # Calculate delta energy
        # Only need to check contacts involving the moved segment
        # Old contacts: segment vs rest
        old_contacts = 0
        for i in range(seg_len):
            idx = k + i
            if is_h[idx]:
                for nb in _neighbours(old_segment[i]):
                    j = occupied.get(nb)
                    if j is not None and (j < k or j >= k + seg_len) and is_h[j]:
                        old_contacts += 1
                        
        # New contacts: new segment vs rest
        new_contacts = 0
        for i in range(seg_len):
            idx = k + i
            if is_h[idx]:
                for nb in _neighbours(new_segment[i]):
                    j = occupied.get(nb)
                    if j is not None and (j < k or j >= k + seg_len) and is_h[j]:
                        new_contacts += 1
                        
        delta_energy = -(new_contacts - old_contacts)
        
        if not kick_needed:
            if delta_energy < 0:
                # Apply move
                for i, p in enumerate(new_segment):
                    occupied[p] = k + i
                    current_chain[k + i] = p
                current_energy += delta_energy
                improved = True
                stuck_count = 0
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = current_chain.copy()
            else:
                # Restore
                for i, p in enumerate(old_segment):
                    occupied[p] = k + i
        else:
            # Kick: Apply any valid move, even if worse
            for i, p in enumerate(new_segment):
                occupied[p] = k + i
                current_chain[k + i] = p
            current_energy += delta_energy
            # Do not update best here, we are escaping a local min
            improved = True # Mark as "action taken" to reset loop logic if needed, but we already reset stuck_count

    return best_chain

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0]*dimension]
    
    # Total budget is 330s for 26 sequences. ~12s per sequence.
    # We will use 11.5s per call to be safe.
    PER_CALL_TIME_LIMIT = 11.5
    start_time = time.time()
    deadline = start_time + PER_CALL_TIME_LIMIT
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = float('inf')
    
    # Beam Search Initialization
    # Smaller beam for longer chains to save time in initialization
    bw = 40 if n > 60 else 60
    initial_chain = _beam_search_growth(sequence, dimension, rng, beam_width=bw)
    if initial_chain is None or len(initial_chain) != n:
        initial_chain = [(i, 0, 0) for i in range(n)]
        
    init_energy = _compute_energy(sequence, initial_chain)
    best_energy = init_energy
    best_chain = initial_chain.copy()
    
    # Local Search Refinement
    remaining_time = deadline - time.time()
    if remaining_time > 1.0:
        # Allocate most of the remaining time to local search
        ls_time = remaining_time - 0.5
        refined_chain = _local_search_steepest(sequence, initial_chain, dimension, rng, ls_time)
        ref_energy = _compute_energy(sequence, refined_chain)
        if ref_energy < best_energy:
            best_energy = ref_energy
            best_chain = refined_chain.copy()
            
    # Second restart if time permits
    remaining_time = deadline - time.time()
    if remaining_time > 3.0:
        rng2 = random.Random(99999 + len(sequence))
        bw2 = 30 if n > 60 else 40
        initial_chain2 = _beam_search_growth(sequence, dimension, rng2, beam_width=bw2)
        if initial_chain2 and len(initial_chain2) == n:
            rem2 = deadline - time.time() - 0.2
            if rem2 > 0.5:
                refined_chain2 = _local_search_steepest(sequence, initial_chain2, dimension, rng2, rem2)
                ref_energy2 = _compute_energy(sequence, refined_chain2)
                if ref_energy2 < best_energy:
                    best_energy = ref_energy2
                    best_chain = refined_chain2.copy()
            else:
                e2 = _compute_energy(sequence, initial_chain2)
                if e2 < best_energy:
                    best_energy = e2
                    best_chain = initial_chain2.copy()
    
    # Third restart if still time
    remaining_time = deadline - time.time()
    if remaining_time > 3.0:
        rng3 = random.Random(55555 + len(sequence))
        bw3 = 25 if n > 60 else 30
        initial_chain3 = _beam_search_growth(sequence, dimension, rng3, beam_width=bw3)
        if initial_chain3 and len(initial_chain3) == n:
            rem3 = deadline - time.time() - 0.2
            if rem3 > 0.5:
                refined_chain3 = _local_search_steepest(sequence, initial_chain3, dimension, rng3, rem3)
                ref_energy3 = _compute_energy(sequence, refined_chain3)
                if ref_energy3 < best_energy:
                    best_energy = ref_energy3
                    best_chain = refined_chain3.copy()
            else:
                e3 = _compute_energy(sequence, initial_chain3)
                if e3 < best_energy:
                    best_energy = e3
                    best_chain = initial_chain3.copy()

    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
        
    # Final validation
    seen = set()
    for i, p in enumerate(best_chain):
        if p in seen:
            best_chain = [(i, 0, 0) for i in range(n)]
            break
        seen.add(p)
        if i > 0:
            prev = best_chain[i-1]
            d = abs(p[0]-prev[0]) + abs(p[1]-prev[1]) + abs(p[2]-prev[2])
            if d != 1:
                best_chain = [(i, 0, 0) for i in range(n)]
                break
    
    return [list(p) for p in best_chain]
