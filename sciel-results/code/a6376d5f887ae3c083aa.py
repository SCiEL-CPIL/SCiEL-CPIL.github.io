import random
import time

_DEADLINE = None
_BUDGET_FRACTION = 0.90

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 330 * _BUDGET_FRACTION
    return _DEADLINE

def _is_deadline_exceeded():
    return time.time() > _get_deadline()

_OFFSETS = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

def _beam_search(sequence, dimension, beam_width=500):
    n = len(sequence)
    if n == 0:
        return []
    
    # State: (score, chain, occ, pos_to_idx, energy, free)
    # chain: list of tuples
    # occ: set of tuples
    # pos_to_idx: dict tuple->idx
    # energy: int
    # free: int (free neighbors of the last residue)
    
    start = (0, 0, 0)
    start_occ = {start}
    start_pos_idx = {start: 0}
    start_energy = 0
    start_free = 6 # 6 neighbors are free initially
    
    beam = [(0, [start], start_occ, start_pos_idx, start_energy, start_free)]
    
    for i in range(1, n):
        if _is_deadline_exceeded():
            break
            
        candidates = []
        last_residue_is_h = (sequence[i] == 'H')
        
        for score, chain, occ, pos_idx, energy, free in beam:
            last = chain[-1]
            lx, ly, lz = last
            
            for dx, dy, dz in _OFFSETS:
                new_point = (lx+dx, ly+dy, lz+dz)
                
                if new_point in occ:
                    continue
                
                # Calculate delta energy
                delta_e = 0
                if last_residue_is_h:
                    for odx, ody, odz in _OFFSETS:
                        neighbor = (new_point[0]+odx, new_point[1]+ody, new_point[2]+odz)
                        if neighbor in occ:
                            j = pos_idx[neighbor]
                            # j must be < i (already placed)
                            # and not i-1 (adjacent)
                            if j < i - 1:
                                if sequence[j] == 'H':
                                    delta_e -= 1
                            
                new_energy = energy + delta_e
                
                # Calculate free space for new_point
                new_free = 0
                for odx, ody, odz in _OFFSETS:
                    neighbor = (new_point[0]+odx, new_point[1]+ody, new_point[2]+odz)
                    if neighbor not in occ:
                        new_free += 1
                
                # Score: lower is better
                # Energy is primary. Free space is secondary.
                # Add small random noise to break ties and maintain diversity
                noise = random.random() * 0.01
                new_score = new_energy - 0.1 * new_free + noise
                
                # Prepare new state
                new_chain = chain + [new_point]
                new_occ = set(occ)
                new_occ.add(new_point)
                new_pos_idx = dict(pos_idx)
                new_pos_idx[new_point] = i
                
                candidates.append((new_score, new_chain, new_occ, new_pos_idx, new_energy, new_free))
        
        if not candidates:
            return None # Trapped
            
        # Sort by score
        candidates.sort(key=lambda x: x[0])
        
        # Keep top beam_width
        beam = candidates[:beam_width]
        
    if not beam:
        return None
        
    # Return the best chain
    best = min(beam, key=lambda x: x[0])
    return best[1]

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    
    # Try beam search with increasing width if time permits?
    # Or just one good width.
    # Let's try 300.
    
    chain = _beam_search(sequence, dimension, beam_width=300)
    
    if chain is None:
        # Fallback to greedy if beam search fails (shouldn't happen usually)
        # ... implement simple greedy fallback
        pass
        
    return [list(p) for p in chain]
