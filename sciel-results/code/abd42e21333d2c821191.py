import random
import math
import time

def _neighbours(point):
    # Same as before

def _count_contacts(sequence, chain, where):
    # Same as before

def _initial_chain(sequence, dimension, rng):
    # Same as before

def _monte_carlo_search(sequence, dimension, rng, time_limit=8.0):
    n = len(sequence)
    if n <= 2:
        return _initial_chain(sequence, dimension, rng)
    
    chain = _initial_chain(sequence, dimension, rng)
    where = {p: i for i, p in enumerate(chain)}
    
    # Initial Energy
    contacts = _count_contacts(sequence, chain, where)
    current_energy = -contacts
    best_chain = list(chain)
    best_energy = current_energy
    
    start_temp = 10.0
    end_temp = 0.1
    max_steps = 20000 # Adjust based on speed
    start_time = time.time()
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
        
        progress = step / max_steps
        temp = start_temp * math.exp(-5 * progress)
        if temp < end_temp:
            temp = end_temp
            
        # Move: Segment Re-attachment
        if n > 4:
            # Pick segment length
            # Smaller segments are faster to check
            max_seg_len = min(10, n // 2)
            seg_len = rng.randint(1, max_seg_len)
            i = rng.randint(1, n - seg_len - 1)
            # Segment is chain[i+1 : i+1+seg_len]
            # Indices: i+1 to i+seg_len
            
            # Try to move this segment
            # Remove segment
            seg = chain[i+1 : i+1+seg_len]
            # The "gap" is between i and i+seg_len+1 (original indices)
            # In the remaining chain, the predecessor is i, successor is i+seg_len+1
            
            # We want to insert it at a new position k
            # k is an index in the *original* chain, but we need to be careful with indices.
            # Let's construct the "fixed" chain indices
            # Fixed part: 0..i and i+seg_len+1..n-1
            
            # Candidate insertion points:
            # 1. Between any two adjacent residues in the fixed part.
            # 2. At the ends? No, must be internal to maintain connectivity?
            #    Actually, we can attach to any residue `j` in the fixed part.
            #    The new segment starts at a neighbor of `chain[j]`.
            
            # To simplify, let's just pick a random `j` from the fixed part (excluding i and i+seg_len+1 neighbors to avoid trivial moves? No, allow all).
            # However, we must ensure the segment doesn't overlap.
            
            # Let's pick a random `j` from `0` to `n-1` that is not in the segment.
            # If `j` is `i` or `i+seg_len+1`, it's a local move (shift).
            # If `j` is far, it's a global move.
            
            valid_js = [idx for idx in range(n) if not (i < idx < i+seg_len+1)]
            if not valid_js: continue
            j = rng.choice(valid_js)
            
            # We want to attach `seg[0]` to `chain[j]`.
            # `seg[0]` must be a neighbor of `chain[j]`.
            # Let's pick a random neighbor `new_pos` of `chain[j]`.
            neighbors = list(_neighbours(chain[j]))
            rng.shuffle(neighbors)
            
            moved = False
            for new_pos in neighbors:
                # Check if new_pos is occupied by fixed part
                if new_pos in where:
                    idx_occupied = where[new_pos]
                    if not (i < idx_occupied < i+seg_len+1):
                        continue # Occupied by fixed part, invalid
                
                # Construct new segment positions
                # We need to translate `seg` so that `seg[0]` is at `new_pos`.
                # Old `seg[0]` is `chain[i+1]`.
                offset = [new_pos[d] - chain[i+1][d] for d in range(dimension)]
                
                new_seg = []
                valid = True
                occupied_set = set()
                
                # Check self-avoidance of the new segment and against fixed part
                for s_idx, pos in enumerate(seg):
                    new_s_pos = tuple(pos[d] + offset[d] for d in range(dimension))
                    
                    # Check against fixed part
                    if new_s_pos in where:
                        idx_occ = where[new_s_pos]
                        if not (i < idx_occ < i+seg_len+1):
                            valid = False
                            break
                    
                    # Check against other parts of the new segment (self-avoidance)
                    if new_s_pos in occupied_set:
                        valid = False
                        break
                    occupied_set.add(new_s_pos)
                    
                    new_seg.append(new_s_pos)
                
                if not valid:
                    continue
                
                # Check connectivity: new_seg[0] must be adjacent to chain[j]
                # We picked new_pos as neighbor of chain[j], so this is true.
                # Also need to check if the segment was attached to something else?
                # The segment is detached from i and i+seg_len+1.
                # It is now attached to j.
                # We must ensure that if j was i or i+seg_len+1, we don't create a loop or conflict.
                # If j == i, the segment is re-attached to its left neighbor.
                # If j == i+seg_len+1, re-attached to right neighbor.
                
                # Calculate Energy Change
                # Old Inter-contacts:
                # Contacts between seg and fixed part.
                old_inter = 0
                # For each residue in seg, check neighbors in fixed part
                for s_idx, pos in enumerate(seg):
                    if sequence[i+1+s_idx] == 'H':
                        for nb in _neighbours(pos):
                            idx_nb = where.get(nb)
                            if idx_nb is not None:
                                if not (i < idx_nb < i+seg_len+1):
                                    if sequence[idx_nb] == 'H':
                                        old_inter += 1
                                        
                # New Inter-contacts:
                new_inter = 0
                for s_idx, pos in enumerate(new_seg):
                    if sequence[i+1+s_idx] == 'H':
                        for nb in _neighbours(pos):
                            idx_nb = where.get(nb)
                            if idx_nb is not None:
                                if not (i < idx_nb < i+seg_len+1):
                                    if sequence[idx_nb] == 'H':
                                        new_inter += 1
                                        
                delta_contacts = new_inter - old_inter
                delta_energy = -delta_contacts
                
                if delta_energy <= 0 or rng.random() < math.exp(-delta_energy / temp):
                    # Accept
                    # Update chain and where
                    # Remove old seg positions from where
                    for pos in seg:
                        del where[pos]
                    # Add new seg positions to where
                    for idx, pos in enumerate(new_seg):
                        where[pos] = i+1+idx
                    # Update chain
                    chain[i+1 : i+1+seg_len] = new_seg
                    
                    current_energy += delta_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                    moved = True
                    break
            
            if moved:
                continue
                
    return best_chain

def fold(sequence, dimension):
    # ... same as before
