"""HP lattice protein folding with beam search and simulated annealing refinement."""
# EVOLVE-BLOCK-START
import random
import time

_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    _init_time()
    return 110.0 - (time.time() - _START_TIME)

def _neighbours(point):
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

def _count_contacts(sequence, chain):
    n = len(chain)
    if n == 0:
        return 0
    occupied = {p: i for i, p in enumerate(chain)}
    contacts = 0
    is_h = [c == 'H' for c in sequence]
    for i in range(n):
        if not is_h[i]:
            continue
        x, y = chain[i]
        for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
            j = occupied.get((nx, ny))
            if j is not None and j > i + 1 and is_h[j]:
                contacts += 1
    return contacts

def _beam_search_fold(sequence, dimension, rng, beam_width=50):
    """
    Beam search: maintain a set of partial chains, expanding each by one residue
    at a time and keeping only the top-scoring partials.
    """
    n = len(sequence)
    if n == 0:
        return None, 0
    
    is_h = [c == 'H' for c in sequence]
    
    # Each beam entry: (chain, occupied_set, contacts_count)
    # Start with a small set of initial positions to diversify
    start_positions = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)]
    
    beams = []
    for sp in start_positions:
        beams.append(([sp], {sp: 0}, 0))
    
    for step in range(1, n):
        if _time_remaining() < 2.0:
            return None, 0
        
        new_beams = []
        
        for chain, occupied, contacts in beams:
            current = chain[-1]
            is_h_curr = is_h[step]
            
            for site in _neighbours(current):
                if site in occupied:
                    continue
                
                # Calculate contact gain
                gain = 0
                if is_h_curr:
                    for nb in _neighbours(site):
                        j = occupied.get(nb)
                        if j is not None and j < step - 1 and is_h[j]:
                            gain += 1
                
                # Count free neighbours for flexibility
                free = sum(1 for nb in _neighbours(site) if nb not in occupied)
                
                # Score: contacts dominate, then free space, then diversity
                score = gain * 100.0 + free * 1.0 + rng.random() * 0.5
                
                new_chain = chain + [site]
                new_occupied = dict(occupied)
                new_occupied[site] = step
                new_contacts = contacts + gain
                
                new_beams.append((score, new_chain, new_occupied, new_contacts))
        
        if not new_beams:
            return None, 0
        
        # Keep top beam_width entries
        new_beams.sort(key=lambda x: x[0], reverse=True)
        top_beams = new_beams[:beam_width]
        
        # Deduplicate chains to maintain diversity
        seen = set()
        diverse_beams = []
        for score, chain, occupied, contacts in top_beams:
            key = tuple(chain)
            if key not in seen:
                seen.add(key)
                diverse_beams.append((chain, occupied, contacts))
        
        # If deduplication reduced us too much, add back from the rest
        if len(diverse_beams) < beam_width // 2:
            for score, chain, occupied, contacts in top_beams:
                if len(diverse_beams) >= beam_width:
                    break
                key = tuple(chain)
                if key not in seen:
                    seen.add(key)
                    diverse_beams.append((chain, occupied, contacts))
        
        beams = diverse_beams
        if not beams:
            return None, 0
    
    # Return the best complete chain
    best = max(beams, key=lambda x: x[2])
    return best[0], -best[2]

def _local_search(sequence, chain, rng, max_steps=300):
    """
    Local search with kink moves and segment moves using simulated annealing.
    """
    n = len(chain)
    if n < 3:
        return chain, -_count_contacts(sequence, chain)
    
    occupied = {p: i for i, p in enumerate(chain)}
    current_contacts = _count_contacts(sequence, chain)
    current_energy = -current_contacts
    is_h = [c == 'H' for c in sequence]
    
    start_temp = 1.5
    end_temp = 0.03
    
    for step in range(max_steps):
        if _time_remaining() < 1.0:
            break
        
        progress = step / max(1, max_steps)
        temp = start_temp * (end_temp / start_temp) ** progress
        
        move_type = rng.random()
        
        if move_type < 0.65:
            # Kink move
            if n < 3:
                continue
            i = rng.randint(1, n - 2)
            prev = chain[i-1]
            curr = chain[i]
            next_ = chain[i+1]
            
            candidates = set()
            for p in _neighbours(prev):
                if p == curr:
                    continue
                dist = abs(p[0] - next_[0]) + abs(p[1] - next_[1])
                if dist == 1:
                    candidates.add(p)
            
            if not candidates:
                continue
            
            new_pos = rng.choice(list(candidates))
            if new_pos in occupied:
                continue
            
            # Apply
            del occupied[curr]
            occupied[new_pos] = i
            chain[i] = new_pos
            
            new_contacts = _count_contacts(sequence, chain)
            new_energy = -new_contacts
            delta = new_energy - current_energy
            
            if delta < 0 or (temp > 0.01 and rng.random() < pow(2.718281828, -delta / temp)):
                current_energy = new_energy
                current_contacts = new_contacts
            else:
                chain[i] = curr
                del occupied[new_pos]
                occupied[curr] = i
                
        elif move_type < 0.85:
            # Segment move length 2
            if n < 4:
                continue
            start_idx = rng.randint(1, n - 3)
            end_idx = start_idx + 1
            
            seg_indices = [start_idx, end_idx]
            old_seg = [chain[i] for i in seg_indices]
            
            anchor_start = chain[start_idx - 1]
            anchor_end = chain[end_idx + 1]
            
            found_valid = False
            for _ in range(30):
                pos = anchor_start
                path = [pos]
                for _ in range(2):
                    pos = rng.choice(_neighbours(pos))
                    path.append(pos)
                
                if path[-1] != anchor_end:
                    continue
                
                new_seg = path[1:]
                
                collision = False
                for p in new_seg:
                    if p in occupied:
                        if p not in old_seg:
                            collision = True
                            break
                
                if collision or len(set(new_seg)) != 2:
                    continue
                
                for i in seg_indices:
                    del occupied[chain[i]]
                
                for k, idx in enumerate(seg_indices):
                    occupied[new_seg[k]] = idx
                    chain[idx] = new_seg[k]
                
                new_contacts = _count_contacts(sequence, chain)
                new_energy = -new_contacts
                delta = new_energy - current_energy
                
                if delta < 0 or (temp > 0.01 and rng.random() < pow(2.718281828, -delta / temp)):
                    current_energy = new_energy
                    current_contacts = new_contacts
                    found_valid = True
                else:
                    for k, idx in enumerate(seg_indices):
                        del occupied[chain[idx]]
                        occupied[old_seg[k]] = idx
                        chain[idx] = old_seg[k]
                
                break
            
        else:
            # Segment move length 3
            if n < 5:
                continue
            start_idx = rng.randint(1, n - 4)
            end_idx = start_idx + 2
            
            seg_indices = list(range(start_idx, end_idx + 1))
            old_seg = [chain[i] for i in seg_indices]
            
            anchor_start = chain[start_idx - 1]
            anchor_end = chain[end_idx + 1]
            
            found_valid = False
            for _ in range(20):
                pos = anchor_start
                path = [pos]
                for _ in range(3):
                    pos = rng.choice(_neighbours(pos))
                    path.append(pos)
                
                if path[-1] != anchor_end:
                    continue
                
                new_seg = path[1:]
                
                collision = False
                for p in new_seg:
                    if p in occupied:
                        if p not in old_seg:
                            collision = True
                            break
                
                if collision or len(set(new_seg)) != 3:
                    continue
                
                for i in seg_indices:
                    del occupied[chain[i]]
                
                for k, idx in enumerate(seg_indices):
                    occupied[new_seg[k]] = idx
                    chain[idx] = new_seg[k]
                
                new_contacts = _count_contacts(sequence, chain)
                new_energy = -new_contacts
                delta = new_energy - current_energy
                
                if delta < 0 or (temp > 0.01 and rng.random() < pow(2.718281828, -delta / temp)):
                    current_energy = new_energy
                    current_contacts = new_contacts
                    found_valid = True
                else:
                    for k, idx in enumerate(seg_indices):
                        del occupied[chain[idx]]
                        occupied[old_seg[k]] = idx
                        chain[idx] = old_seg[k]
                
                break
    
    return chain, current_energy

def fold(sequence, dimension):
    """
    Fold an HP sequence on the square lattice.
    Uses beam search to explore multiple topologies, then local search refinement.
    """
    _init_time()
    
    n = len(sequence)
    if n == 0:
        return []
    
    if _time_remaining() < 1.0:
        return [[i, 0] for i in range(n)]
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0
    
    # Adaptive beam width based on sequence length
    if n <= 30:
        beam_width = 80
    elif n <= 60:
        beam_width = 50
    else:
        beam_width = 30
    
    # Multiple beam search restarts
    max_restarts = 30
    restart = 0
    
    while _time_remaining() > 8.0 and restart < max_restarts:
        restart += 1
        
        # Beam search with slightly varied parameters for diversity
        bw = max(10, beam_width + rng.randint(-10, 10))
        chain, energy = _beam_search_fold(sequence, dimension, rng, beam_width=bw)
        
        if chain is None:
            continue
        
        # Local search refinement
        ls_steps = 200 if n <= 40 else 150
        chain, energy = _local_search(sequence, chain, rng, max_steps=ls_steps)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
