"""
HP lattice protein folding using Beam Search + Local Search Refinement.
Combines beam search for initial conformation generation with simulated annealing
local search (kink moves and segment moves) for refinement.
"""
# EVOLVE-BLOCK-START
import time
import random

_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    _init_time()
    return 200.0 - (time.time() - _START_TIME)

def _neighbours(point):
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

def _count_contacts_fast(sequence, chain):
    """Fast contact counting using position-to-index mapping."""
    n = len(chain)
    if n == 0:
        return 0
    is_h = [c == 'H' for c in sequence]
    pos_to_idx = {}
    for i, p in enumerate(chain):
        pos_to_idx[p] = i
    contacts = 0
    for i in range(n):
        if not is_h[i]:
            continue
        x, y = chain[i]
        for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
            j = pos_to_idx.get((nx, ny))
            if j is not None and j > i + 1 and is_h[j]:
                contacts += 1
    return contacts

def _beam_search(sequence, dimension, beam_width, time_limit, rng):
    """Beam search returning best partial conformation."""
    n = len(sequence)
    is_h = [c == 'H' for c in sequence]
    start_pos = (0, 0)
    initial_chain = [start_pos]
    initial_occupied = {start_pos}
    initial_h = {start_pos} if is_h[0] else set()
    initial_energy = 0
    beam = [(initial_energy, initial_chain, initial_occupied, initial_h)]
    start_time = time.time()
    
    for i in range(1, n):
        new_beam = []
        is_current_h = is_h[i]
        for energy, chain, occupied, h_pos in beam:
            last_pos = chain[-1]
            pred_pos = chain[-2] if len(chain) > 1 else None
            for nb in _neighbours(last_pos):
                if nb in occupied:
                    continue
                new_energy = energy
                new_h_pos = h_pos
                new_occupied = occupied | {nb}
                if is_current_h:
                    new_h_pos = h_pos | {nb}
                    for q in _neighbours(nb):
                        if q in h_pos and q != pred_pos:
                            new_energy -= 1
                new_chain = chain + [nb]
                new_beam.append((new_energy, new_chain, new_occupied, new_h_pos))
        if not new_beam:
            break
        new_beam.sort(key=lambda x: x[0])
        if len(new_beam) > beam_width:
            cutoff = new_beam[beam_width - 1][0]
            keep = [s for s in new_beam if s[0] < cutoff]
            maybe = [s for s in new_beam if s[0] == cutoff]
            slots = beam_width - len(keep)
            if slots > 0:
                rng.shuffle(maybe)
                keep.extend(maybe[:slots])
            beam = keep
        else:
            beam = new_beam
        if time.time() - start_time > time_limit:
            break
    
    if not beam:
        return None
    best = min(beam, key=lambda x: x[0])
    chain = best[1]
    occupied = best[2]
    h_pos = best[3]
    
    if len(chain) < n:
        for j in range(len(chain), n):
            is_j_h = is_h[j]
            last_pos = chain[-1]
            pred_pos = chain[-2] if len(chain) > 1 else None
            candidates = []
            for nb in _neighbours(last_pos):
                if nb in occupied:
                    continue
                e = 0
                if is_j_h:
                    for q in _neighbours(nb):
                        if q in h_pos and q != pred_pos:
                            e -= 1
                candidates.append((e, nb))
            if not candidates:
                return None
            candidates.sort(key=lambda x: x[0])
            best_val = candidates[0][0]
            best_opts = [nb for e, nb in candidates if e == best_val]
            best_nb = rng.choice(best_opts)
            chain.append(best_nb)
            occupied.add(best_nb)
            if is_j_h:
                h_pos.add(best_nb)
    return chain

def _local_search(sequence, chain, rng, max_steps, time_limit):
    """Simulated annealing local search with kink and segment moves."""
    n = len(chain)
    if n < 3:
        return chain, -_count_contacts_fast(sequence, chain)
    
    chain = list(chain)
    occupied = {p: i for i, p in enumerate(chain)}
    is_h = [c == 'H' for c in sequence]
    current_contacts = _count_contacts_fast(sequence, chain)
    current_energy = -current_contacts
    best_chain = list(chain)
    best_energy = current_energy
    
    start_temp = 2.0
    end_temp = 0.05
    start_time = time.time()
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
        progress = step / max(1, max_steps)
        temp = start_temp * (end_temp / start_temp) ** progress
        move_type = rng.random()
        
        if move_type < 0.7:
            # Kink move
            i = rng.randint(1, n - 2)
            prev = chain[i-1]
            curr = chain[i]
            next_ = chain[i+1]
            candidates = set()
            for p in _neighbours(prev):
                if p == curr:
                    continue
                if abs(p[0] - next_[0]) + abs(p[1] - next_[1]) == 1:
                    candidates.add(p)
            if not candidates:
                continue
            new_pos = rng.choice(list(candidates))
            if new_pos in occupied:
                continue
            del occupied[curr]
            occupied[new_pos] = i
            chain[i] = new_pos
            new_contacts = _count_contacts_fast(sequence, chain)
            new_energy = -new_contacts
            delta = new_energy - current_energy
            if delta < 0 or (temp > 0.01 and rng.random() < 2.718281828 ** (-delta / temp)):
                current_energy = new_energy
                current_contacts = new_contacts
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
            else:
                chain[i] = curr
                del occupied[new_pos]
                occupied[curr] = i
        else:
            # Segment move
            seg_len = rng.choice([2, 3])
            if n < seg_len + 2:
                continue
            start_idx = rng.randint(1, n - seg_len - 1)
            end_idx = start_idx + seg_len - 1
            seg_indices = list(range(start_idx, end_idx + 1))
            old_seg = [chain[i] for i in seg_indices]
            anchor_start = chain[start_idx - 1]
            anchor_end = chain[end_idx + 1]
            found_valid = False
            for _ in range(30):
                pos = anchor_start
                path = [pos]
                valid = True
                for _ in range(seg_len):
                    pos = rng.choice(_neighbours(pos))
                    path.append(pos)
                if path[-1] != anchor_end:
                    continue
                new_seg = path[1:]
                collision = False
                for p in new_seg:
                    if p in occupied:
                        is_old = any(p == op for op in old_seg)
                        if not is_old:
                            collision = True
                            break
                if collision:
                    continue
                if len(set(new_seg)) != len(new_seg):
                    continue
                for idx in seg_indices:
                    del occupied[chain[idx]]
                for k, idx in enumerate(seg_indices):
                    occupied[new_seg[k]] = idx
                    chain[idx] = new_seg[k]
                new_contacts = _count_contacts_fast(sequence, chain)
                new_energy = -new_contacts
                delta = new_energy - current_energy
                if delta < 0 or (temp > 0.01 and rng.random() < 2.718281828 ** (-delta / temp)):
                    current_energy = new_energy
                    current_contacts = new_contacts
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                    found_valid = True
                    break
                else:
                    for idx in seg_indices:
                        del occupied[chain[idx]]
                    for k, idx in enumerate(seg_indices):
                        occupied[old_seg[k]] = idx
                        chain[idx] = old_seg[k]
    
    return best_chain, best_energy

def fold(sequence, dimension):
    _init_time()
    n = len(sequence)
    
    if n <= 20:
        base_bw = 300
        base_tl = 4.0
        ls_steps = 300
        ls_tl = 5.0
    elif n <= 50:
        base_bw = 150
        base_tl = 8.0
        ls_steps = 200
        ls_tl = 4.0
    else:
        base_bw = 80
        base_tl = 12.0
        ls_steps = 150
        ls_tl = 3.0
    
    best_chain = None
    best_energy = 0
    
    max_runs = 8
    for run in range(max_runs):
        if _time_remaining() < 15:
            break
        
        seed = 12345 + run * 7
        rng = random.Random(seed)
        
        chain = _beam_search(sequence, dimension, base_bw, base_tl, rng)
        
        if chain is not None and len(chain) == n:
            # Validate
            valid = True
            occ = set()
            for i, p in enumerate(chain):
                if p in occ:
                    valid = False
                    break
                occ.add(p)
                if i > 0:
                    prev = chain[i-1]
                    if abs(p[0]-prev[0]) + abs(p[1]-prev[1]) != 1:
                        valid = False
                        break
            if valid:
                # Run local search
                refined_chain, refined_energy = _local_search(sequence, chain, rng, ls_steps, ls_tl)
                if refined_energy < best_energy:
                    best_energy = refined_energy
                    best_chain = refined_chain
    
    if best_chain is None:
        best_chain = [[i, 0] for i in range(n)]
    
    return best_chain
# EVOLVE-BLOCK-END
