import random
import time
import sys
import math

# EVOLVE-BLOCK-START

def _neighbours(point):
    """Return the 4 neighbours of a point on the square lattice."""
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

def _energy(sequence, chain):
    """Compute the HP contact energy of a chain conformation.
    Energy = - (number of H-H contacts)."""
    n = len(chain)
    if n == 0:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _energy_delta_contact(sequence, pos_dict, i, old_pos, new_pos):
    """
    Compute the change in energy when moving residue i from old_pos to new_pos.
    pos_dict maps position -> index.
    """
    if sequence[i] != 'H':
        return 0
    
    delta = 0
    
    # Remove contacts with old_pos
    for q in _neighbours(old_pos):
        j = pos_dict.get(q)
        if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
            delta += 1  # Losing a contact increases energy by 1
            
    # Add contacts with new_pos
    for q in _neighbours(new_pos):
        j = pos_dict.get(q)
        if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
            delta -= 1  # Gaining a contact decreases energy by 1
            
    return delta

def _greedy_grow(sequence, rng=None, temperature=0.0):
    """Fast greedy growth to get an initial conformation with optional stochasticity."""
    n = len(sequence)
    if n == 0:
        return [], 0
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            gain += 1
            # Also consider future potential: prefer sites with more open neighbors
            open_count = 0
            for q in _neighbours(site):
                if q not in occupied:
                    open_count += 1
            # Combine gain and openness
            score = gain * 10 + open_count
            candidates.append((score, site))
        
        if not candidates:
            return None, 0
        
        if temperature > 0 and len(candidates) > 1:
            # Softmax selection
            scores = [c[0] for c in candidates]
            max_score = max(scores)
            exp_scores = [math.exp((s - max_score) / temperature) for s in scores]
            total = sum(exp_scores)
            r = rng.random() * total
            cumsum = 0
            chosen_idx = 0
            for idx, es in enumerate(exp_scores):
                cumsum += es
                if r <= cumsum:
                    chosen_idx = idx
                    break
            best_site = candidates[chosen_idx][1]
        else:
            candidates.sort(key=lambda x: -x[0])
            best_site = candidates[0][1]
        
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain, _energy(sequence, chain)

def _random_walk(sequence, rng):
    """Generate a random self-avoiding walk."""
    n = len(sequence)
    if n == 0:
        return []
    chain = [(0, 0)]
    occupied = {(0, 0)}
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        for dx, dy in dirs:
            new_pos = (last[0] + dx, last[1] + dy)
            if new_pos not in occupied:
                candidates.append(new_pos)
        if not candidates:
            return None
        new_pos = rng.choice(candidates)
        chain.append(new_pos)
        occupied.add(new_pos)
    return chain

def _polish_pass(sequence, chain, deadline, rng):
    """High-precision polish: try all possible single-residue moves and accept improvements."""
    n = len(chain)
    if n < 3:
        return list(chain), _energy(sequence, chain)
    
    current_chain = list(chain)
    pos_dict = {p: i for i, p in enumerate(current_chain)}
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    
    while time.time() < deadline:
        improved = False
        for i in range(1, n - 1):
            if time.time() >= deadline:
                break
            old_pos = current_chain[i]
            prev_pos = current_chain[i - 1]
            next_pos = current_chain[i + 1]
            
            # Find all valid new positions for residue i
            candidates = []
            for nb in _neighbours(prev_pos):
                if nb == old_pos:
                    continue
                if nb in _neighbours(next_pos):
                    if nb not in pos_dict or pos_dict[nb] == i:
                        candidates.append(nb)
            
            for new_pos in candidates:
                delta = _energy_delta_contact(sequence, pos_dict, i, old_pos, new_pos)
                if delta < 0:
                    # Accept improvement
                    del pos_dict[old_pos]
                    current_chain[i] = new_pos
                    pos_dict[new_pos] = i
                    current_energy += delta
                    improved = True
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                    break  # Move to next residue after first improvement
    
        if not improved:
            break
    
    return best_chain, best_energy

def _local_search(sequence, chain, deadline, rng):
    """
    Local search with single-residue moves and segment moves.
    Returns (best_chain, best_energy).
    """
    n = len(sequence)
    if n < 3:
        return list(chain), _energy(sequence, chain)

    current_chain = list(chain)
    pos_dict = {p: i for i, p in enumerate(current_chain)}
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    T = 3.0
    cooling = 0.999
    min_T = 0.1
    
    start_time = time.time()
    steps = 0
    no_improve_count = 0
    
    while time.time() < deadline:
        T = max(T * cooling, min_T)
        steps += 1
        
        # Choose move type
        if rng.random() < 0.6 and n > 5:
            # Single residue move
            i = rng.randint(1, n - 2)
            
            prev_pos = current_chain[i - 1]
            next_pos = current_chain[i + 1]
            old_pos = current_chain[i]
            
            # New pos must be neighbor of prev_pos AND next_pos
            # And not occupied (except by i)
            candidates = []
            for nb in _neighbours(prev_pos):
                if nb == old_pos:
                    continue
                if nb in _neighbours(next_pos):
                    if nb not in pos_dict or pos_dict[nb] == i:
                        candidates.append(nb)
            
            if not candidates:
                continue
            
            new_pos = rng.choice(candidates)
            
            delta = _energy_delta_contact(sequence, pos_dict, i, old_pos, new_pos)
            
            if delta < 0 or rng.random() < math.exp(-delta / T):
                # Accept
                del pos_dict[old_pos]
                current_chain[i] = new_pos
                pos_dict[new_pos] = i
                current_energy += delta
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        else:
            # Segment move
            max_seg = min(n // 2, 8)
            if max_seg < 2:
                max_seg = 2
            seg_len = rng.randint(2, max_seg)
            i = rng.randint(0, n - seg_len)
            j = i + seg_len - 1
            
            head = current_chain[:i]
            tail = current_chain[j+1:]
            
            # Build occupied set for head and tail
            occupied_ht = set(head) | set(tail)
            
            # Find valid start positions for the segment
            anchor = head[-1] if head else (0, 0)
            start_candidates = []
            for nb in _neighbours(anchor):
                if nb not in occupied_ht:
                    start_candidates.append(nb)
            
            if not start_candidates:
                continue
            
            start_pos = rng.choice(start_candidates)
            new_segment = [start_pos]
            new_occupied = set(occupied_ht) | {start_pos}
            
            success = True
            for k in range(1, seg_len):
                last_pos = new_segment[-1]
                next_candidates = []
                for nb in _neighbours(last_pos):
                    if nb not in new_occupied:
                        next_candidates.append(nb)
                
                if not next_candidates:
                    success = False
                    break
                
                next_pos = rng.choice(next_candidates)
                new_segment.append(next_pos)
                new_occupied.add(next_pos)
            
            if not success:
                continue
            
            # Check connection to tail
            if tail:
                tail_start = tail[0]
                d = abs(new_segment[-1][0] - tail_start[0]) + abs(new_segment[-1][1] - tail_start[1])
                if d != 1:
                    continue
            
            new_chain = head + new_segment + tail
            
            # Verify self-avoidance
            if len(set(new_chain)) != n:
                continue
            
            new_energy = _energy(sequence, new_chain)
            delta = new_energy - current_energy
            
            if delta < 0 or rng.random() < math.exp(-delta / T):
                current_chain = new_chain
                current_energy = new_energy
                pos_dict = {p: idx for idx, p in enumerate(current_chain)}
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        
        # Restart if stuck too long
        if no_improve_count > 300:
            T = max(T * 2.0, 2.0)
            no_improve_count = 0
    
    return best_chain, best_energy

def _restart_local_search(sequence, n_restarts, deadline, rng):
    """
    Run multiple restarts of greedy + local search.
    Returns the best (chain, energy) found.
    """
    n = len(sequence)
    best_chain = None
    best_energy = 0
    
    start_time = time.time()
    per_restart = (deadline - start_time) / max(1, n_restarts)
    
    for r in range(n_restarts):
        restart_deadline = start_time + (r + 1) * per_restart
        if time.time() >= deadline:
            break
        
        # Generate initial conformation with varying temperature
        temp = 0.0 if r < n_restarts // 3 else (r % 5) * 0.5 + 0.5
        init_chain, init_energy = _greedy_grow(sequence, rng, temperature=temp)
        
        if init_chain is None:
            # Try random walk
            init_chain = _random_walk(sequence, rng)
            if init_chain is None:
                init_chain = [(i, 0) for i in range(n)]
                init_energy = _energy(sequence, init_chain)
            else:
                init_energy = _energy(sequence, init_chain)
        
        # Run local search
        ls_deadline = restart_deadline - 0.1
        ls_chain, ls_energy = _local_search(sequence, init_chain, ls_deadline, rng)
        
        # Polish pass
        if time.time() < deadline - 0.5:
            polish_deadline = min(deadline - 0.5, time.time() + (deadline - time.time()) * 0.2)
            polished_chain, polished_energy = _polish_pass(sequence, ls_chain, polish_deadline, rng)
            if polished_energy < ls_energy:
                ls_chain, ls_energy = polished_chain, polished_energy
        
        if ls_energy < best_energy:
            best_energy = ls_energy
            best_chain = ls_chain
    
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    return best_chain, best_energy

def fold(sequence, dimension):
    """Fold a sequence into a low-energy self-avoiding conformation."""
    start_time = time.time()
    n = len(sequence)
    
    # Total budget: 120s for all sequences. 
    # We have 10 sequences. 120/10 = 12s per sequence.
    # Use 10s per sequence for safety.
    deadline = start_time + 10.0
    
    if n == 0:
        return []
    
    # Number of restarts: more for longer sequences
    if n <= 20:
        n_restarts = 30
    elif n <= 40:
        n_restarts = 25
    else:
        n_restarts = 20
    
    rng = random.Random(12345 + n)
    
    best_chain, best_energy = _restart_local_search(sequence, n_restarts, deadline, rng)
    
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
