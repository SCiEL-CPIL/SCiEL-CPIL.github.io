import random
import time
import sys
import math

# EVOLVE-BLOCK-START

_GLOBAL_START = time.time()
_GLOBAL_DEADLINE = _GLOBAL_START + 115.0  # Safety margin from 120s

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
            delta += 1
            
    # Add contacts with new_pos
    for q in _neighbours(new_pos):
        j = pos_dict.get(q)
        if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
            delta -= 1
            
    return delta

def _greedy_grow(sequence, rng=None, temperature=0.0):
    """Fast greedy growth to get an initial conformation."""
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
            open_count = 0
            for q in _neighbours(site):
                if q not in occupied:
                    open_count += 1
            score = gain * 10 + open_count
            candidates.append((score, site))
        
        if not candidates:
            return None, 0
        
        if temperature > 0 and len(candidates) > 1:
            scores = [c[0] for c in candidates]
            max_score = max(scores)
            exp_scores = [math.exp((s - max_score) / temperature) for s in scores]
            total = sum(exp_scores)
            r = rng.random() * total
            cumsum = 0
            chosen = candidates[-1][1]
            for idx, es in enumerate(exp_scores):
                cumsum += es
                if r <= cumsum:
                    chosen = candidates[idx][1]
                    break
        else:
            candidates.sort(key=lambda x: -x[0])
            chosen = candidates[0][1]
        
        chain.append(chosen)
        occupied[chosen] = i
    
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
        
        next_pos = rng.choice(candidates)
        chain.append(next_pos)
        occupied.add(next_pos)
    
    return chain

def _local_search(sequence, chain, deadline, rng, aggressive=False):
    """
    Local search with single-residue moves, segment moves, pinch moves, and two-point swaps.
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
    
    if aggressive:
        T = 3.0
        cooling = 0.999
        min_T = 0.1
    else:
        T = 1.5
        cooling = 0.9995
        min_T = 0.05
    
    steps = 0
    no_improve_count = 0
    
    while time.time() < deadline:
        T = max(T * cooling, min_T)
        steps += 1
        
        move_type = rng.random()
        if move_type < 0.35 and n > 5:
            # Single residue move
            i = rng.randint(1, n - 2)
            
            if i == 0 or i == n - 1:
                neighbor_idx = 1 if i == 0 else n - 2
                neighbor_pos = current_chain[neighbor_idx]
                old_pos = current_chain[i]
                
                candidates = []
                for nb in _neighbours(neighbor_pos):
                    if nb == old_pos:
                        continue
                    if nb in pos_dict and pos_dict[nb] != i:
                        continue
                    candidates.append(nb)
                
                if not candidates:
                    continue
                
                new_pos = rng.choice(candidates)
                
                delta = _energy_delta_contact(sequence, pos_dict, i, old_pos, new_pos)
                
                if delta < 0 or rng.random() < math.exp(-delta / T):
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
                prev_pos = current_chain[i - 1]
                next_pos = current_chain[i + 1]
                old_pos = current_chain[i]
                
                candidates = []
                for nb in _neighbours(prev_pos):
                    if nb == old_pos:
                        continue
                    if nb in _neighbours(next_pos):
                        if nb in pos_dict and pos_dict[nb] != i:
                            continue
                        candidates.append(nb)
                
                if not candidates:
                    continue
                
                new_pos = rng.choice(candidates)
                
                delta = _energy_delta_contact(sequence, pos_dict, i, old_pos, new_pos)
                
                if delta < 0 or rng.random() < math.exp(-delta / T):
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
        elif move_type < 0.55 and n > 5:
            # Two-point swap move: swap positions of two non-adjacent residues
            i, j = rng.sample(range(n), 2)
            if abs(i - j) <= 2:
                continue
            
            pos_i = current_chain[i]
            pos_j = current_chain[j]
            
            # Check if swapping maintains self-avoidance
            # After swap, residue i goes to pos_j and residue j goes to pos_i
            # Need to check that pos_j is not occupied by anything other than i (it's occupied by j)
            # and pos_i is not occupied by anything other than j (it's occupied by i)
            # Since we're swapping, the only issue is if the new positions conflict with neighbors
            # Actually for a simple swap of two residues in the chain, we need to verify
            # that the chain remains a valid self-avoiding walk with unit steps.
            # This is complex, so let's do a simpler check: just try it and verify.
            
            # Check chain connectivity after swap
            new_chain = list(current_chain)
            new_chain[i], new_chain[j] = pos_j, pos_i
            
            # Verify self-avoidance
            if len(set(new_chain)) != n:
                continue
            
            # Verify unit steps
            valid = True
            for k in range(n - 1):
                dx = abs(new_chain[k][0] - new_chain[k+1][0])
                dy = abs(new_chain[k][1] - new_chain[k+1][1])
                if dx + dy != 1:
                    valid = False
                    break
            
            if not valid:
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
        elif move_type < 0.75 and n > 5:
            # Pinch move: pick two H residues that are far apart, try to bring them closer
            h_indices = [i for i in range(n) if sequence[i] == 'H']
            if len(h_indices) >= 2:
                i, j = rng.sample(h_indices, 2)
                if abs(i - j) <= 2:
                    continue
                
                target_pos = current_chain[j]
                old_pos = current_chain[i]
                
                candidates = []
                for nb in _neighbours(target_pos):
                    if nb == old_pos:
                        continue
                    if nb in pos_dict and pos_dict[nb] != i:
                        continue
                    if i == 0:
                        if nb != current_chain[1]:
                            continue
                    elif i == n - 1:
                        if nb != current_chain[n-2]:
                            continue
                    else:
                        if nb not in _neighbours(current_chain[i-1]):
                            continue
                        if nb not in _neighbours(current_chain[i+1]):
                            continue
                    candidates.append(nb)
                
                if candidates:
                    new_pos = rng.choice(candidates)
                    delta = _energy_delta_contact(sequence, pos_dict, i, old_pos, new_pos)
                    
                    if delta < 0 or rng.random() < math.exp(-delta / T):
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
            max_seg = min(n // 2, 12)
            if max_seg < 2:
                max_seg = 2
            seg_len = rng.randint(2, max_seg)
            i = rng.randint(0, n - seg_len)
            j = i + seg_len - 1
            
            head = current_chain[:i]
            tail = current_chain[j+1:]
            
            occupied_ht = set(head) | set(tail)
            
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
            
            if tail:
                tail_start = tail[0]
                d = abs(new_segment[-1][0] - tail_start[0]) + abs(new_segment[-1][1] - tail_start[1])
                if d != 1:
                    continue
            
            new_chain = head + new_segment + tail
            
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
        
        if no_improve_count > 300:
            T = max(T * 1.5, 1.0)
            no_improve_count = 0
    
    return best_chain, best_energy

def _polish_pass(sequence, chain, deadline, rng):
    """
    A high-precision polish pass using only improving moves (hill climbing).
    Includes two-point swap checks for additional improvement.
    """
    n = len(sequence)
    if n < 3:
        return list(chain), _energy(sequence, chain)
    
    current_chain = list(chain)
    pos_dict = {p: i for i, p in enumerate(current_chain)}
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    no_improve = 0
    max_no_improve = 150
    
    while time.time() < deadline and no_improve < max_no_improve:
        improved = False
        
        # Single residue moves
        for i in range(n):
            if i == 0 or i == n - 1:
                neighbor_idx = 1 if i == 0 else n - 2
                neighbor_pos = current_chain[neighbor_idx]
                old_pos = current_chain[i]
                
                candidates = []
                for nb in _neighbours(neighbor_pos):
                    if nb == old_pos:
                        continue
                    if nb in pos_dict and pos_dict[nb] != i:
                        continue
                    candidates.append(nb)
                
                for new_pos in candidates:
                    delta = _energy_delta_contact(sequence, pos_dict, i, old_pos, new_pos)
                    if delta < 0:
                        del pos_dict[old_pos]
                        current_chain[i] = new_pos
                        pos_dict[new_pos] = i
                        current_energy += delta
                        improved = True
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(current_chain)
                        break
            else:
                prev_pos = current_chain[i - 1]
                next_pos = current_chain[i + 1]
                old_pos = current_chain[i]
                
                candidates = []
                for nb in _neighbours(prev_pos):
                    if nb == old_pos:
                        continue
                    if nb in _neighbours(next_pos):
                        if nb in pos_dict and pos_dict[nb] != i:
                            continue
                        candidates.append(nb)
                
                for new_pos in candidates:
                    delta = _energy_delta_contact(sequence, pos_dict, i, old_pos, new_pos)
                    if delta < 0:
                        del pos_dict[old_pos]
                        current_chain[i] = new_pos
                        pos_dict[new_pos] = i
                        current_energy += delta
                        improved = True
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_chain = list(current_chain)
                        break
        
        # Two-point swap moves in polish
        if not improved and n > 5:
            h_indices = [i for i in range(n) if sequence[i] == 'H']
            if len(h_indices) >= 2:
                # Try swapping pairs of H residues
                for idx1 in range(len(h_indices)):
                    for idx2 in range(idx1 + 1, len(h_indices)):
                        i, j = h_indices[idx1], h_indices[idx2]
                        if abs(i - j) <= 2:
                            continue
                        
                        pos_i = current_chain[i]
                        pos_j = current_chain[j]
                        
                        new_chain = list(current_chain)
                        new_chain[i], new_chain[j] = pos_j, pos_i
                        
                        if len(set(new_chain)) != n:
                            continue
                        
                        valid = True
                        for k in range(n - 1):
                            dx = abs(new_chain[k][0] - new_chain[k+1][0])
                            dy = abs(new_chain[k][1] - new_chain[k+1][1])
                            if dx + dy != 1:
                                valid = False
                                break
                        
                        if not valid:
                            continue
                        
                        new_energy = _energy(sequence, new_chain)
                        if new_energy < current_energy:
                            current_chain = new_chain
                            current_energy = new_energy
                            pos_dict = {p: idx for idx, p in enumerate(current_chain)}
                            improved = True
                            if current_energy < best_energy:
                                best_energy = current_energy
                                best_chain = list(current_chain)
                            break
                    if improved:
                        break
        
        if not improved:
            no_improve += 1
        else:
            no_improve = 0
    
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
        
        if r % 4 == 0:
            init_chain, init_energy = _greedy_grow(sequence, rng, temperature=0.0)
            if init_chain is None:
                init_chain = [(i, 0) for i in range(n)]
                init_energy = _energy(sequence, init_chain)
        elif r % 4 == 1:
            init_chain, init_energy = _greedy_grow(sequence, rng, temperature=2.0)
            if init_chain is None:
                init_chain = [(i, 0) for i in range(n)]
                init_energy = _energy(sequence, init_chain)
        elif r % 4 == 2:
            init_chain = _random_walk(sequence, rng)
            if init_chain is None:
                init_chain, init_energy = _greedy_grow(sequence, rng, temperature=0.0)
                if init_chain is None:
                    init_chain = [(i, 0) for i in range(n)]
                    init_energy = _energy(sequence, init_chain)
            else:
                init_energy = _energy(sequence, init_chain)
        else:
            # Higher temperature greedy for more diversity
            init_chain, init_energy = _greedy_grow(sequence, rng, temperature=5.0)
            if init_chain is None:
                init_chain = [(i, 0) for i in range(n)]
                init_energy = _energy(sequence, init_chain)
        
        aggressive = r >= n_restarts // 2
        ls_deadline = restart_deadline - 0.05
        ls_chain, ls_energy = _local_search(sequence, init_chain, ls_deadline, rng, aggressive=aggressive)
        
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
    
    remaining = _GLOBAL_DEADLINE - time.time()
    if remaining < 1.0:
        chain, _ = _greedy_grow(sequence)
        if chain is None:
            return [list(p) for p in [(i, 0) for i in range(n)]]
        return [list(p) for p in chain]
    
    if n <= 30:
        seq_time = min(8.0, remaining * 0.35)
    elif n <= 50:
        seq_time = min(10.0, remaining * 0.30)
    else:
        seq_time = min(12.0, remaining * 0.25)
    
    deadline = time.time() + seq_time
    
    if n == 0:
        return []
    
    if n <= 30:
        n_restarts = 28
    elif n <= 50:
        n_restarts = 22
    else:
        n_restarts = 18
    
    rng = random.Random(12345 + n)
    
    best_chain, best_energy = _restart_local_search(sequence, n_restarts, deadline, rng)
    
    polish_deadline = min(deadline, _GLOBAL_DEADLINE - 0.5)
    if time.time() < polish_deadline:
        best_chain, best_energy = _polish_pass(sequence, best_chain, polish_deadline, rng)
    
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
