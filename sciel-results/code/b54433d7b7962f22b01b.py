import random
import time
import sys
import math

# EVOLVE-BLOCK-START

# Global time management to ensure we don't exceed the 120s budget
# The evaluator allows 120s CPU. We use 115s to be safe against OS overhead.
_GLOBAL_START = time.time()
_GLOBAL_DEADLINE = _GLOBAL_START + 115.0

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
            delta += 1  # Losing a contact increases energy (makes it less negative)
            
    # Add contacts with new_pos
    for q in _neighbours(new_pos):
        j = pos_dict.get(q)
        if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
            delta -= 1  # Gaining a contact decreases energy (makes it more negative)
            
    return delta

def _greedy_grow(sequence, rng=None, temperature=0.5):
    """
    Fast greedy growth to get an initial conformation.
    Uses a softmax-like selection to allow some exploration, 
    but strongly favors high-gain moves.
    """
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
            # Openness heuristic: prefer sites that don't trap the chain
            open_count = 0
            for q in _neighbours(site):
                if q not in occupied:
                    open_count += 1
            # Score: high gain is critical, openness prevents dead-ends
            score = gain * 10.0 + open_count
            candidates.append((score, site))
        
        if not candidates:
            return None, 0
        
        # Selection strategy
        if temperature > 0 and len(candidates) > 1:
            # Softmax selection
            max_score = max(c[0] for c in candidates)
            # Scale temperature to be relative to score differences
            # Scores are typically small integers, so T=0.5 is reasonable
            exp_scores = [math.exp((s - max_score) / temperature) for s, _ in candidates]
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

def _local_search(sequence, chain, deadline, rng):
    """
    Local search with single-residue moves and targeted pinch moves.
    Pinch moves specifically attempt to bring two H residues together.
    """
    n = len(sequence)
    if n < 3:
        return list(chain), _energy(sequence, chain)

    current_chain = list(chain)
    pos_dict = {p: i for i, p in enumerate(current_chain)}
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Adaptive cooling
    T = 2.0
    cooling = 0.999
    min_T = 0.05
    
    no_improve_count = 0
    last_improve_time = time.time()
    
    # Pre-compute H indices for faster pinch move generation
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    
    while time.time() < deadline:
        T = max(T * cooling, min_T)
        
        # Decide move type:
        # 60% Single residue (cheap, good for fine tuning)
        # 30% Pinch (targeted, good for finding new contacts)
        # 10% Segment (expensive, good for large rearrangements if stuck)
        r_move = rng.random()
        
        move_accepted = False
        delta = 0
        
        if r_move < 0.6:
            # Single residue move
            # Pick a random internal residue or end
            if n > 2 and rng.random() < 0.8:
                i = rng.randint(1, n - 2)
            else:
                i = rng.choice([0, n-1])
                
            old_pos = current_chain[i]
            
            candidates = []
            if i == 0:
                neighbor_pos = current_chain[1]
                for nb in _neighbours(neighbor_pos):
                    if nb != old_pos and nb not in pos_dict:
                        candidates.append(nb)
            elif i == n - 1:
                neighbor_pos = current_chain[n-2]
                for nb in _neighbours(neighbor_pos):
                    if nb != old_pos and nb not in pos_dict:
                        candidates.append(nb)
            else:
                prev_pos = current_chain[i-1]
                next_pos = current_chain[i+1]
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
                # Accept move
                del pos_dict[old_pos]
                current_chain[i] = new_pos
                pos_dict[new_pos] = i
                current_energy += delta
                move_accepted = True

        elif r_move < 0.9 and len(h_indices) > 1:
            # Pinch Move: Try to bring two H residues closer
            # Pick a random H residue (i) and try to move it next to another H residue (j)
            # or vice versa.
            # We pick i to move, and look for a target H residue j.
            if rng.random() < 0.5:
                i = rng.choice(h_indices)
            else:
                i = rng.choice(h_indices)
                
            # To make it efficient, we don't check all j. We just try to move i 
            # to a random empty neighbor of a random H residue j != i.
            # This is a heuristic "pinch".
            
            # Pick a target H residue j
            if len(h_indices) > 1:
                # Avoid picking i itself
                j_candidates = [j for j in h_indices if j != i]
                if j_candidates:
                    j = rng.choice(j_candidates)
                    target_pos = current_chain[j]
                    
                    # Find empty neighbors of target_pos
                    candidates = []
                    for nb in _neighbours(target_pos):
                        if nb not in pos_dict:
                            # Ensure moving i to nb doesn't break chain connectivity for i
                            # i is internal?
                            if 0 < i < n - 1:
                                prev_pos = current_chain[i-1]
                                next_pos = current_chain[i+1]
                                # nb must be adjacent to prev or next? No, 
                                # single residue move constraint: 
                                # If i is internal, new_pos must be adjacent to BOTH prev and next?
                                # Wait, standard single residue move in HP lattice:
                                # If we move residue i, its neighbors i-1 and i+1 stay fixed.
                                # So new_pos MUST be a neighbor of i-1 AND a neighbor of i+1.
                                # This is the "pinch" or "diamond" move constraint.
                                
                                # Let's re-evaluate the "Single Residue" logic above.
                                # In the r_move < 0.6 block, I used:
                                # "for nb in _neighbours(prev_pos): if nb in _neighbours(next_pos)..."
                                # This is correct for internal residues.
                                
                                # So for Pinch, we must also respect this constraint.
                                # The target_pos neighbor 'nb' must also be a neighbor of prev and next of i.
                                
                                if nb in _neighbours(prev_pos) and nb in _neighbours(next_pos):
                                    candidates.append(nb)
                            else:
                                # End residue: just needs to be neighbor of the single chain neighbor
                                chain_neighbor = current_chain[1] if i == 0 else current_chain[n-2]
                                if nb in _neighbours(chain_neighbor):
                                    candidates.append(nb)
                    
                    if candidates:
                        new_pos = rng.choice(candidates)
                        old_pos = current_chain[i]
                        
                        if new_pos != old_pos:
                            delta = _energy_delta_contact(sequence, pos_dict, i, old_pos, new_pos)
                            if delta < 0 or rng.random() < math.exp(-delta / T):
                                del pos_dict[old_pos]
                                current_chain[i] = new_pos
                                pos_dict[new_pos] = i
                                current_energy += delta
                                move_accepted = True
        else:
            # Segment Move (Rare, expensive, for escaping deep local minima)
            if n > 5:
                max_seg = min(n // 3, 8)
                seg_len = rng.randint(2, max_seg)
                i = rng.randint(0, n - seg_len)
                j = i + seg_len - 1
                
                head = current_chain[:i]
                tail = current_chain[j+1:]
                
                occupied_ht = set(head) | set(tail)
                anchor = head[-1] if head else (0, 0)
                
                start_candidates = [nb for nb in _neighbours(anchor) if nb not in occupied_ht]
                if start_candidates:
                    start_pos = rng.choice(start_candidates)
                    new_segment = [start_pos]
                    new_occupied = set(occupied_ht) | {start_pos}
                    
                    success = True
                    for k in range(1, seg_len):
                        last_pos = new_segment[-1]
                        next_candidates = [nb for nb in _neighbours(last_pos) if nb not in new_occupied]
                        if not next_candidates:
                            success = False
                            break
                        next_pos = rng.choice(next_candidates)
                        new_segment.append(next_pos)
                        new_occupied.add(next_pos)
                    
                    if success and tail:
                        tail_start = tail[0]
                        if abs(new_segment[-1][0] - tail_start[0]) + abs(new_segment[-1][1] - tail_start[1]) == 1:
                            new_chain = head + new_segment + tail
                            if len(set(new_chain)) == n:
                                new_energy = _energy(sequence, new_chain)
                                delta = new_energy - current_energy
                                if delta < 0 or rng.random() < math.exp(-delta / T):
                                    current_chain = new_chain
                                    current_energy = new_energy
                                    pos_dict = {p: idx for idx, p in enumerate(current_chain)}
                                    move_accepted = True
                            else:
                                success = False
                    elif success and not tail:
                        # Segment is at the end
                        new_chain = head + new_segment
                        if len(set(new_chain)) == n:
                            new_energy = _energy(sequence, new_chain)
                            delta = new_energy - current_energy
                            if delta < 0 or rng.random() < math.exp(-delta / T):
                                current_chain = new_chain
                                current_energy = new_energy
                                pos_dict = {p: idx for idx, p in enumerate(current_chain)}
                                move_accepted = True

        if move_accepted:
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                no_improve_count = 0
                last_improve_time = time.time()
            else:
                no_improve_count += 1
                
            # If stuck for a long time, reheat
            if no_improve_count > 200:
                T = max(T * 1.5, 1.0)
                no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 500:
                T = max(T * 1.2, 0.5)
                no_improve_count = 0

    return best_chain, best_energy

def fold(sequence, dimension):
    """Fold a sequence into a low-energy self-avoiding conformation."""
    n = len(sequence)
    if n == 0:
        return []
    
    # Determine time budget for this sequence
    # We want to ensure we finish before _GLOBAL_DEADLINE.
    # However, fold is called sequentially. We don't know how many sequences are left.
    # Strategy: Allocate a fixed chunk, but if we are fast, we can't give it back.
    # Better Strategy: Use a global "remaining time" concept if we could track it.
    # Since we can't easily pass state between calls in a sandboxed subprocess without globals,
    # and the prompt says "fold is called once per benchmark sequence",
    # we assume the interpreter stays alive or we just use a generous per-call budget 
    # that sums to < 120s.
    
    # 10 sequences. 120s / 10 = 12s.
    # Let's use 10s per sequence to be safe.
    # BUT, the inspiration suggests that dynamic allocation is better.
    # If we use a global deadline, we can check time.time() at the start.
    # If this is the first call, we have ~115s.
    # If this is the last call, we have ~10s.
    
    # Heuristic: 
    # We don't know the index. 
    # However, standard benchmarks are often sorted by length or fixed order.
    # Let's stick to a fixed budget per call to avoid complexity, but make it slightly larger 
    # for longer sequences if we can detect it? No, keep it simple.
    
    # Actually, looking at the inspiration code, it used a global deadline.
    # If the process is persistent, `time.time()` works.
    # Let's assume the worst case: 12s per sequence is the hard limit to stay under 120s total.
    # We will use 11s to allow for overhead.
    
    deadline = time.time() + 11.0
    
    # Also respect the global deadline if we are in a long-running process
    if time.time() + 11.0 > _GLOBAL_DEADLINE:
        deadline = _GLOBAL_DEADLINE - 1.0 # 1s safety margin
    
    if deadline <= time.time():
        # No time left, return a trivial solution
        return [[i, 0] for i in range(n)]
        
    rng = random.Random(12345 + n)
    
    best_chain = None
    best_energy = 0
    
    # Number of restarts depends on length and time available
    # Short sequences: more restarts. Long sequences: fewer, longer runs.
    time_available = deadline - time.time()
    
    if n < 30:
        n_restarts = 10
    elif n < 60:
        n_restarts = 5
    else:
        n_restarts = 3
        
    per_restart = time_available / max(1, n_restarts)
    
    for r in range(n_restarts):
        if time.time() >= deadline:
            break
            
        restart_deadline = time.time() + per_restart
        
        # Generate initial conformation
        # Mix of greedy and random
        if r % 2 == 0:
            # Greedy with some temperature
            init_chain, init_energy = _greedy_grow(sequence, rng, temperature=0.5)
            if init_chain is None:
                init_chain = _random_walk(sequence, rng)
                if init_chain is None:
                    init_chain = [[i, 0] for i in range(n)]
                    init_energy = 0
                else:
                    init_energy = _energy(sequence, init_chain)
        else:
            init_chain = _random_walk(sequence, rng)
            if init_chain is None:
                init_chain, init_energy = _greedy_grow(sequence, rng, temperature=0.0)
                if init_chain is None:
                    init_chain = [[i, 0] for i in range(n)]
                    init_energy = 0
            else:
                init_energy = _energy(sequence, init_chain)
        
        if init_chain is None:
            continue
            
        # Run local search
        # Leave 0.05s margin
        ls_deadline = restart_deadline - 0.05
        ls_chain, ls_energy = _local_search(sequence, init_chain, ls_deadline, rng)
        
        if ls_energy < best_energy:
            best_energy = ls_energy
            best_chain = ls_chain
            
    if best_chain is None:
        # Fallback
        best_chain = [[i, 0] for i in range(n)]
        
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
