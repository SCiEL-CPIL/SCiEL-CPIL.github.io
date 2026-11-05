"""HP lattice protein folding using greedy growth, local search, and polishing."""
# EVOLVE-BLOCK-START
import random
import time


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


def _energy_delta(sequence, pos_dict, i, old_pos, new_pos):
    """
    Compute the change in energy when moving residue i from old_pos to new_pos.
    pos_dict maps position -> index.
    Returns the delta (new_energy - old_energy).
    """
    if sequence[i] != 'H':
        return 0
    
    delta = 0
    
    # Remove contacts with old_pos
    # We check neighbors of old_pos. If a neighbor is an H residue j not adjacent in chain,
    # we are losing a contact.
    for q in _neighbours(old_pos):
        j = pos_dict.get(q)
        if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
            delta += 1  # Losing a contact increases energy by 1 (e.g., -5 -> -4 is +1)
            
    # Add contacts with new_pos
    # We check neighbors of new_pos. If a neighbor is an H residue j not adjacent in chain,
    # we are gaining a contact.
    for q in _neighbours(new_pos):
        j = pos_dict.get(q)
        if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
            delta -= 1  # Gaining a contact decreases energy by 1 (e.g., -5 -> -6 is -1)
            
    return delta


def _greedy_grow(sequence, rng):
    """Fast randomized greedy growth to get an initial conformation."""
    n = len(sequence)
    if n == 0:
        return []
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
            # Prefer sites with more open neighbors to avoid trapping
            open_count = sum(1 for q in _neighbours(site) if q not in occupied)
            # Score: gain is dominant, but we add randomness and openness
            score = gain * 10.0 + open_count * 0.1 + rng.random()
            candidates.append((score, site))
        
        if not candidates:
            return None
        
        # Pick the best candidate
        candidates.sort(key=lambda x: -x[0])
        best_site = candidates[0][1]
        chain.append(best_site)
        occupied[best_site] = i
    
    return chain


def _polish(sequence, chain, deadline):
    """
    Steepest descent polish: repeatedly scan all residues and apply the best
    improving move if any exists. Stops when no improvement is found or deadline reached.
    """
    n = len(chain)
    if n < 3:
        return list(chain), _energy(sequence, chain)
    
    current_chain = list(chain)
    pos_dict = {p: i for i, p in enumerate(current_chain)}
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Iterate over residues in a random order to break symmetry? 
        # Or sequential? Sequential is faster but might be biased. 
        # Let's use sequential for speed in polish, as it's a refinement step.
        for i in range(1, n - 1):
            if time.time() >= deadline:
                break
            old_pos = current_chain[i]
            prev_pos = current_chain[i - 1]
            next_pos = current_chain[i + 1]
            
            # Find valid new positions:
            # Must be adjacent to prev_pos AND adjacent to next_pos.
            # Must not be occupied by any other residue.
            # Note: old_pos is occupied by i, so we must exclude it from 'occupied' check 
            # or handle it carefully.
            
            # Valid positions are intersection of neighbors(prev) and neighbors(next)
            # excluding old_pos itself (since we are moving away) and any other occupied site.
            
            valid_moves = []
            for nb in _neighbours(prev_pos):
                if nb == old_pos:
                    continue
                # Check if nb is adjacent to next_pos
                if abs(nb[0] - next_pos[0]) + abs(nb[1] - next_pos[1]) != 1:
                    continue
                # Check if nb is occupied by someone else
                if nb in pos_dict:
                    if pos_dict[nb] != i:
                        continue
                # It's a valid move
                valid_moves.append(nb)
            
            if not valid_moves:
                continue
                
            # Find the best move (lowest energy delta)
            best_delta = 0
            best_move = None
            for new_pos in valid_moves:
                # Temporarily remove i from old_pos to calculate delta correctly?
                # _energy_delta assumes pos_dict has i at old_pos.
                # If we move i to new_pos, we remove i from old_pos and add to new_pos.
                # The delta function handles the "remove from old" and "add to new" logic
                # based on the current pos_dict.
                # However, if new_pos is occupied by someone else, we already filtered it out.
                # So pos_dict[new_pos] is None (or i if it was old_pos, but we excluded old_pos).
                
                delta = _energy_delta(sequence, pos_dict, i, old_pos, new_pos)
                if delta < best_delta:
                    best_delta = delta
                    best_move = new_pos
            
            if best_move is not None and best_delta < 0:
                # Apply the move
                del pos_dict[old_pos]
                current_chain[i] = best_move
                pos_dict[best_move] = i
                current_energy += best_delta
                improved = True
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
    
    return best_chain, best_energy


def _local_search(sequence, chain, deadline, rng):
    """
    Randomized local search with single-residue moves.
    """
    n = len(chain)
    if n < 4:
        return list(chain), _energy(sequence, chain)
    
    current_chain = list(chain)
    pos_dict = {p: i for i, p in enumerate(current_chain)}
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    # Simple simulated annealing or hill climbing with random restarts of search?
    # Let's do hill climbing with random move selection.
    # If stuck, we can restart the local search from the best found so far?
    # Or just keep going.
    
    is_h = [c == 'H' for c in sequence]
    
    while time.time() < deadline:
        # Pick a random interior residue
        i = rng.randint(1, n - 2)
        old_pos = current_chain[i]
        prev_pos = current_chain[i - 1]
        next_pos = current_chain[i + 1]
        
        # Find valid moves
        valid_moves = []
        for nb in _neighbours(prev_pos):
            if nb == old_pos:
                continue
            if abs(nb[0] - next_pos[0]) + abs(nb[1] - next_pos[1]) != 1:
                continue
            if nb in pos_dict and pos_dict[nb] != i:
                continue
            valid_moves.append(nb)
            
        if not valid_moves:
            continue
            
        # Pick a random valid move to try
        new_pos = rng.choice(valid_moves)
        
        delta = _energy_delta(sequence, pos_dict, i, old_pos, new_pos)
        
        # Accept if improving, or with small probability if neutral (to escape plateaus)
        if delta < 0 or (delta == 0 and rng.random() < 0.1):
            # Apply move
            del pos_dict[old_pos]
            current_chain[i] = new_pos
            pos_dict[new_pos] = i
            current_energy += delta
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        # Else reject, state remains unchanged

    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation via multiple greedy growths followed by local search and polish."""
    n = len(sequence)
    # 120s budget. Use 110s for safety.
    deadline = time.time() + 110
    
    rng = random.Random(12345 + len(sequence))
    best_chain = None
    best_energy = 0
    
    # Number of restarts depends on sequence length and time.
    # Short sequences: many restarts. Long sequences: fewer, deeper searches.
    if n <= 20:
        n_restarts = 50
        local_search_time_frac = 0.5
    elif n <= 40:
        n_restarts = 30
        local_search_time_frac = 0.5
    elif n <= 60:
        n_restarts = 15
        local_search_time_frac = 0.5
    else:
        n_restarts = 5
        local_search_time_frac = 0.5

    # We don't know exactly how much time each restart takes, so we just loop until deadline.
    # But we can estimate.
    # Let's just run restarts until time runs out.
    
    while time.time() < deadline:
        # 1. Generate initial conformation
        chain = _greedy_grow(sequence, rng)
        if chain is None:
            # If greedy fails, try a random walk? Or just skip.
            # For robustness, let's try a simple random walk if greedy fails.
            chain = []
            if n > 0:
                chain.append((0,0))
                occupied = {(0,0)}
                dirs = [(1,0), (-1,0), (0,1), (0,-1)]
                for i in range(1, n):
                    last = chain[-1]
                    cands = [ (last[0]+dx, last[1]+dy) for dx,dy in dirs if (last[0]+dx, last[1]+dy) not in occupied]
                    if not cands:
                        chain = None
                        break
                    nxt = rng.choice(cands)
                    chain.append(nxt)
                    occupied.add(nxt)
            if chain is None:
                continue

        # 2. Local Search
        # Allocate some time for local search. 
        # Since we don't know exact time, let's just run local search with a sub-deadline?
        # Or just run it and check time in the loop.
        # _local_search has its own deadline check.
        # Let's give it the remaining time? No, that would prevent restarts.
        # Let's give it a fraction of the remaining time per restart.
        remaining_time = deadline - time.time()
        # If we have little time left, don't start a new restart.
        if remaining_time < 1.0:
            break
        
        # Heuristic: give local search 50% of remaining time? 
        # That's risky if it takes long. 
        # Better: run local search for a fixed small amount or until a certain number of steps?
        # The _local_search function takes a deadline. Let's pass a deadline that is 
        # current_time + (remaining_time * 0.5) ? 
        # If we do that, we effectively halve the time for each subsequent restart.
        # This ensures we get multiple restarts.
        
        ls_deadline = time.time() + remaining_time * 0.4
        
        refined_chain, refined_energy = _local_search(sequence, chain, ls_deadline, rng)
        
        # 3. Polish
        # Polish is usually fast. Give it the rest of the allocated time for this restart?
        # Or just a small fixed time?
        # Let's give it until the original deadline minus a small buffer, 
        # but capped so we don't consume all time for one restart if it's slow.
        # Actually, _polish is steepest descent, it stops when no improvement.
        # It should be fast.
        polish_deadline = time.time() + (deadline - time.time()) * 0.5
        polished_chain, polished_energy = _polish(sequence, refined_chain, polish_deadline)
        
        if polished_energy < best_energy:
            best_energy = polished_energy
            best_chain = list(polished_chain)
            
        # Check if we should stop
        if time.time() > deadline - 5: # 5s buffer for final return
            break

    if best_chain is None:
        # Fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
