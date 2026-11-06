"""
HP lattice protein folding using optimized Monte Carlo search.
Implements Kink moves and Tail Regeneration (Pivot) moves with Simulated Annealing.
Uses incremental energy updates for speed.
"""
# EVOLVE-BLOCK-START
import random
import time
import math

_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    _init_time()
    # Safety margin: 210s limit, leave 10s buffer -> 200s usable
    return 200.0 - (time.time() - _START_TIME)

def _neighbours(point):
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _energy(sequence, chain):
    """Calculate energy: -number of H-H contacts (non-bonded neighbors)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(sequence)
    is_h = [c == 'H' for c in sequence]
    for i, p in enumerate(chain):
        if is_h[i]:
            # Check 4 neighbors
            x, y = p
            # Right
            q = (x + 1, y)
            j = where.get(q)
            if j is not None and j > i + 1 and is_h[j]:
                contacts += 1
            # Left
            q = (x - 1, y)
            j = where.get(q)
            if j is not None and j > i + 1 and is_h[j]:
                contacts += 1
            # Up
            q = (x, y + 1)
            j = where.get(q)
            if j is not None and j > i + 1 and is_h[j]:
                contacts += 1
            # Down
            q = (x, y - 1)
            j = where.get(q)
            if j is not None and j > i + 1 and is_h[j]:
                contacts += 1
    return -contacts

def _generate_random_saw(n, rng, max_attempts=50):
    """Generate a random self-avoiding walk of length n on the square lattice."""
    for _ in range(max_attempts):
        chain = [(0, 0)]
        occupied = {(0, 0): 0}
        success = True
        for i in range(1, n):
            neighbors = list(_neighbours(chain[-1]))
            rng.shuffle(neighbors)
            placed = False
            for site in neighbors:
                if site not in occupied:
                    chain.append(site)
                    occupied[site] = i
                    placed = True
                    break
            if not placed:
                success = False
                break
        if success and len(chain) == n:
            return chain
    return None

def _get_kink_moves(chain, i, occupied_set):
    """
    Yield valid new positions (new_p_curr, new_p_next) for a kink at i.
    i is the index of the first residue to move.
    occupied_set contains all positions in the chain.
    """
    p_prev = chain[i - 1]
    p_curr = chain[i]
    p_next = chain[i + 1]
    p_after = chain[i + 2]
    
    # Get all neighbors of p_prev
    new_p_curr_candidates = list(_neighbours(p_prev))
    
    for new_p_curr in new_p_curr_candidates:
        # Must not be p_after (would make length 0 or overlap)
        if new_p_curr == p_after:
            continue
        # Must not be p_curr (no move)
        if new_p_curr == p_curr:
            continue
        # Check if new_p_curr is free (excluding p_curr and p_next which are being moved)
        # We check against occupied_set. If it's p_curr or p_next, it's in the set, but we allow it.
        # However, new_p_curr cannot be p_curr (checked). Can it be p_next?
        # If new_p_curr == p_next, then new_p_next must be neighbor of p_next and p_after.
        # If new_p_curr == p_next, the chain segment is p_prev - p_next - new_p_next - p_after.
        # This is a valid kink shape.
        
        # We need to ensure new_p_curr is not occupied by *other* parts of the chain.
        if new_p_curr in occupied_set and new_p_curr != p_next:
             continue
             
        # Get neighbors of new_p_curr
        new_p_next_candidates = list(_neighbours(new_p_curr))
        
        for new_p_next in new_p_next_candidates:
            # Must be adjacent to p_after
            if abs(new_p_next[0] - p_after[0]) + abs(new_p_next[1] - p_after[1]) != 1:
                continue
            # Must not be p_curr
            if new_p_next == p_curr:
                continue
            # Must not be p_next (no move) -- wait, if new_p_curr == p_next, then new_p_next != p_next is required?
            # If new_p_curr == p_next, then the residue at i moves to p_next's old spot.
            # The residue at i+1 moves to new_p_next.
            # So new_p_next cannot be p_next (the old spot of i+1) unless it's the same, which is no move.
            if new_p_next == p_next:
                continue
                
            # Check if new_p_next is free (excluding p_curr and p_next)
            if new_p_next in occupied_set and new_p_next != p_curr:
                 continue
                 
            # Valid move
            yield (new_p_curr, new_p_next)

def _tail_regeneration(chain, rng, occupied_set, max_steps=100):
    """
    Attempt a tail regeneration move (pivot at i).
    Regenerates chain[i+1:] as a random SAW avoiding chain[:i+1].
    """
    n = len(chain)
    if n < 5:
        return None
    
    # Choose pivot index i
    # Bias towards earlier pivots? Random is fine.
    i = rng.randint(0, n - 5)
    
    pivot_pos = chain[i]
    
    # The fixed head is chain[:i+1]. 
    # We need to avoid all positions in chain[:i+1] and also the new tail must avoid itself.
    # We can use a temporary set for the tail generation, starting with the fixed head positions.
    # But we don't want to copy the whole occupied_set (O(N)).
    # However, N is small (<=100). Copying a set of 100 tuples is fast.
    fixed_head_positions = set(chain[:i + 1])
    
    remaining_len = n - i - 1
    if remaining_len == 0:
        return None
        
    # Try a few random paths for the tail
    for _ in range(max_steps):
        # Shuffle neighbors of pivot
        neighbors = list(_neighbours(pivot_pos))
        rng.shuffle(neighbors)
        
        valid_path = True
        path = []
        occupied = set(fixed_head_positions)
        
        for step in range(remaining_len):
            # Get neighbors of current
            if step == 0:
                opts = neighbors
            else:
                opts = list(_neighbours(path[-1]))
                rng.shuffle(opts)
            
            placed = False
            for nb in opts:
                if nb in occupied:
                    continue
                path.append(nb)
                occupied.add(nb)
                placed = True
                break
            
            if not placed:
                valid_path = False
                break
        
        if valid_path and len(path) == remaining_len:
            new_chain = chain[:i + 1] + path
            # Check if it's actually different
            if new_chain != chain:
                return new_chain
            
    return None

def _local_search(sequence, chain, rng, max_iters, start_temp=5.0, end_temp=0.1):
    """
    Simulated Annealing local search with incremental energy updates.
    """
    n = len(chain)
    current = list(chain)
    
    # Precompute is_h
    is_h = [c == 'H' for c in sequence]
    
    # Maintain occupied set for fast collision checks
    occupied = {p: i for i, p in enumerate(current)}
    
    # Maintain contact set for fast energy updates
    contacts = set()
    for i, p in enumerate(current):
        if is_h[i]:
            for q in _neighbours(p):
                j = occupied.get(q)
                if j is not None and j > i + 1 and is_h[j]:
                    contacts.add((i, j))
    
    current_energy = -len(contacts)
    
    for it in range(max_iters):
        if _time_remaining() < 5:
            break
        
        # Annealing schedule
        progress = it / max(1, max_iters)
        temp = start_temp * (end_temp / start_temp) ** progress
        
        # Choose move type
        # Kinks are cheap, Tail Regen is expensive but powerful
        if rng.random() < 0.8:
            # Kink Move
            if n < 4:
                continue
            i = rng.randint(1, n - 3)
            
            # Get valid moves
            valid_moves = list(_get_kink_moves(current, i, occupied))
            if not valid_moves:
                continue
            
            new_p_curr, new_p_next = rng.choice(valid_moves)
            
            # Save old state for potential revert
            old_p_curr = current[i]
            old_p_next = current[i + 1]
            old_contacts = contacts.copy()
            
            # Apply move
            current[i] = new_p_curr
            current[i + 1] = new_p_next
            
            # Update occupied
            occupied.pop(old_p_curr)
            occupied.pop(old_p_next)
            occupied[new_p_curr] = i
            occupied[new_p_next] = i + 1
            
            # Update Energy Incrementally
            new_contacts = set()
            for c in old_contacts:
                if c[0] != i and c[1] != i and c[0] != i+1 and c[1] != i+1:
                    new_contacts.add(c)
            
            # Add new contacts involving i or i+1
            if is_h[i]:
                for q in _neighbours(current[i]):
                    j = occupied.get(q)
                    if j is not None and j > i + 1 and is_h[j]:
                        new_contacts.add((i, j))
            if is_h[i+1]:
                for q in _neighbours(current[i+1]):
                    j = occupied.get(q)
                    if j is not None and j > i + 2 and is_h[j]:
                        new_contacts.add((i+1, j))
            
            contacts = new_contacts
            new_energy = -len(contacts)
            
            # Decide accept/reject
            delta = new_energy - current_energy
            if delta < 0 or (temp > 0.01 and rng.random() < math.exp(-delta / temp)):
                current_energy = new_energy
            else:
                # Revert
                current[i] = old_p_curr
                current[i + 1] = old_p_next
                occupied.pop(new_p_curr)
                occupied.pop(new_p_next)
                occupied[old_p_curr] = i
                occupied[old_p_next] = i + 1
                contacts = old_contacts
                
        else:
            # Tail Regeneration
            new_chain = _tail_regeneration(current, rng, occupied)
            if new_chain is None:
                continue
            
            # Calculate new energy for tail regen
            new_occupied = {p: i for i, p in enumerate(new_chain)}
            new_contacts = set()
            for i, p in enumerate(new_chain):
                if is_h[i]:
                    for q in _neighbours(p):
                        j = new_occupied.get(q)
                        if j is not None and j > i + 1 and is_h[j]:
                            new_contacts.add((i, j))
            
            new_energy = -len(new_contacts)
            
            # Decide accept/reject
            delta = new_energy - current_energy
            if delta < 0 or (temp > 0.01 and rng.random() < math.exp(-delta / temp)):
                current = new_chain
                occupied = new_occupied
                contacts = new_contacts
                current_energy = new_energy
            # If rejected, do nothing (keep old state)

    return current

def fold(sequence, dimension):
    """
    Fold the sequence using fragment-based Monte Carlo search with
    random SAW initialization and segment moves.
    """
    _init_time()
    n = len(sequence)
    
    # Determine search parameters based on sequence length
    # With O(1) energy updates, we can do more iterations
    if n <= 20:
        num_restarts = 50
        iters_per_restart = 5000
    elif n <= 50:
        num_restarts = 30
        iters_per_restart = 3000
    else:
        num_restarts = 20
        iters_per_restart = 2000
    
    best_chain = None
    best_energy = 0  # 0 is neutral, we want negative (lower is better)
    
    for restart in range(num_restarts):
        if _time_remaining() < 10:
            break
        
        rng = random.Random(12345 + len(sequence) + restart * 7919)
        
        # Initialize with random SAW
        chain = _generate_random_saw(n, rng)
        if chain is None:
            # Fallback: simple greedy growth
            chain = [(0, 0)]
            occupied = {(0, 0): 0}
            for i in range(1, n):
                neighbors = list(_neighbours(chain[-1]))
                rng.shuffle(neighbors)
                placed = False
                for site in neighbors:
                    if site not in occupied:
                        chain.append(site)
                        occupied[site] = i
                        placed = True
                        break
                if not placed:
                    break
            if len(chain) < n:
                continue
        
        # Refine with local search
        refined_chain = _local_search(sequence, chain, rng, max_iters=iters_per_restart)
        
        energy = _energy(sequence, refined_chain)
        if best_chain is None or energy < best_energy:
            best_chain = refined_chain
            best_energy = energy
    
    if best_chain is None:
        # Final fallback: straight line
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
