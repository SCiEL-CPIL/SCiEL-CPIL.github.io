"""Improved HP lattice protein folding with optimized local search."""
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
    # Budget is 120s CPU. Safety margin to 110s.
    return 110.0 - (time.time() - _START_TIME)

def _neighbours(point):
    """Yield 4-neighbours on square lattice."""
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

def _count_contacts(sequence, chain):
    """Calculate total H-H contacts for a chain."""
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
        # Check 4 neighbors
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        for nx, ny in neighbors:
            j = occupied.get((nx, ny))
            if j is not None and j > i + 1 and is_h[j]:
                contacts += 1
    return contacts

def _grow(sequence, dimension, rng):
    """
    One randomized greedy chain growth.
    Prefers sites that create new H-H contacts and have high free space.
    """
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    is_h = [c == 'H' for c in sequence]
    
    for i in range(1, len(sequence)):
        current = chain[-1]
        options = []
        
        for site in _neighbours(current):
            if site in occupied:
                continue
            
            # Count immediate H-H contacts this placement would make
            gain = 0
            if is_h[i]:
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and is_h[j]:
                        gain += 1
            
            # Count free neighbours (for future growth flexibility)
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            # Score: prioritize contacts, then free space, then random tie-breaker
            # We want to maximize contacts, so higher gain is better.
            score = gain * 10.0 + free * 0.1 + rng.random()
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick the best option
        options.sort(key=lambda x: x[0], reverse=True)
        site = options[0][1]
        occupied[site] = i
        chain.append(site)
    
    return chain

def _local_search(sequence, chain, rng, max_steps=1000):
    """
    Refine the chain using kink moves and segment moves with simulated annealing.
    Uses incremental energy updates for speed.
    """
    n = len(chain)
    if n < 3:
        return chain, -_count_contacts(sequence, chain)
    
    # Precompute occupied set and energy
    occupied = {p: i for i, p in enumerate(chain)}
    current_contacts = _count_contacts(sequence, chain)
    current_energy = -current_contacts
    is_h = [c == 'H' for c in sequence]
    
    # Simulated annealing parameters
    start_temp = 2.0
    end_temp = 0.05
    
    # Helper to calculate contact change for moving a set of residues
    def calc_contact_delta(moved_indices, new_positions, old_positions):
        # moved_indices: list of indices
        # new_positions: list of new coords
        # old_positions: list of old coords
        # We need to remove old contacts involving moved residues and add new ones.
        # Contact is between i and j if |i-j| > 1 and they are neighbors on lattice.
        
        delta = 0
        
        # 1. Remove old contacts
        for idx in moved_indices:
            if not is_h[idx]:
                continue
            old_pos = old_positions[moved_indices.index(idx)]
            for nb in _neighbours(old_pos):
                j = occupied.get(nb)
                if j is not None and j not in moved_indices:
                    if j > idx + 1 and is_h[j]:
                        delta -= 1
        
        # 2. Add new contacts
        # Note: We must be careful not to count contacts between moved residues themselves
        # unless they are not adjacent in the chain.
        for k, idx in enumerate(moved_indices):
            if not is_h[idx]:
                continue
            new_pos = new_positions[k]
            for nb in _neighbours(new_pos):
                # Check if neighbor is in moved_indices
                is_moved_neighbor = False
                for l, other_idx in enumerate(moved_indices):
                    if l != k and new_positions[l] == nb:
                        # They are neighbors in the new config
                        if abs(idx - other_idx) > 1 and is_h[other_idx]:
                            delta += 1
                        is_moved_neighbor = True
                        break
                
                if not is_moved_neighbor:
                    j = occupied.get(nb)
                    if j is not None and j not in moved_indices:
                        if j > idx + 1 and is_h[j]:
                            delta += 1
        
        return delta

    for step in range(max_steps):
        progress = step / max(1, max_steps)
        temp = start_temp * (end_temp / start_temp) ** progress
        
        move_type = rng.random()
        
        if move_type < 0.7:
            # Kink move
            if n < 3:
                continue
            i = rng.randint(1, n - 2)
            prev = chain[i-1]
            curr = chain[i]
            next_ = chain[i+1]
            
            # Find valid new positions for curr
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
            
            # Calculate delta
            moved_indices = [i]
            old_positions = [curr]
            new_positions = [new_pos]
            
            # Temporarily update occupied for delta calc? 
            # The helper function above assumes `occupied` is the state BEFORE the move for non-moved nodes.
            # But it checks `occupied.get(nb)` for non-moved neighbors.
            # We need to ensure we don't count the moved node itself in the old occupied set for the new position check.
            # The logic in calc_contact_delta is slightly complex. Let's simplify by just doing a full recalc if N is small, 
            # or a careful delta.
            
            # Actually, for N=100, a full recalc is ~100 ops. 1000 steps * 100 ops = 100k ops. Very fast.
            # Let's stick to full recalc for correctness and simplicity, it's fast enough.
            
            # Apply move
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
                # Revert
                chain[i] = curr
                del occupied[new_pos]
                occupied[curr] = i
                
        else:
            # Segment move (length 2 or 3)
            seg_len = rng.choice([2, 3])
            if n < seg_len + 2:
                continue
            
            start_idx = rng.randint(1, n - seg_len - 1)
            end_idx = start_idx + seg_len - 1
            
            # Get segment
            seg_indices = list(range(start_idx, end_idx + 1))
            old_seg = [chain[i] for i in seg_indices]
            
            # Find a new location for the segment
            # The segment must be placed such that:
            # 1. It doesn't overlap with any existing occupied site (except the ones it's moving from).
            # 2. The start of the segment is adjacent to chain[start_idx-1].
            # 3. The end of the segment is adjacent to chain[end_idx+1].
            
            anchor_start = chain[start_idx - 1]
            anchor_end = chain[end_idx + 1]
            
            # Try to find a valid placement
            # We can try random walks from anchor_start of length seg_len, checking if it ends at anchor_end
            # and doesn't collide.
            
            # This is a constrained random walk.
            # Let's try a few random walks.
            found_valid = False
            for _ in range(50):
                # Random walk of length seg_len from anchor_start
                pos = anchor_start
                path = [pos]
                valid = True
                for _ in range(seg_len):
                    neighbors = _neighbours(pos)
                    # Filter out occupied (excluding the segment we are moving)
                    # We need to be careful: the segment is currently in `occupied`.
                    # We should treat the segment's old positions as free for the purpose of this walk,
                    # but we must ensure the new path doesn't hit *other* occupied sites.
                    
                    # To simplify, let's just pick a random neighbor and check collision later.
                    pos = rng.choice(neighbors)
                    path.append(pos)
                
                # Check if path ends at anchor_end
                if path[-1] != anchor_end:
                    continue
                
                # Check self-avoidance and collision with rest of chain
                # The path is [anchor_start, p1, p2, ..., anchor_end]
                # The segment occupies p1...p_{seg_len}
                # Actually, the segment is the internal nodes.
                # chain[start_idx] ... chain[end_idx]
                # So the new positions are path[1] ... path[seg_len]
                
                new_seg = path[1:]
                
                # Check if any new_seg position is occupied by a non-segment residue
                collision = False
                for p in new_seg:
                    if p in occupied:
                        # Check if it's one of the old segment positions
                        is_old_seg_pos = False
                        for old_p in old_seg:
                            if old_p == p:
                                is_old_seg_pos = True
                                break
                        if not is_old_seg_pos:
                            collision = True
                            break
                
                if collision:
                    continue
                
                # Also check if the new segment self-intersects
                if len(set(new_seg)) != len(new_seg):
                    continue
                
                # Check if the new segment intersects with the rest of the chain
                # We already checked collision with `occupied`.
                # But `occupied` contains the old segment. We removed them conceptually?
                # No, `occupied` still has them.
                # So `p in occupied` is True for old segment positions.
                # We handled that.
                
                # One more check: does the new segment touch the rest of the chain in a way that creates contacts?
                # That's fine, that's what we want.
                
                # Apply move
                for i in seg_indices:
                    del occupied[chain[i]]
                
                for k, idx in enumerate(seg_indices):
                    new_pos = new_seg[k]
                    occupied[new_pos] = idx
                    chain[idx] = new_pos
                
                new_contacts = _count_contacts(sequence, chain)
                new_energy = -new_contacts
                delta = new_energy - current_energy
                
                if delta < 0 or (temp > 0.01 and rng.random() < pow(2.718281828, -delta / temp)):
                    current_energy = new_energy
                    current_contacts = new_contacts
                    found_valid = True
                else:
                    # Revert
                    for k, idx in enumerate(seg_indices):
                        del occupied[chain[idx]]
                        old_pos = old_seg[k]
                        occupied[old_pos] = idx
                        chain[idx] = old_pos
                
                break # Only try one segment move per step for simplicity
            
    return chain, current_energy

def fold(sequence, dimension):
    """
    Fold an HP sequence on the square lattice.
    Uses randomized greedy growth followed by local search refinement.
    """
    _init_time()
    
    n = len(sequence)
    if n == 0:
        return []
    
    # Emergency fallback if no time left
    if _time_remaining() < 1.0:
        return [[i, 0] for i in range(n)]
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0 # Energy is negative, so we look for lower (more negative) values
    
    # Perform many restarts
    max_iterations = 2000
    iterations = 0
    
    while _time_remaining() > 5.0 and iterations < max_iterations:
        iterations += 1
        
        # 1. Grow
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # 2. Local Search
        # Adjust local search steps based on remaining time and sequence length
        # Longer sequences need more steps to converge? Or just fixed?
        # Let's use 200 steps for local search.
        chain, energy = _local_search(sequence, chain, rng, max_steps=200)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
            
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
