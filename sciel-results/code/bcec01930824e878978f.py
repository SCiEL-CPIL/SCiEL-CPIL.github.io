"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import heapq

_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    _init_time()
    # Budget is 120s CPU for all sequences, 210s wall-clock kill.
    # Use 190s as a safe deadline to ensure we finish before kill.
    return 190.0 - (time.time() - _START_TIME)

def _neighbours(point):
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

def _energy(sequence, chain):
    """Compute HP contact energy: -1 for each H-H pair on neighbouring sites not adjacent in chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _grow_beam(sequence, dimension, rng, beam_width=15):
    """
    Beam search growth.
    Keeps the top `beam_width` partial chains based on a heuristic score.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    # Each item in beam is (negative_score, chain, occupied_dict)
    # We use a min-heap, so we negate the score to get max-heap behavior.
    # Score = Energy (negative is good) + Heuristic (free space is good)
    # We want to maximize score. So we store -score in heap.
    
    start = (0, 0)
    initial_chain = [start]
    initial_occupied = {start: 0}
    
    # Initial score: 0 energy + heuristic
    # Heuristic: count free neighbours of the last residue
    free = sum(1 for q in _neighbours(start) if q not in initial_occupied)
    initial_score = 0 + free * 0.1
    beam = [(-initial_score, initial_chain, initial_occupied)]
    
    for i in range(1, n):
        new_beam = []
        # Expand all states in the current beam
        for neg_score, chain, occupied in beam:
            current = chain[-1]
            for site in _neighbours(current):
                if site in occupied:
                    continue
                
                # Calculate gain in energy
                gain = 0
                if sequence[i] == 'H':
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            gain += 1
                
                # Calculate free neighbours for heuristic
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                
                # New score: Previous score + gain + free heuristic
                # Previous score was -neg_score
                prev_score = -neg_score
                new_score = prev_score + gain + free * 0.1
                
                # Add to new beam candidate
                new_chain = chain + [site]
                new_occupied = dict(occupied)
                new_occupied[site] = i
                
                new_beam.append((-new_score, new_chain, new_occupied))
        
        if not new_beam:
            return None
            
        # Keep only the top beam_width candidates
        # Sort by -new_score (i.e., score descending)
        # heapq.nsmallest on negative scores gives us the largest scores
        new_beam = heapq.nsmallest(beam_width, new_beam)
        beam = new_beam
        
    if not beam:
        return None
        
    # Return the best chain from the beam
    # The first item in the heap (smallest negative score) is the best
    best_neg_score, best_chain, _ = beam[0]
    return best_chain

def _local_search(sequence, chain, rng, max_steps=500):
    """
    Local search using single-residue moves and segment shifts.
    """
    n = len(chain)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    
    for step in range(max_steps):
        if _time_remaining() < 2.0:
            break
            
        improved = False
        
        # 1. Try single residue moves (Steepest Descent style)
        # Pick a random residue
        i = rng.randint(1, n - 2)
        prev = chain[i-1]
        curr = chain[i]
        next_ = chain[i+1]
        
        # Find valid positions: neighbors of prev AND next_
        # Note: In a lattice, a single residue move is only valid if the new pos
        # is adjacent to both prev and next_. This is a very restrictive move.
        # However, it's fast.
        candidates = set(_neighbours(prev)) & set(_neighbours(next_))
        candidates.discard(curr)
        
        best_move_pos = None
        best_move_energy = current_energy
        
        for new_pos in candidates:
            if new_pos in occupied:
                continue
            
            # Temporarily move
            del occupied[curr]
            occupied[new_pos] = i
            chain[i] = new_pos
            
            new_energy = _energy(sequence, chain)
            
            if new_energy < best_move_energy:
                best_move_energy = new_energy
                best_move_pos = new_pos
            else:
                # Revert if not better than current best in this iteration
                # We will revert fully after loop if no improvement
                pass
                
        if best_move_pos is not None:
            # Apply the best move found
            # Note: chain[i] might be curr or some other candidate.
            # We need to ensure we are at the state of best_move_pos.
            # Since we only kept track of the best, let's just re-apply the best one cleanly.
            # Actually, the loop above modified chain[i] multiple times.
            # Let's restart this logic more safely.
            pass
            
        # Safer Single Residue Move Implementation
        i = rng.randint(1, n - 2)
        prev = chain[i-1]
        curr = chain[i]
        next_ = chain[i+1]
        
        candidates = set(_neighbours(prev)) & set(_neighbours(next_))
        candidates.discard(curr)
        
        for new_pos in candidates:
            if new_pos in occupied:
                continue
            
            del occupied[curr]
            occupied[new_pos] = i
            chain[i] = new_pos
            
            new_energy = _energy(sequence, chain)
            
            if new_energy < current_energy:
                current_energy = new_energy
                improved = True
            else:
                # Revert
                del occupied[new_pos]
                occupied[curr] = i
                chain[i] = curr
                break # Try next random residue if this one didn't improve
        
        # 2. Try Segment Shift (every 50 steps)
        if step % 50 == 0 and not improved:
            # Pick a random segment
            seg_len = rng.randint(2, min(5, n//2))
            start_idx = rng.randint(0, n - seg_len)
            end_idx = start_idx + seg_len - 1
            
            # Try to shift the segment by (dx, dy)
            directions = [(1,0), (-1,0), (0,1), (0,-1)]
            rng.shuffle(directions)
            
            for dx, dy in directions:
                # Check if all new positions are valid
                new_positions = []
                valid = True
                for k in range(start_idx, end_idx + 1):
                    nx = chain[k][0] + dx
                    ny = chain[k][1] + dy
                    np = (nx, ny)
                    new_positions.append(np)
                    
                    # Check if occupied by non-segment parts
                    if np in occupied:
                        occ_idx = occupied[np]
                        if not (start_idx <= occ_idx <= end_idx):
                            valid = False
                            break
                            
                if not valid:
                    continue
                    
                # Check connectivity
                # Left connection
                if start_idx > 0:
                    left_anchor = chain[start_idx - 1]
                    new_start = new_positions[0]
                    if abs(left_anchor[0] - new_start[0]) + abs(left_anchor[1] - new_start[1]) != 1:
                        continue
                        
                # Right connection
                if end_idx < n - 1:
                    right_anchor = chain[end_idx + 1]
                    new_end = new_positions[-1]
                    if abs(right_anchor[0] - new_end[0]) + abs(right_anchor[1] - new_end[1]) != 1:
                        continue
                        
                # Check self-avoidance within the segment (should be fine if original was)
                # But check if new positions overlap with each other?
                # Since it's a translation, relative distances are same.
                
                # Calculate new energy
                # Temporarily apply
                old_positions = [chain[k] for k in range(start_idx, end_idx + 1)]
                
                # Remove old
                for k in range(start_idx, end_idx + 1):
                    del occupied[chain[k]]
                    
                # Add new
                for k, np in zip(range(start_idx, end_idx + 1), new_positions):
                    occupied[np] = k
                    chain[k] = np
                    
                new_energy = _energy(sequence, chain)
                
                if new_energy < current_energy:
                    current_energy = new_energy
                    improved = True
                else:
                    # Revert
                    for k in range(start_idx, end_idx + 1):
                        del occupied[chain[k]]
                        chain[k] = old_positions[k - start_idx]
                        occupied[old_positions[k - start_idx]] = k
                        
                if improved:
                    break # Try next random segment if this one improved
        
        if not improved and step > 100:
            # Allow some random acceptance to escape local minima?
            # For now, strict improvement to keep it simple and fast.
            pass

    return chain, current_energy

def fold(sequence, dimension):
    """
    Fold an HP sequence on the square lattice.
    Uses beam search growth followed by local search refinement.
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
    
    # Perform many restarts.
    # We loop until time runs out, but cap iterations to avoid infinite loops if time is mocked.
    max_iterations = 2000
    iterations = 0
    
    while _time_remaining() > 5.0 and iterations < max_iterations:
        iterations += 1
        
        # Generate a new conformation using beam search
        chain = _grow_beam(sequence, dimension, rng, beam_width=15)
        if chain is None:
            continue
        
        # Refine with local search
        # Limit local search steps to ensure we can do many restarts
        chain, energy = _local_search(sequence, chain, rng, max_steps=100)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
            
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
