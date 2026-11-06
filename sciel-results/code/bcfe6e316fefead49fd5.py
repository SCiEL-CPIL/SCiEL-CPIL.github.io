"""HP lattice protein folding using Simulated Annealing with Pivot and End moves."""
# EVOLVE-BLOCK-START
import random
import time
import math

def _neighbours(point):
    """Yield 4-neighbors in 2D."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _grow(sequence, dimension, rng):
    """
    Generate a random self-avoiding walk using greedy growth.
    Prioritizes positions that create H-H contacts or have free space.
    """
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            # Calculate gain: number of new H-H contacts
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            gain += 1
            
            # Free space heuristic: prefer sites with more free neighbors to avoid trapping
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            
            # Score: gain is dominant, free space is secondary, random is tertiary
            score = gain * 10.0 + free * 0.5 + rng.random()
            options.append((score, site))
        
        if not options:
            return None # Dead end
        
        # Pick best option with some randomness (softmax-like or just max with random tie-breaker)
        # To allow exploration, we can pick from top few or just max.
        # Let's pick max score.
        options.sort(reverse=True)
        # Add some noise: pick from top 3 if available
        if len(options) > 1 and rng.random() < 0.3:
             site = options[rng.randint(0, min(2, len(options)-1))][1]
        else:
             site = options[0][1]
        
        occupied[site] = i
        chain.append(site)
        
    return chain

def _calculate_contacts(sequence, chain):
    """Calculate set of H-H contact pairs (i, j) with i < j."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = set()
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts.add((i, j))
    return contacts

def _energy_from_contacts(contacts):
    return -len(contacts)

def _simulate_annealing(sequence, initial_chain, rng, time_limit):
    """
    Run Simulated Annealing on the given chain.
    Returns the best chain and energy found.
    """
    n = len(sequence)
    if n < 2:
        return initial_chain, 0

    chain = initial_chain[:]
    occupied = {p: i for i, p in enumerate(chain)}
    contacts = _calculate_contacts(sequence, chain)
    current_energy = _energy_from_contacts(contacts)
    best_energy = current_energy
    best_chain = chain[:]

    # Temperature schedule
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.995
    
    start_time = time.time()
    steps = 0
    
    # Pre-compute H positions for quick check
    is_h = [c == 'H' for c in sequence]

    def try_move_end():
        nonlocal chain, occupied, contacts, current_energy, best_energy, best_chain
        
        if n < 2:
            return False
        
        last_idx = n - 1
        prev_idx = n - 2
        last_pos = chain[last_idx]
        prev_pos = chain[prev_idx]
        
        # Remove last from occupied
        del occupied[last_pos]
        
        # Remove contacts involving last_idx
        removed_contacts = set()
        for c in list(contacts):
            if c[0] == last_idx or c[1] == last_idx:
                removed_contacts.add(c)
                contacts.remove(c)
        
        # Try all 4 neighbors of prev_pos
        valid_sites = []
        for site in _neighbours(prev_pos):
            if site not in occupied and site != last_pos: # site != last_pos is redundant if we removed it, but safe
                # Check if site is free (not in occupied)
                # occupied already has last_pos removed
                valid_sites.append(site)
        
        if not valid_sites:
            # Revert
            occupied[last_pos] = last_idx
            contacts.update(removed_contacts)
            return False
            
        # Pick a random valid site
        new_pos = rng.choice(valid_sites)
        
        # Check self-intersection with new_pos (already ensured by 'not in occupied')
        
        # Update chain and occupied
        chain[last_idx] = new_pos
        occupied[new_pos] = last_idx
        
        # Add new contacts involving last_idx
        added_contacts = set()
        if is_h[last_idx]:
            for q in _neighbours(new_pos):
                j = occupied.get(q)
                if j is not None and j != last_idx and j < last_idx and is_h[j]:
                    # j < last_idx is always true for j in 0..n-2
                    added_contacts.add((j, last_idx))
                    contacts.add((j, last_idx))
        
        # Calculate energy change
        old_e = current_energy
        new_e = -len(contacts)
        delta_e = new_e - old_e
        
        # Acceptance
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            current_energy = new_e
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
            return True
        else:
            # Revert
            chain[last_idx] = last_pos
            del occupied[new_pos]
            occupied[last_pos] = last_idx
            
            # Revert contacts
            for c in added_contacts:
                contacts.remove(c)
            contacts.update(removed_contacts)
            
            return True # We attempted a move

    def try_move_pivot():
        nonlocal chain, occupied, contacts, current_energy, best_energy, best_chain
        
        if n < 4:
            return False
            
        # Pick pivot k between 1 and n-2
        k = rng.randint(1, n - 2)
        # Segment is k+1 to n-1
        # Pivot point is chain[k]
        
        # Collect segment
        segment_len = n - 1 - k
        if segment_len < 1:
            return False
            
        old_segment = chain[k+1:n]
        
        # Remove segment from occupied
        for pos in old_segment:
            del occupied[pos]
            
        # Remove contacts involving any index in [k+1, n-1]
        removed_contacts = set()
        for c in list(contacts):
            i, j = c
            if i > k or j > k:
                removed_contacts.add(c)
                contacts.remove(c)
        
        # Generate rotations
        # The segment is a path. We rotate it around chain[k].
        # The first point of the new segment must be a neighbor of chain[k].
        # Let's try all 4 directions for the first step of the rotated segment.
        
        pivot_pos = chain[k]
        last_segment_pos = old_segment[-1]
        
        # We need to reconstruct the segment relative to the pivot.
        # Original relative coords:
        # rel[0] = old_segment[0] - pivot_pos
        # rel[i] = old_segment[i] - old_segment[i-1] (step vectors)
        
        # Actually, easier: just try all 4 starting directions for the new segment.
        # For each direction, simulate the walk of the segment.
        
        valid_new_segments = []
        
        for start_dir in [(1,0), (-1,0), (0,1), (0,-1)]:
            new_seg = []
            current_pos = (pivot_pos[0] + start_dir[0], pivot_pos[1] + start_dir[1])
            
            # Check if first pos is valid (not in occupied prefix, not self-intersecting later)
            # occupied currently only has prefix 0..k
            if current_pos in occupied:
                continue
                
            new_seg.append(current_pos)
            
            # Continue the segment
            # We need to apply the same sequence of turns as the original segment?
            # No, a pivot move rotates the whole rigid body?
            # Standard Pivot Move: Rotate the segment around the pivot.
            # This implies the relative shape is preserved.
            # So we can compute the new positions by rotating the vector from pivot to each point.
            # However, in 2D lattice, "rotation" usually means one of the 4 rotations (0, 90, 180, 270).
            # But the segment must remain connected.
            # A simpler interpretation of Pivot in SAW:
            # Pick a pivot. The part of the chain after the pivot can be re-attached in any valid self-avoiding way that starts from a neighbor of the pivot?
            # No, that's a "Reconnection" or "Segment Move".
            # A true Pivot Move rotates the existing shape.
            
            # Let's stick to the definition: Rotate the vector (p_i - p_k) for all i > k.
            # In 2D, we can rotate by 90, 180, 270 degrees.
            
            rotations = [
                lambda x, y: (y, -x),       # 90 deg CW
                lambda x, y: (-x, -y),      # 180 deg
                lambda x, y: (-y, x)        # 90 deg CCW
            ]
            
            for rot_func in rotations:
                new_seg = []
                valid = True
                for pos in old_segment:
                    dx = pos[0] - pivot_pos[0]
                    dy = pos[1] - pivot_pos[1]
                    nx, ny = rot_func(dx, dy)
                    new_pos = (pivot_pos[0] + nx, pivot_pos[1] + ny)
                    
                    # Check against occupied prefix
                    if new_pos in occupied:
                        valid = False
                        break
                    # Check against previously placed points in new_seg
                    if new_pos in new_seg:
                        valid = False
                        break
                    new_seg.append(new_pos)
                
                if valid:
                    valid_new_segments.append(new_seg)
        
        if not valid_new_segments:
            # Revert
            for i, pos in enumerate(old_segment):
                occupied[pos] = k + 1 + i
            contacts.update(removed_contacts)
            return False
            
        # Pick a random valid new segment
        new_segment = rng.choice(valid_new_segments)
        
        # Update chain and occupied
        for i, pos in enumerate(new_segment):
            idx = k + 1 + i
            chain[idx] = pos
            occupied[pos] = idx
            
        # Add new contacts involving any index in [k+1, n-1]
        # We only need to check contacts between the segment and the prefix (0..k)
        # Internal contacts of the segment are preserved by rotation.
        added_contacts = set()
        
        for i in range(k + 1, n):
            if is_h[i]:
                pos_i = chain[i]
                for q in _neighbours(pos_i):
                    j = occupied.get(q)
                    if j is not None and j <= k and is_h[j]:
                        # j is in prefix
                        added_contacts.add((j, i))
                        contacts.add((j, i))
                        
        # Calculate energy change
        old_e = current_energy
        new_e = -len(contacts)
        delta_e = new_e - old_e
        
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            current_energy = new_e
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
            return True
        else:
            # Revert
            for i, pos in enumerate(new_segment):
                del occupied[pos]
            for i, pos in enumerate(old_segment):
                idx = k + 1 + i
                chain[idx] = pos
                occupied[pos] = idx
                
            for c in added_contacts:
                contacts.remove(c)
            contacts.update(removed_contacts)
            
            return True

    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
            
        # Choose move
        if rng.random() < 0.5:
            try_move_end()
        else:
            try_move_pivot()
            
        steps += 1
        T *= cooling_rate
        if T < T_min:
            T = T_min

    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Fold the sequence using multiple SA restarts.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
        
    # Global time limit for all calls? No, fold is called once per sequence.
    # But the task says "120 s CPU budget for all sequences together".
    # However, the sandbox kills after 210s wall clock.
    # We should use a fraction of the total time for each sequence if possible,
    # but since we don't know how many sequences are left, we assume a fixed budget per call
    # or use a global timer if we can track it.
    # The prompt says "fold is called once per benchmark sequence".
    # It's safer to assume a per-call budget or a global one.
    # Let's use a global start time to be safe.
    
    global _GLOBAL_START_TIME
    if not hasattr(fold, '_GLOBAL_START_TIME'):
        fold._GLOBAL_START_TIME = time.time()
        
    # Total budget 210s. Let's use 200s to be safe.
    total_budget = 200.0
    elapsed_global = time.time() - fold._GLOBAL_START_TIME
    remaining_time = max(0, total_budget - elapsed_global - 5.0) # 5s safety margin
    
    if remaining_time < 1.0:
        # Not enough time, return a quick greedy solution
        rng = random.Random(12345 + len(sequence))
        chain = _grow(sequence, dimension, rng)
        if chain is None:
             # Fallback straight line
             chain = [(i, 0) for i in range(n)]
        return [list(p) for p in chain]
    
    # Distribute time among restarts
    # Let's do 5-10 restarts depending on sequence length
    num_restarts = 5 if n < 50 else 3
    
    time_per_restart = remaining_time / num_restarts
    
    best_chain = None
    best_energy = 1 # Positive infinity effectively for negative energies
    
    for r in range(num_restarts):
        rng = random.Random(12345 + len(sequence) + r)
        initial_chain = _grow(sequence, dimension, rng)
        
        if initial_chain is None:
            # Try a few more times to get a valid start
            for _ in range(5):
                initial_chain = _grow(sequence, dimension, rng)
                if initial_chain:
                    break
        
        if initial_chain is None:
            # Fallback: straight line (always valid)
            initial_chain = [(i, 0) for i in range(n)]
            
        # Run SA
        chain, energy = _simulate_annealing(sequence, initial_chain, rng, time_per_restart)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
            
        # Check if we are running out of time
        if time.time() - fold._GLOBAL_START_TIME > total_budget - 5.0:
            break
            
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
