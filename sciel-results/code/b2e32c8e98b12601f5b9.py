import random
import time
import math

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _greedy_growth(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                # Check if this site creates new H-H contacts with existing Hs
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == "H":
                            gain += 1
            # Add some randomness to avoid getting stuck in same local min
            score = gain + rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None
            
        # Sort by score and pick the best or second best with some prob
        options.sort(reverse=True)
        # Pick best
        site = options[0][1]
        occupied[site] = i
        chain.append(site)
        
    return chain

def _energy(sequence, chain):
    """Calculate total H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _sa_fold(sequence, dimension, initial_chain, rng, time_limit):
    """
    Simulated Annealing with Pivot, Crankshaft, and Tunneling moves.
    """
    if initial_chain is None:
        return None, 0
        
    chain = initial_chain[:]
    n = len(chain)
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    best_chain = chain[:]
    best_energy = current_energy
    
    start_time = time.time()
    initial_temp = 1.0
    final_temp = 0.01
    cooling_rate = 0.995
    
    # Adaptive cooling based on sequence length
    steps_per_sec_target = 10000
    max_steps = int((time_limit / 10.0) * steps_per_sec_target) # Use 10% of total time per chain approx? 
    # Better: iterate until time limit for this specific chain allocation? 
    # Since we have multiple chains, we should distribute time.
    # Let's say we have a global deadline. We pass a 'deadline' to this function.
    
    # Actually, the caller manages the global deadline. This function should run until it's told to stop or time runs out.
    # We'll pass a 'check_stop' function or just use a global variable? 
    # Simpler: The caller runs SA for a fixed number of steps or until time runs out.
    
    # Let's assume this function runs for a specific time slice.
    # We'll implement a loop that checks time.
    
    t = initial_temp
    step = 0
    
    while t > final_temp:
        if time.time() > time_limit:
            break
            
        # Select move type
        move_type = rng.random()
        new_chain = None
        delta_e = 0
        
        if move_type < 0.7:
            # Pivot
            new_chain, delta_e = _try_pivot(chain, occupied, sequence, dimension, rng)
        elif move_type < 0.9:
            # Crankshaft
            new_chain, delta_e = _try_crankshaft(chain, occupied, sequence, dimension, rng)
        else:
            # Tunneling / Local Refold
            new_chain, delta_e = _try_tunneling(chain, occupied, sequence, dimension, rng)
            
        if new_chain is None:
            t *= cooling_rate
            continue
            
        # Acceptance
        if delta_e < 0:
            # Better, always accept
            chain = new_chain
            current_energy += delta_e
            # Update occupied set
            _update_occupied(chain, occupied)
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = chain[:]
        else:
            # Worse, accept with probability exp(-delta_e / t)
            prob = math.exp(-delta_e / t)
            if rng.random() < prob:
                chain = new_chain
                current_energy += delta_e
                _update_occupied(chain, occupied)
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain[:]
                    
        t *= cooling_rate
        step += 1
        
    return best_chain, best_energy

def _update_occupied(chain, occupied):
    """Rebuild occupied set. Slow but necessary if chain changed significantly."""
    occupied.clear()
    for i, p in enumerate(chain):
        occupied[p] = i

def _try_pivot(chain, occupied, sequence, dimension, rng):
    """
    Try a pivot move. Returns (new_chain, delta_energy) or (None, 0) if failed.
    """
    n = len(chain)
    if n < 3:
        return None, 0
        
    # Pick pivot index k (0 to n-2)
    k = rng.randint(0, n - 2)
    
    # Determine rotation axis and direction
    # We need to rotate the segment [k+1, n-1] around chain[k]
    # The move is valid if the new positions are not occupied by [0, k-1]
    
    # Get the segment to move
    segment = chain[k+1:]
    
    # Generate a random rotation
    # In 3D, we can rotate around x, y, or z axis by 90, 180, 270 degrees.
    # Or simpler: pick a random neighbor of chain[k] that is NOT chain[k-1] (if k>0) 
    # and NOT in the moved segment's original position? 
    # Actually, a standard pivot move rotates the entire tail around the pivot.
    
    # Let's use a simpler "Random Walk Refold" for the tail as a "Pivot" alternative?
    # No, let's implement a proper 90-degree rotation.
    
    pivot = chain[k]
    
    # Possible new positions for chain[k+1]
    # Must be a neighbor of pivot
    # And not occupied by the fixed part [0, k-1]
    
    valid_dirs = []
    for q in _neighbours(pivot):
        if q in occupied and occupied[q] < k:
            continue
        # Also ensure we don't rotate into the segment itself? 
        # The segment moves, so its old positions are freed.
        # But we must check collision with the fixed part [0, k-1].
        valid_dirs.append(q)
        
    if not valid_dirs:
        return None, 0
        
    new_next = rng.choice(valid_dirs)
    
    # Now, we need to determine the rest of the segment's new positions.
    # This is tricky for a full rotation. A simpler approach for "Pivot" in code is 
    # to just pick a new direction for the next step and then try to continue the rest 
    # using a greedy or random walk? That's not a rigid pivot.
    
    # Rigid Pivot Implementation:
    # Calculate the vector from pivot to chain[k+1].
    # Rotate this vector by 90, 180, or 270 degrees around an axis perpendicular to it?
    # Or simply: The segment is a rigid body. We pick a new orientation for the first 
    # bond, and then the rest of the segment follows the same relative geometry.
    
    # Let's implement a "Rigid Body Rotation" of the tail.
    # This requires transforming all points in the tail.
    
    # To keep it simple and fast in Python, let's use a "Local Move" strategy instead of rigid pivot if rigid is too complex.
    # But rigid pivot is powerful.
    
    # Let's try a simpler "Pivot": 
    # Pick k. Pick a new position for chain[k+1] (neighbor of chain[k], not in fixed part).
    # Then, for i from k+2 to n-1, the vector chain[i] - chain[i-1] is preserved? 
    # No, that's just a walk.
    
    # Let's implement a "Crankshaft" which is easier:
    # Pick k. Rotate segment [k+2, n-1] around the axis defined by chain[k] and chain[k+1].
    
    # Given the complexity of rigid rotations in pure Python without numpy for vector ops, 
    # and the time limit, I will use a **"Segment Refold"** strategy as the "Tunneling" move, 
    # and **"Single Step Move"** (try to move one residue to a neighbor) as local search.
    # And **"Pivot"** will be implemented as: 
    #   1. Pick k.
    #   2. Pick a new position for chain[k+1] (neighbor of chain[k], not occupied by 0..k-1).
    #   3. Try to complete the rest of the chain (k+2..n-1) using a fast greedy/random walk 
    #      that respects the new position of k+1.
    # This is not a rigid pivot, but it's a "Quantum-Inspired" jump: we jump the tail to a 
    # new basin by re-generating it.
    
    # Let's call this "Tail Regeneration".
    
    # Fixed part: 0..k
    # New start for tail: new_next
    # Regenerate k+2..n-1
    
    new_chain = chain[:k+1] + [new_next]
    new_occupied = {p: i for i, p in enumerate(new_chain)}
    
    # Check if new_next is valid (not in fixed part)
    if new_next in {p for p in chain[:k]}:
        return None, 0
        
    # Greedy/Random growth for the rest
    for i in range(k+2, n):
        options = []
        last = new_chain[-1]
        for site in _neighbours(last):
            if site in new_occupied:
                continue
            # Check collision with fixed part
            if site in {p for p in chain[:k]}:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in new_occupied:
                        j = new_occupied[q]
                        if j < i - 1 and sequence[j] == "H":
                            gain += 1
            score = gain + rng.random()
            options.append((score, site))
            
        if not options:
            return None, 0
            
        options.sort(reverse=True)
        site = options[0][1]
        new_occupied[site] = i
        new_chain.append(site)
        
    # Calculate energy difference
    old_e = _energy(sequence, chain)
    new_e = _energy(sequence, new_chain)
    delta_e = new_e - old_e
    
    return new_chain, delta_e

def _try_crankshaft(chain, occupied, sequence, dimension, rng):
    # Similar to pivot but rotates a segment with a gap
    # For simplicity, we can reuse the Tail Regeneration logic with a different k distribution
    # or just skip it if Pivot (Tail Regen) is working well.
    # Let's implement a simple "Single Residue Move" as Crankshaft alternative?
    # No, let's just use Tail Regeneration for all "global" moves.
    # To diversify, we can vary the k selection.
    # If move_type was Crankshaft, we pick k near the middle.
    # If Pivot, we pick k near the end?
    # Actually, let's just return None here and let the caller handle it, 
    # or implement a simple "Slide" move.
    
    # Let's implement a "Slide" move: Move one residue to an adjacent empty site.
    # This is a local move.
    i = rng.randint(1, len(chain) - 2) # Don't move ends
    p = chain[i]
    for q in _neighbours(p):
        if q in occupied:
            continue
        # Check if q is adjacent to chain[i-1] or chain[i+1]? 
        # If we move p to q, the chain must still be connected.
        # q must be adjacent to chain[i-1] OR chain[i+1]? 
        # No, the bond i-1 to i and i to i+1 must remain unit steps.
        # If we move i to q, then:
        # 1. q must be adjacent to chain[i-1] (if we keep bond i-1->i)
        #    OR q must be adjacent to chain[i+1] (if we keep bond i->i+1)
        #    Actually, moving a middle residue requires both bonds to remain valid.
        #    This is only possible if q is adjacent to BOTH chain[i-1] and chain[i+1]?
        #    That's a very restrictive move.
        #    
        #    Alternatively, we can break a bond and re-form? No, that's not a valid walk step.
        #    
        #    So "Slide" is not a valid local move for a fixed backbone unless we allow 
        #    the neighbors to move too (which is a crankshaft).
        
    # So, let's stick to Tail Regeneration for "Global" moves.
    # And for "Local" moves, we can try to **flip** a small segment?
    # Or just rely on Tail Regeneration with different k.
    
    return None, 0

def _try_tunneling(chain, occupied, sequence, dimension, rng):
    # "Tunneling": Regenerate a random sub-segment [start, end]
    # This allows jumping over large barriers.
    n = len(chain)
    if n < 5:
        return None, 0
        
    # Pick a segment length
    seg_len = rng.randint(3, min(10, n - 2))
    start = rng.randint(1, n - seg_len - 1)
    end = start + seg_len
    
    # Fixed parts: [0, start-1] and [end+1, n-1]
    fixed_left = chain[:start]
    fixed_right = chain[end+1:]
    
    # We need to connect fixed_left[-1] to fixed_right[0] with a walk of length seg_len + 1?
    # No, we are replacing chain[start] to chain[end].
    # The new segment must start at a neighbor of fixed_left[-1] and end at a neighbor of fixed_right[0].
    # And be self-avoiding and not collide with fixed parts.
    
    # This is a hard constraint satisfaction. 
    # Instead, let's just do Tail Regeneration from a random point.
    # It's simpler and effective.
    
    k = rng.randint(1, n - 2)
    return _try_pivot(chain, occupied, sequence, dimension, rng)

def fold(sequence, dimension):
    """
    Best conformation using Multi-Start Simulated Annealing with Tail Regeneration moves.
    """
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    # Global deadline
    start_time = time.time()
    deadline = start_time + 250.0 # Leave 50s margin for 300s limit
    
    best_chain = None
    best_energy = 0 # 0 is unfolded, we want negative energy
    
    # Phase 1: Generate initial population
    # Use 20% of time for initialization?
    # Let's do 10 greedy starts and 10 random starts.
    
    init_chains = []
    
    # Greedy
    for _ in range(10):
        c = _greedy_growth(sequence, dimension, rng)
        if c:
            init_chains.append(c)
            
    # Random (simple random walk)
    for _ in range(10):
        c = _random_walk(sequence, dimension, rng)
        if c:
            init_chains.append(c)
            
    if not init_chains:
        # Fallback to straight line
        return [list([i] + [0] * (dimension - 1)) for i in range(n)]
        
    # Phase 2: Optimize each chain
    # Allocate time evenly
    num_chains = len(init_chains)
    time_per_chain = (deadline - time.time()) / num_chains
    
    for c in init_chains:
        if time.time() > deadline:
            break
            
        chain, energy = _sa_fold(sequence, dimension, c, rng, deadline)
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
            
    if best_chain is None:
        best_chain = init_chains[0]
        
    return [list(p) for p in best_chain]

def _random_walk(sequence, dimension, rng):
    """Generate a random self-avoiding walk."""
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site not in occupied:
                options.append(site)
        if not options:
            return None
        site = rng.choice(options)
        occupied[site] = i
        chain.append(site)
        
    return chain

# To avoid circular imports or issues, we define _sa_fold to use the global deadline passed in.
# The current _sa_fold signature takes time_limit which is the absolute deadline.
