"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time

# Explicit deadline: evaluator kills after 210s wall-clock; keep 30s safety margin.
_DEADLINE = time.time() + 170.0

# Directions: 0:R, 1:D, 2:L, 3:U
DIRS = [(1, 0), (0, -1), (-1, 0), (0, 1)]
DIR_NAMES = ['R', 'D', 'L', 'U']

def _neighbours(point):
    """Yield the 4 unit-step neighbours of a 2D point (tuple)."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _compute_energy(sequence, chain):
    """Compute the negative number of non-adjacent H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2

def _pivot_move(chain, pivot, direction, rng):
    """
    Perform a pivot move: rotate the segment from pivot to end by 0, 1, or 2 steps.
    Returns None if the new chain is not self-avoiding.
    """
    n = len(chain)
    if pivot <= 0 or pivot >= n - 1:
        return None
    
    # Directions: 0=Right, 1=Down, 2=Left, 3=Up
    # Rotation: +1 (CW), +2 (180), +3 (CCW)
    # We pick a random rotation amount
    rot = rng.choice([1, 2, 3])
    
    # Extract the tail segment
    tail = chain[pivot:]
    head = chain[:pivot]
    
    # The connection point is head[-1]
    # The first step of the tail is determined by the direction from head[-2] to head[-1]
    # Actually, it's simpler to just reconstruct the tail based on the new direction of the first step.
    
    # Determine the original direction of the first step in the tail
    # The tail starts at pivot. The step is from pivot-1 to pivot.
    # Wait, standard pivot: pick a pivot point i. The segment i..n-1 is rotated.
    # The connection is at i-1.
    # The direction of the step i-1 -> i is fixed.
    # We rotate the rest of the chain relative to that step.
    
    # Let's implement it properly:
    # 1. Identify the vector v = chain[pivot] - chain[pivot-1]
    # 2. Rotate all subsequent vectors by 'rot' steps.
    
    if pivot < 1:
        return None
        
    v0 = (chain[pivot][0] - chain[pivot-1][0], chain[pivot][1] - chain[pivot-1][1])
    
    # Rotate v0 by rot
    # R: (1,0) -> D:(0,-1) -> L:(-1,0) -> U:(0,1)
    # Index: 0->1->2->3->0
    # We need to map vectors to indices
    vec_to_idx = {(1,0):0, (0,-1):1, (-1,0):2, (0,1):3}
    
    if v0 not in vec_to_idx:
        return None
        
    start_idx = vec_to_idx[v0]
    
    new_tail = [chain[pivot]]
    current_pos = chain[pivot]
    current_dir_idx = start_idx
    
    for i in range(pivot + 1, n):
        # Get original direction
        prev_pos = chain[i-1]
        curr_pos = chain[i]
        v = (curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1])
        
        if v not in vec_to_idx:
            return None
            
        orig_idx = vec_to_idx[v]
        
        # The turn relative to the previous step
        # turn = (orig_idx - prev_dir_idx) % 4
        # But we are reconstructing.
        # Let's just store the sequence of turns in the tail.
        # This is getting complicated. Let's use a simpler pivot implementation:
        # Store the chain as a list of positions.
        # To rotate:
        # 1. Calculate the new position of each point in the tail.
        # 2. Check for collisions.
        
        pass # Skip this complex logic, use a simpler "regrow" or "flip" if pivot is too hard to implement correctly quickly.
    
    # Alternative: Use a simpler "segment flip" or "regrow" which is robust.
    # But the directive asks for a structural change. 
    # Let's implement a proper pivot using relative directions.
    
    # Re-implementing pivot correctly:
    # 1. Extract tail chain: T = chain[pivot:]
    # 2. Calculate relative directions: D = [T[i] - T[i-1] for i in 1..len(T)-1]
    # 3. Rotate D by rot.
    # 4. Reconstruct T' starting from chain[pivot-1] using rotated D.
    # 5. Check self-avoidance.
    
    T = chain[pivot:]
    L = len(T)
    if L == 0:
        return chain
    
    # Relative directions
    rel_dirs = []
    for i in range(1, L):
        dx = T[i][0] - T[i-1][0]
        dy = T[i][1] - T[i-1][1]
        if (dx, dy) in vec_to_idx:
            rel_dirs.append(vec_to_idx[(dx, dy)])
        else:
            return None
            
    # Rotate
    new_rel_dirs = [(d + rot) % 4 for d in rel_dirs]
    
    # Reconstruct
    new_T = [T[0]]
    pos = T[0]
    for d in new_rel_dirs:
        dx, dy = DIRS[d]
        pos = (pos[0] + dx, pos[1] + dy)
        new_T.append(pos)
        
    # Check collision
    # The new chain is chain[:pivot] + new_T
    # But wait, the connection point chain[pivot-1] is fixed.
    # The first point of new_T is T[0] which is chain[pivot].
    # We need to check if the new_T points collide with chain[:pivot] (excluding the last point? No, all of it)
    # And if new_T points collide with each other (they shouldn't if relative dirs are valid, but let's check).
    
    # Actually, the pivot rotates the segment *including* the pivot point?
    # Standard definition: Pick i. Rotate segment i..n-1 around i-1? No, around the bond i-1..i?
    # If we rotate around the bond, the point i stays fixed? No, the bond vector changes.
    # Let's stick to: Rotate the segment starting at index `pivot` (inclusive) such that the bond (pivot-1, pivot) is rotated.
    # This means chain[pivot] moves.
    
    # My reconstruction above kept T[0] fixed. That's not a pivot, that's a "tail move".
    # Let's do a true pivot:
    # 1. Fix chain[:pivot].
    # 2. The bond from pivot-1 to pivot is rotated.
    # 3. This implies chain[pivot] moves.
    
    # Let's try a different, simpler structural change: **Regrow with Lookahead** + **Global SA**.
    # The current program uses Greedy + Regrow SA.
    # I will use **Random Restart Hill Climbing with Pivot Moves**.
    
    # Let's implement a simple Pivot:
    # 1. Pick pivot index i (1 to n-2).
    # 2. The segment i..n-1 is "detached".
    # 3. We try to reattach it by rotating the first step (i-1 to i) by 1, 2, or 3 steps.
    # 4. The rest of the segment follows the same relative turns.
    
    # This is what I started above. Let's fix the reconstruction.
    # The relative turns are preserved.
    # The first direction is rotated.
    
    # Let's re-do the reconstruction for True Pivot:
    # 1. Get relative turns of the tail segment (i..n-1).
    #    Turn at i is determined by dir(i-1->i) and dir(i->i+1).
    #    This is complex.
    
    # Easier: Store the tail as a list of steps (directions).
    # Steps: s_i = dir(chain[i-1] -> chain[i]) for i in 1..n-1.
    # Pivot at i: Rotate s_i, s_{i+1}, ..., s_{n-1} by rot.
    # Reconstruct chain from s_0...s_{i-1} (fixed) + rotated s_i...s_{n-1}.
    
    # Let's implement this.
    
    # 1. Convert chain to steps
    # 2. Rotate steps i..end
    # 3. Reconstruct
    # 4. Check collision
    
    # This is O(N).
    
    # Let's write a helper to convert chain to steps and back.
    
    def _chain_to_steps(chain):
        steps = []
        for i in range(1, len(chain)):
            dx = chain[i][0] - chain[i-1][0]
            dy = chain[i][1] - chain[i-1][1]
            steps.append(vec_to_idx[(dx, dy)])
        return steps

    def _steps_to_chain(steps):
        chain = [(0, 0)]
        for d in steps:
            dx, dy = DIRS[d]
            chain.append((chain[-1][0] + dx, chain[-1][1] + dy))
        return chain

    # Pivot Move Implementation
    if pivot < 1:
        return None
        
    steps = _chain_to_steps(chain)
    
    # Rotate steps from pivot to end
    # Note: step index k corresponds to bond k-1 -> k.
    # If we pivot at index `pivot` (node), we rotate bonds starting from `pivot`?
    # If we rotate bond `pivot` (node pivot-1 to pivot), then node pivot moves.
    # So we rotate steps[pivot-1 : ].
    
    # Let's say we pick a bond index b (0 to n-2).
    # We rotate steps[b:].
    # This moves node b+1 and all subsequent nodes.
    
    # My input `pivot` is a node index.
    # Let's assume `pivot` is the bond index for simplicity in the call.
    # If caller passes node index, we adjust.
    
    # Let's assume caller passes bond index `b`.
    b = pivot
    if b < 0 or b >= len(steps):
        return None
        
    new_steps = steps[:]
    for i in range(b, len(new_steps)):
        new_steps[i] = (new_steps[i] + rot) % 4
        
    new_chain = _steps_to_chain(new_steps)
    
    # Check self-avoidance
    seen = set()
    for p in new_chain:
        if p in seen:
            return None
        seen.add(p)
        
    return new_chain

def _generate_random_saw(n, rng):
    """Generate a random self-avoiding walk of length n."""
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
        
    # Start with a random direction
    start_dir = rng.randint(0, 3)
    chain = [(0, 0)]
    occupied = {(0, 0)}
    current_dir = start_dir
    
    for i in range(1, n):
        # Try to move
        # Prefer straight, then left, then right (or random)
        # To ensure we don't get stuck easily, we can use a simple heuristic.
        # But for random SAW, we just pick a valid random direction.
        
        options = []
        for d in range(4):
            dx, dy = DIRS[d]
            new_pos = (chain[-1][0] + dx, chain[-1][1] + dy)
            if new_pos not in occupied:
                options.append(d)
                
        if not options:
            return None # Failed to generate
            
        next_dir = rng.choice(options)
        dx, dy = DIRS[next_dir]
        new_pos = (chain[-1][0] + dx, chain[-1][1] + dy)
        chain.append(new_pos)
        occupied.add(new_pos)
        
    return chain

def _local_improve_pivot(sequence, chain, rng, max_iters=2000, start_temp=3.0, cool_rate=0.997):
    """
    Improve conformation using pivot moves with simulated annealing.
    """
    n = len(chain)
    if n < 4:
        return chain, _compute_energy(sequence, chain)
        
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    temp = start_temp
    no_improve = 0
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 10:
            break
        if temp < 0.05:
            break
            
        # Pick a random bond to pivot
        # Avoid pivots at the very beginning or end?
        # Bond index 0 to n-2
        b = rng.randint(0, n - 2)
        
        new_chain = _pivot_move(chain, b, 0, rng)
        
        if new_chain is None:
            temp *= cool_rate
            no_improve += 1
            continue
            
        new_energy = _compute_energy(sequence, new_chain)
        
        delta = new_energy - current_energy
        if delta < 0 or (temp > 0 and rng.random() < (10 ** (-delta / max(temp, 0.1)))):
            chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
            
        temp *= cool_rate
        
        if no_improve > 500:
            break
            
    return best_chain, best_energy

def _ga_crossover(sequence, chain1, chain2, rng):
    """
    Genetic Algorithm crossover:
    1. Convert chains to step sequences.
    2. Perform uniform crossover on steps.
    3. Reconstruct chain.
    4. If invalid (collision), try to repair or return None.
    """
    n = len(sequence)
    
    def _chain_to_steps(chain):
        steps = []
        for i in range(1, len(chain)):
            dx = chain[i][0] - chain[i-1][0]
            dy = chain[i][1] - chain[i-1][1]
            vec_to_idx = {(1,0):0, (0,-1):1, (-1,0):2, (0,1):3}
            steps.append(vec_to_idx[(dx, dy)])
        return steps

    def _steps_to_chain(steps):
        chain = [(0, 0)]
        for d in steps:
            dx, dy = DIRS[d]
            chain.append((chain[-1][0] + dx, chain[-1][1] + dy))
        return chain

    steps1 = _chain_to_steps(chain1)
    steps2 = _chain_to_steps(chain2)
    
    # Uniform crossover
    new_steps = []
    for i in range(len(steps1)):
        if rng.random() < 0.5:
            new_steps.append(steps1[i])
        else:
            new_steps.append(steps2[i])
            
    # Mutation
    for i in range(len(new_steps)):
        if rng.random() < 0.1:
            new_steps[i] = rng.randint(0, 3)
            
    new_chain = _steps_to_chain(new_steps)
    
    # Check validity
    seen = set()
    for p in new_chain:
        if p in seen:
            return None
        seen.add(p)
        
    return new_chain

def fold(sequence, dimension):
    """
    Fold the sequence using a portfolio of Random SAW + Pivot SA + GA.
    """
    rng = random.Random(42)
    best_chain = None
    best_energy = 0  # We want to minimize energy (more negative is better)

    n = len(sequence)
    
    # Adaptive time allocation
    if n < 30:
        sa_time = 80.0
        ga_time = 50.0
    else:
        sa_time = 100.0
        ga_time = 40.0
        
    # Phase 1: Random SAW + Pivot SA
    # Generate several random starting conformations and refine with SA
    phase1_end = time.time() + sa_time
    
    # For short sequences, random SAW might fail often. Use greedy as fallback?
    # Let's try to generate valid SAWs.
    
    attempts = 0
    while time.time() < phase1_end and attempts < 200:
        attempts += 1
        chain = _generate_random_saw(n, rng)
        if chain is None:
            continue
            
        # Refine with Pivot SA
        sa_chain, sa_energy = _local_improve_pivot(
            sequence, chain, rng, max_iters=1000, start_temp=3.0, cool_rate=0.995)
            
        if best_chain is None or sa_energy < best_energy:
            best_chain = sa_chain
            best_energy = sa_energy
            
    # Fallback if no SAW generated (shouldn't happen for n<100 usually)
    if best_chain is None:
        # Use a simple greedy growth as fallback
        # (Reusing logic from previous program if needed, but let's try a simple straight line + random walk repair)
        # Actually, let's just use a simple greedy growth here to ensure we have a starting point.
        
        def _greedy_grow_fallback(sequence, rng):
            start = (0, 0)
            chain = [start]
            occupied = {start: 0}
            n = len(sequence)
            
            for i in range(1, n):
                last = chain[-1]
                options = []
                for site in _neighbours(last):
                    if site in occupied:
                        continue
                    gain = 0
                    if sequence[i] == 'H':
                        for q in _neighbours(site):
                            j = occupied.get(q)
                            if j is not None and j < i - 1 and sequence[j] == 'H':
                                gain += 1
                    free = sum(1 for q in _neighbours(site) if q not in occupied)
                    score = gain * 10.0 + free * 0.1 + rng.random() * 1.0
                    options.append((score, site))
                    
                if not options:
                    return None
                    
                best_score, best_site = max(options, key=lambda x: x[0])
                occupied[best_site] = i
                chain.append(best_site)
                
            return chain

        chain = _greedy_grow_fallback(sequence, rng)
        if chain:
            best_chain = chain
            best_energy = _compute_energy(sequence, chain)
            # Refine it
            sa_chain, sa_energy = _local_improve_pivot(
                sequence, chain, rng, max_iters=2000, start_temp=3.0, cool_rate=0.995)
            if sa_energy < best_energy:
                best_chain = sa_chain
                best_energy = sa_energy

    # Phase 2: Genetic Algorithm
    if best_chain is not None:
        phase2_end = time.time() + ga_time
        population = [best_chain]
        
        # Add some diversity
        for _ in range(5):
            c = _generate_random_saw(n, rng)
            if c:
                population.append(c)
                
        for gen in range(50):
            if time.time() > phase2_end:
                break
                
            new_pop = []
            
            # Elitism
            new_pop.append(best_chain)
            
            # Crossover and Mutation
            while len(new_pop) < 10:
                p1 = rng.choice(population)
                p2 = rng.choice(population)
                child = _ga_crossover(sequence, p1, p2, rng)
                if child is not None:
                    new_pop.append(child)
                    
            # Evaluate and Select
            scored = []
            for c in new_pop:
                e = _compute_energy(sequence, c)
                scored.append((e, c))
                
            scored.sort(key=lambda x: x[0])
            
            # Keep top 10
            new_pop = [c for e, c in scored[:10]]
            
            # Update best
            if scored[0][0] < best_energy:
                best_energy = scored[0][0]
                best_chain = scored[0][1]
                
            population = new_pop
            
    # Final Local Improvement
    if best_chain is not None:
        improved_chain, improved_energy = _local_improve_pivot(
            sequence, best_chain, rng, max_iters=2000, start_temp=1.0, cool_rate=0.999)
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy

    # Fallback: straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
