"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Explicit deadline: evaluator kills after 210s wall-clock; keep 30s safety margin.
_DEADLINE = time.time() + 170.0

# Directions: 0:R, 1:D, 2:L, 3:U
DIRS = [(1, 0), (0, -1), (-1, 0), (0, 1)]
VEC_TO_IDX = {(1,0):0, (0,-1):1, (-1,0):2, (0,1):3}

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

def _generate_random_saw(n, rng):
    """
    Generate a random self-avoiding walk of length n using a rejection-free 
    backtracking method or simple random walk with restarts.
    For short-to-medium chains, simple random walk with restarts is often fast enough
    if we bias towards "open" spaces, but for HP folding, we want compactness.
    However, for the initial state of a Global Search, a valid SAW is required.
    We use a simple greedy random growth to ensure validity quickly.
    """
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    # Try to generate a compact random SAW
    # We use a simple randomized greedy approach to get a valid start quickly
    # This is better than pure random walk which often gets stuck.
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site not in occupied:
                # Prefer sites that are already surrounded (compactness)
                # but ensure we don't trap ourselves
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                # Score: prefer lower free space (compact) but allow some randomness
                # If we are too greedy on compactness, we might trap.
                # Let's just pick a random valid move, weighted by "closeness" to existing chain
                # Actually, just random is fine for diversity in multi-start.
                options.append(site)
        
        if not options:
            # Failed to grow, restart (should be rare for n<100)
            return _generate_random_saw(n, rng)
        
        # Pick randomly
        next_site = rng.choice(options)
        chain.append(next_site)
        occupied[next_site] = i
        
    return chain

def _pivot_move(chain, sequence, rng):
    """
    Perform a pivot move: rotate the suffix of the chain starting at index k
    by 90 or 180 degrees.
    Returns (new_chain, new_energy_delta) or (None, 0) if invalid.
    """
    n = len(chain)
    if n <= 2:
        return None, 0
    
    # Choose pivot point k. 
    # Uniform distribution over k in [1, n-1] is standard, but 
    # longer suffixes allow larger conformational changes.
    # We sample k uniformly.
    k = rng.randint(1, n - 1)
    
    # Choose rotation: 90 CW (1), 90 CCW (-1), 180 (2)
    # 180 is a 2-step rotation.
    rot_type = rng.choice([1, -1, 2])
    
    # Extract suffix
    suffix = chain[k:]
    
    # Calculate new positions for suffix
    # The pivot point is chain[k]. The segment chain[k+1:] is rotated around chain[k].
    # Actually, standard pivot: rotate segment k..N-1 around vertex k-1? 
    # No, usually rotate segment k..N-1 around vertex k.
    # Let's define: fix chain[0..k-1]. Rotate chain[k..n-1] around chain[k-1].
    # Wait, if we rotate around k-1, the vector from k-1 to k changes direction.
    # Let's stick to: Fix prefix 0..k-1. Rotate suffix k..n-1 around k-1.
    
    # Vector from pivot (k-1) to first suffix atom (k)
    pivot = chain[k-1]
    v0 = (chain[k][0] - pivot[0], chain[k][1] - pivot[1])
    
    # Apply rotation to v0
    if rot_type == 1: # 90 CW: (x, y) -> (y, -x)
        v_new = (v0[1], -v0[0])
    elif rot_type == -1: # 90 CCW: (x, y) -> (-y, x)
        v_new = (-v0[1], v0[0])
    else: # 180: (x, y) -> (-x, -y)
        v_new = (-v0[0], -v0[1])
        
    new_chain = chain[:k]
    # First new atom
    new_pos = (pivot[0] + v_new[0], pivot[1] + v_new[1])
    
    # Check collision with prefix
    # We need a set of occupied positions for the prefix (0..k-1)
    # And we must check the new suffix against the prefix and itself.
    
    # Optimization: 
    # 1. Create set of prefix positions
    prefix_set = set(chain[:k])
    
    # 2. If new_pos is in prefix_set, fail
    if new_pos in prefix_set:
        return None, 0
        
    new_chain.append(new_pos)
    
    # 3. Rotate the rest of the suffix relative to the new first atom?
    # No, rotation is rigid.
    # The relative vectors between atoms in the suffix remain the same, 
    # but the whole vector space is rotated.
    # Actually, it's easier to rotate the coordinate system of the suffix.
    # Let's store the original suffix coordinates relative to the pivot.
    # Or simpler: 
    # The new chain is chain[:k] + rotated_suffix.
    # The rotated suffix is obtained by rotating the vector (chain[i] - chain[k-1])
    # for each i in k..n-1.
    
    # Let's re-calculate all points in the suffix to ensure consistency.
    # This is O(N) per move, which is fine for N=100.
    
    new_suffix = []
    for i in range(k, n):
        # Vector from pivot (k-1) to original atom i
        vi = (chain[i][0] - pivot[0], chain[i][1] - pivot[1])
        
        if rot_type == 1:
            vi_new = (vi[1], -vi[0])
        elif rot_type == -1:
            vi_new = (-vi[1], vi[0])
        else:
            vi_new = (-vi[0], -vi[1])
            
        new_pt = (pivot[0] + vi_new[0], pivot[1] + vi_new[1])
        new_suffix.append(new_pt)
        
    # Check self-avoidance of new_suffix against prefix and itself
    # We need to check:
    # 1. new_suffix atoms are not in prefix_set
    # 2. new_suffix atoms are unique among themselves
    
    seen_in_suffix = set()
    for pt in new_suffix:
        if pt in prefix_set:
            return None, 0
        if pt in seen_in_suffix:
            return None, 0
        seen_in_suffix.add(pt)
        
    # Also check the junction: chain[k-1] is in prefix. 
    # The first atom of new_suffix is at distance 1 from pivot.
    # We already checked new_pos (which is new_suffix[0]) against prefix_set.
    
    # Construct full new chain
    new_chain = chain[:k] + new_suffix
    
    # Calculate Energy Delta
    # Instead of recomputing full energy (O(N)), we can compute delta.
    # But for N=100, O(N) is ~100 operations. 10k moves/sec is easy.
    # Full recompute is simpler and less error-prone.
    # Let's do full recompute for correctness in the first pass, 
    # but we can optimize later if needed.
    
    # Actually, let's return the chain and let the caller compute energy.
    # Or compute here to avoid duplicate work if rejected?
    # The caller needs the energy to decide acceptance.
    # Let's compute energy here.
    
    # To speed up energy calc, we can use the fact that only the suffix moved.
    # But contacts can form between:
    # 1. Suffix-Suffix (internal to new_suffix)
    # 2. Suffix-Prefix (boundary)
    # 3. Prefix-Prefix (unchanged)
    
    # Since we are doing this in a tight loop, let's just recompute.
    # N=100 is small.
    
    return new_chain, 0 # Placeholder, caller computes

def _compute_energy_fast(sequence, chain):
    """
    Optimized energy computation using a set of H positions.
    """
    h_positions = []
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            h_positions.append((i, p))
            
    contacts = 0
    # For each H, check neighbors
    # We need a map of position to index
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    
    for i, p in h_positions:
        for q in _neighbours(p):
            j = pos_to_idx.get(q)
            if j is not None and j > i and abs(j - i) > 1:
                # Since we only iterate i < j in the pair logic?
                # No, we iterate all H. 
                # If we count (i,j) and (j,i), we divide by 2.
                # Condition j > i ensures we count each pair once.
                # But we must ensure sequence[j] is H.
                # Since we only added H to h_positions, j is H.
                contacts += 1
                
    return -contacts

def fold(sequence, dimension):
    """
    Fold the sequence using Global Stochastic Local Search with Pivot Moves.
    """
    rng = random.Random(42)
    n = len(sequence)
    
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    # Time allocation
    # We have ~170s.
    # We will run multiple independent SA restarts.
    
    best_chain = None
    best_energy = 0
    
    # Number of restarts based on sequence length
    # Shorter sequences can afford more restarts.
    if n < 20:
        num_restarts = 10
        time_per_restart = 15.0
    elif n < 50:
        num_restarts = 5
        time_per_restart = 30.0
    else:
        num_restarts = 3
        time_per_restart = 50.0
        
    total_time = num_restarts * time_per_restart
    
    for run_id in range(num_restarts):
        if time.time() > _DEADLINE - 10:
            break
            
        # 1. Generate Initial Conformation
        # Use a compact random SAW
        init_chain = _generate_random_saw(n, rng)
        if init_chain is None:
            # Fallback: straight line (rarely fails for SAW gen)
            init_chain = [(i, 0) for i in range(n)]
            
        current_chain = init_chain
        current_energy = _compute_energy_fast(sequence, current_chain)
        
        # 2. Simulated Annealing with Pivot Moves
        # Start temperature: High enough to accept bad moves
        # Energy range is roughly [-N, 0]. Delta E is usually 1-5.
        # T_start = 5.0 is a good start.
        T = 5.0
        T_end = 0.1
        cool_rate = 0.9995 # Slow cooling
        
        start_time = time.time()
        local_best_chain = list(current_chain)
        local_best_energy = current_energy
        
        while time.time() < start_time + time_per_restart and T > T_end:
            # Perform a pivot move
            new_chain, _ = _pivot_move(current_chain, sequence, rng)
            
            if new_chain is None:
                T *= cool_rate
                continue
                
            new_energy = _compute_energy_fast(sequence, new_chain)
            delta = new_energy - current_energy
            
            # Acceptance criterion
            if delta < 0 or (T > 0 and rng.random() < math.exp(-delta / max(T, 0.01))):
                current_chain = new_chain
                current_energy = new_energy
                
                if current_energy < local_best_energy:
                    local_best_energy = current_energy
                    local_best_chain = list(current_chain)
                    
            T *= cool_rate
            
        # Update global best
        if best_chain is None or local_best_energy < best_energy:
            best_chain = local_best_chain
            best_energy = local_best_energy

    # Final Polish: Run a short SA on the global best to clean up
    if best_chain is not None:
        # Use a lower temperature and shorter time
        polish_chain = list(best_chain)
        polish_energy = _compute_energy_fast(sequence, polish_chain)
        
        T = 1.0
        T_end = 0.1
        cool_rate = 0.999
        start_time = time.time()
        
        while time.time() < start_time + 10.0 and T > T_end:
            new_chain, _ = _pivot_move(polish_chain, sequence, rng)
            if new_chain is None:
                T *= cool_rate
                continue
                
            new_energy = _compute_energy_fast(sequence, new_chain)
            delta = new_energy - polish_energy
            
            if delta < 0 or (T > 0 and rng.random() < math.exp(-delta / max(T, 0.01))):
                polish_chain = new_chain
                polish_energy = new_energy
                
                if polish_energy < best_energy:
                    best_chain = polish_chain
                    best_energy = polish_energy
                    
            T *= cool_rate

    # Fallback
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
