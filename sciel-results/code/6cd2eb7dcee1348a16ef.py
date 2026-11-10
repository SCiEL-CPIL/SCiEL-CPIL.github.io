"""HP lattice protein folding on the cubic lattice."""
# EVOLVE-BLOCK-START
import time
import random
import math

_GLOBAL_DEADLINE = None

def _neighbours(point):
    """Yield 6 neighbors of a point in 3D lattice."""
    for axis in range(3):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Compute energy as minus the number of non-adjacent H-H contacts."""
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _fast_grow(sequence, dimension, rng):
    """
    Fast randomized growth.
    Tries to grow the chain by picking a random valid neighbor.
    If stuck, backtracks (simple random walk with backtracking).
    """
    n = len(sequence)
    if n == 0:
        return []
    
    chain = [(0, 0, 0)]
    occupied = {chain[0]: 0}
    
    for i in range(1, n):
        candidates = []
        for site in _neighbours(chain[-1]):
            if site not in occupied:
                candidates.append(site)
        
        if not candidates:
            # Backtrack
            # This simple implementation might fail for long chains if it gets stuck in a dead end too deep.
            # For HP folding, random walks often get stuck.
            # Let's use a slightly smarter approach: if stuck, restart with a different seed or use a spiral.
            # To keep it simple and robust, we'll just return None if we get stuck, 
            # and the caller will retry or use a fallback.
            return None
        
        site = rng.choice(candidates)
        chain.append(site)
        occupied[site] = i
        
    return chain

def _initial_conformation(sequence, dimension, rng):
    """
    Generate an initial valid conformation.
    Uses a spiral-like or compact random walk.
    """
    n = len(sequence)
    # Try a few times to get a valid conformation using fast growth
    for _ in range(10):
        chain = _fast_grow(sequence, dimension, rng)
        if chain is not None:
            return chain
    
    # Fallback: Straight line (valid but bad energy)
    return [(i, 0, 0) for i in range(n)]

def _rotate_vector(v, axis, angle):
    """
    Rotate a 3D vector v around a given axis (0,1,2) by 90, 180, or 270 degrees.
    Returns the new vector.
    """
    x, y, z = v
    if axis == 0: # X axis
        if angle == 1: # 90 deg
            return (x, -z, y)
        elif angle == 2: # 180 deg
            return (x, -y, -z)
        elif angle == 3: # 270 deg
            return (x, z, -y)
    elif axis == 1: # Y axis
        if angle == 1:
            return (z, y, -x)
        elif angle == 2:
            return (-x, y, z)
        elif angle == 3:
            return (-z, y, x)
    elif axis == 2: # Z axis
        if angle == 1:
            return (y, -x, z)
        elif angle == 2:
            return (-x, -y, z)
        elif angle == 3:
            return (-y, x, z)
    return v

def _pivot_move(sequence, chain, i, rng):
    """
    Pivot move: Rotate suffix chain[i:] around chain[i-1].
    Returns (new_chain, energy_delta) or None.
    """
    n = len(chain)
    if i <= 0 or i >= n:
        return None
    
    pivot = chain[i-1]
    suffix = chain[i:]
    
    # Choose rotation axis and angle
    axis = rng.randint(0, 2)
    angle = rng.randint(1, 3)
    
    new_suffix = []
    for p in suffix:
        v = (p[0] - pivot[0], p[1] - pivot[1], p[2] - pivot[2])
        nv = _rotate_vector(v, axis, angle)
        new_p = (pivot[0] + nv[0], pivot[1] + nv[1], pivot[2] + nv[2])
        new_suffix.append(new_p)
        
    # Check validity
    prefix_set = set(chain[:i])
    seen = set()
    for p in new_suffix:
        if p in seen:
            return None
        seen.add(p)
        
    for p in new_suffix:
        if p in prefix_set:
            return None
            
    new_chain = chain[:i] + new_suffix
    
    # Calculate energy delta
    # Only need to check contacts involving the moved part (suffix)
    # Internal contacts of suffix are preserved.
    # We need to check contacts between suffix and prefix.
    
    # Build a set of prefix points for O(1) lookup
    # Actually, we already have prefix_set.
    # We need to know which prefix points are H and adjacent to suffix points.
    
    old_energy = 0
    new_energy = 0
    
    # To be efficient, we can just recalculate energy for the whole chain if n is small,
    # but for n=100+, it's better to do delta.
    # However, implementing delta for pivot is tricky because the "prefix" is fixed.
    # Let's just recalculate energy for the new chain. It's O(N*6) which is ~600 ops for N=100.
    # In a Python loop, this might be slow if we do 100k steps.
    # Let's try to optimize the energy calculation.
    
    # Actually, let's just return the new chain and let the caller compute energy?
    # No, SA needs delta for acceptance.
    
    # Let's compute energy delta properly.
    # Contacts involving indices >= i.
    # 1. Contacts between suffix and prefix.
    # 2. Contacts within suffix (unchanged).
    
    # So we only need to remove old contacts between suffix and prefix, and add new ones.
    
    # Let's build a map of prefix points to indices and types
    prefix_map = {}
    for idx in range(i):
        prefix_map[chain[idx]] = idx
    
    # Old contacts involving suffix
    # For each H in suffix, check neighbors in prefix
    old_delta = 0
    for idx in range(i, n):
        if sequence[idx] == 'H':
            p = chain[idx]
            for q in _neighbours(p):
                if q in prefix_map:
                    j = prefix_map[q]
                    if sequence[j] == 'H':
                        # Check if they are adjacent in chain (j and idx)
                        # Since j < i <= idx, they are not adjacent unless idx == j+1 which is impossible as j < i.
                        # Wait, if i=1, j=0, idx=1. They are adjacent.
                        # But pivot is at i-1. Suffix starts at i.
                        # So j < i and idx >= i.
                        # They are adjacent if idx == j + 1.
                        # Since j < i and idx >= i, the only case is j = i-1 and idx = i.
                        # But we are checking neighbors of chain[idx].
                        # If idx=i, neighbor is chain[i-1] (pivot).
                        # So we must exclude the bond to the pivot if it's a chain bond.
                        if abs(idx - j) > 1:
                            old_delta += 1

    new_delta = 0
    for idx in range(i, n):
        if sequence[idx] == 'H':
            p = new_chain[idx]
            for q in _neighbours(p):
                if q in prefix_map:
                    j = prefix_map[q]
                    if sequence[j] == 'H':
                        if abs(idx - j) > 1:
                            new_delta += 1

    delta = new_delta - old_delta
    
    return new_chain, delta

def _segment_move(sequence, chain, i, j, rng):
    """
    Segment move: Rotate chain[i+1:j+1] around chain[i].
    Returns (new_chain, energy_delta) or None.
    """
    n = len(chain)
    if i < 0 or j >= n or i >= j:
        return None
        
    pivot = chain[i]
    segment = chain[i+1:j+1]
    
    axis = rng.randint(0, 2)
    angle = rng.randint(1, 3)
    
    new_segment = []
    for p in segment:
        v = (p[0] - pivot[0], p[1] - pivot[1], p[2] - pivot[2])
        nv = _rotate_vector(v, axis, angle)
        new_p = (pivot[0] + nv[0], pivot[1] + nv[1], pivot[2] + nv[2])
        new_segment.append(new_p)
        
    # Check validity
    prefix_set = set(chain[:i+1]) # Includes pivot
    suffix_set = set(chain[j+1:])
    seen = set()
    for p in new_segment:
        if p in seen:
            return None
        seen.add(p)
        
    for p in new_segment:
        if p in prefix_set or p in suffix_set:
            return None
            
    new_chain = chain[:i+1] + new_segment + chain[j+1:]
    
    # Energy delta
    # Internal contacts of segment are preserved.
    # We need to check contacts between segment and (prefix + suffix).
    
    # Map of all points NOT in segment to their indices and types
    # This is O(N).
    # For now, let's just recalculate energy for the whole chain if N < 50, else delta.
    # Given the time constraint, let's try delta.
    
    # Build map of static parts
    static_map = {}
    for idx in range(0, i+1):
        static_map[chain[idx]] = idx
    for idx in range(j+1, n):
        static_map[chain[idx]] = idx
        
    old_delta = 0
    for idx in range(i+1, j+1):
        if sequence[idx] == 'H':
            p = chain[idx]
            for q in _neighbours(p):
                if q in static_map:
                    j_idx = static_map[q]
                    if sequence[j_idx] == 'H':
                        if abs(idx - j_idx) > 1:
                            old_delta += 1

    new_delta = 0
    for idx in range(i+1, j+1):
        if sequence[idx] == 'H':
            p = new_chain[idx]
            for q in _neighbours(p):
                if q in static_map:
                    j_idx = static_map[q]
                    if sequence[j_idx] == 'H':
                        if abs(idx - j_idx) > 1:
                            new_delta += 1

    delta = new_delta - old_delta
    
    return new_chain, delta

def _reversal_move(sequence, chain, i, j, rng):
    """
    Reversal move: Reverse chain[i:j+1].
    Returns (new_chain, energy_delta) or None.
    """
    n = len(chain)
    if i < 0 or j >= n or i >= j:
        return None
        
    segment = chain[i:j+1]
    new_segment = segment[::-1]
    
    # Check validity
    prefix_set = set(chain[:i])
    suffix_set = set(chain[j+1:])
    seen = set()
    for p in new_segment:
        if p in seen:
            return None
        seen.add(p)
        
    for p in new_segment:
        if p in prefix_set or p in suffix_set:
            return None
            
    new_chain = chain[:i] + new_segment + chain[j+1:]
    
    # Energy delta
    # Internal contacts are preserved (distances are same, just reversed).
    # Check contacts between segment and static parts.
    
    static_map = {}
    for idx in range(0, i):
        static_map[chain[idx]] = idx
    for idx in range(j+1, n):
        static_map[chain[idx]] = idx
        
    old_delta = 0
    for idx in range(i, j+1):
        if sequence[idx] == 'H':
            p = chain[idx]
            for q in _neighbours(p):
                if q in static_map:
                    j_idx = static_map[q]
                    if sequence[j_idx] == 'H':
                        if abs(idx - j_idx) > 1:
                            old_delta += 1

    new_delta = 0
    for idx in range(i, j+1):
        if sequence[idx] == 'H':
            p = new_chain[idx]
            for q in _neighbours(p):
                if q in static_map:
                    j_idx = static_map[q]
                    if sequence[j_idx] == 'H':
                        if abs(idx - j_idx) > 1:
                            new_delta += 1

    delta = new_delta - old_delta
    
    return new_chain, delta

def _is_valid_chain(chain):
    """
    Check if the chain is self-avoiding and has unit steps.
    """
    n = len(chain)
    if n == 0:
        return True
    seen = set()
    for i, p in enumerate(chain):
        if p in seen:
            return False
        seen.add(p)
        if i > 0:
            prev = chain[i-1]
            dist = abs(p[0]-prev[0]) + abs(p[1]-prev[1]) + abs(p[2]-prev[2])
            if dist != 1:
                return False
    return True

def _simulated_annealing(sequence, chain, rng, deadline):
    """
    Simulated Annealing with Pivot moves only (Segment/Reversal were invalid).
    """
    n = len(chain)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    
    best_chain = list(current_chain)
    best_energy = current_energy
    
    t_end = deadline
    initial_temp = 5.0
    
    temperature = initial_temp
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < t_end:
        # Only use Pivot moves as they preserve connectivity
        i = rng.randint(1, n - 1)
        result = _pivot_move(sequence, current_chain, i, rng)
        
        if result is None:
            # Failed move (collision), continue
            continue
            
        new_chain, delta = result
        
        # Acceptance
        if delta > 0 or rng.random() < math.exp(delta / temperature):
            current_chain = new_chain
            current_energy -= delta
            no_improve_count = 0
            if current_energy < best_energy:
                best_chain = list(current_chain)
                best_energy = current_energy
        else:
            no_improve_count += 1
            
        # Cooling
        temperature *= 0.999
        
        if temperature < 0.01:
            if no_improve_count > max_no_improve:
                temperature = initial_temp * 0.5
                no_improve_count = 0
            else:
                temperature = initial_temp * 0.1
                no_improve_count = 0

    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Find a good conformation for the given HP sequence.
    Returns a list of lattice points (lists of integers).
    """
    global _GLOBAL_DEADLINE
    if _GLOBAL_DEADLINE is None:
        _GLOBAL_DEADLINE = time.time() + 330.0
    
    start_time = time.time()
    remaining = _GLOBAL_DEADLINE - start_time
    if remaining < 1.0:
        # Not enough time, return a simple valid conformation
        n = len(sequence)
        return [[i, 0, 0] for i in range(n)]
    
    # Use a fraction of the remaining time
    # We have 26 sequences.
    # Let's allocate roughly equal time to each.
    # But we don't know how many have been processed.
    # The prompt says "fold is called once per benchmark sequence".
    # So we assume we have `remaining` time for this specific call?
    # No, the deadline is global.
    # So we should use a portion of the remaining time.
    
    # Heuristic: Assume we have ~1/26th of the total time left on average?
    # Or just use a fixed budget per sequence if possible.
    # Total budget 330s. 26 sequences. ~12s per sequence.
    # Let's use 10s per sequence to be safe.
    budget = min(remaining, 10.0)
    deadline = start_time + budget
    
    n = len(sequence)
    rng = random.Random(12345 + len(sequence) + int(time.time() * 1000) % 10000)
    
    # Initial conformation
    chain = _initial_conformation(sequence, dimension, rng)
    if chain is None:
        chain = [(i, 0, 0) for i in range(n)]
        
    # Simulated Annealing
    best_chain, best_energy = _simulated_annealing(sequence, chain, rng, deadline)
    
    # Validation: Ensure the chain is valid (connected and self-avoiding)
    if not _is_valid_chain(best_chain):
        # Fallback to a valid straight line if SA produced an invalid chain (should not happen with Pivot, but safe)
        best_chain = [(i, 0, 0) for i in range(n)]
    
    # Convert to lists of ints
    return [list(p) for p in best_chain]

# EVOLVE-BLOCK-END
