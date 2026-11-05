"""HP lattice protein folding using Hamiltonian path skeleton and segment pivots."""
# EVOLVE-BLOCK-START
import random
import time
import math

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """
    Compute the energy of the conformation.
    Energy = - (number of H-H contacts)
    """
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _generate_skeleton(n, dimension=2):
    """
    Generate a compact self-avoiding walk (Hamiltonian path on a grid) of length n.
    Uses a serpentine pattern on a grid of size roughly sqrt(n) x sqrt(n).
    """
    if n <= 1:
        return [(0, 0)]
    
    # Find grid dimensions
    w = int(math.ceil(math.sqrt(n)))
    h = int(math.ceil(n / w))
    
    # Generate serpentine path
    path = []
    for r in range(h):
        if r % 2 == 0:
            for c in range(w):
                path.append((c, r))
        else:
            for c in range(w - 1, -1, -1):
                path.append((c, r))
    
    # Truncate to n
    path = path[:n]
    return path

def _pivot_search(sequence, chain, rng, time_limit):
    """
    Segment pivot search on a fixed skeleton.
    We pick a segment and try to rotate it by 90, 180, 270 degrees around one end.
    This is a powerful move for rearranging the chain topology.
    """
    n = len(sequence)
    if n < 2:
        return chain
    
    current_chain = list(chain)
    occupied = {p: i for i, p in enumerate(current_chain)}
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    T = 1.0
    T_min = 0.01
    cooling = 0.99995
    
    while time.time() - start_time < time_limit:
        # Pick a segment to pivot
        # Shorter segments are more likely to fit, but longer segments allow bigger rearrangements.
        # We pick a random pivot point i and a random segment length.
        i = rng.randint(0, n - 1)
        
        # Determine segment length based on distance to ends
        max_len = min(n - i, 15) # Cap length for efficiency and validity probability
        if max_len <= 1:
            continue
            
        length = rng.randint(2, max_len)
        j = i + length - 1
        
        # Pivot point is current_chain[i]
        pivot = current_chain[i]
        
        # The segment is current_chain[i:j+1]
        # We rotate the segment around pivot.
        # The new segment will be current_chain[i] + rotated(current_chain[i+1:j+1])
        
        # Get the vector from pivot to the next point in the segment
        # Actually, we rotate the entire sequence of points relative to the pivot.
        
        # Let's define the relative coordinates of the segment points
        rel_points = []
        for k in range(i+1, j+1):
            px, py = current_chain[k]
            rx, ry = current_chain[i]
            rel_points.append((px - rx, py - ry))
        
        # Try 3 rotations: 90, 180, 270
        # 90 deg: (x, y) -> (-y, x)
        # 180 deg: (x, y) -> (-x, -y)
        # 270 deg: (x, y) -> (y, -x)
        
        rotations = [
            lambda x, y: (-y, x),
            lambda x, y: (-x, -y),
            lambda x, y: (y, -x)
        ]
        
        chosen_rot = rng.choice(rotations)
        
        # Calculate new absolute points
        new_points = [pivot]
        valid = True
        new_occupied_set = set()
        
        # We need to check if the new points collide with the rest of the chain.
        # The rest of the chain is current_chain[0:i] + current_chain[j+1:n]
        # Note: current_chain[i] is the pivot, which stays fixed.
        # So the "rest" excludes index i.
        
        rest_indices = list(range(0, i)) + list(range(j+1, n))
        rest_points = {current_chain[k] for k in rest_indices}
        
        for rx, ry in rel_points:
            nx, ny = chosen_rot(rx, ry)
            new_pt = (pivot[0] + nx, pivot[1] + ny)
            new_points.append(new_pt)
            
            # Check self-avoidance
            if new_pt in new_occupied_set:
                valid = False
                break
            new_occupied_set.add(new_pt)
            
            if new_pt in rest_points:
                valid = False
                break
        
        if not valid:
            continue
        
        # Check connectivity at the boundaries
        # Left boundary: current_chain[i-1] (if i > 0) must be adjacent to new_points[0] (which is pivot, unchanged)
        # Wait, the pivot is current_chain[i]. 
        # The segment is i..j.
        # The point before the segment is i-1.
        # The point after the segment is j+1.
        
        # Connectivity 1: i-1 to i (pivot). This is unchanged because pivot doesn't move.
        # Connectivity 2: j to j+1.
        # The new point at index j is new_points[-1].
        # The old point at index j+1 is current_chain[j+1].
        if j < n - 1:
            next_pt = current_chain[j+1]
            last_new_pt = new_points[-1]
            if abs(next_pt[0] - last_new_pt[0]) + abs(next_pt[1] - last_new_pt[1]) != 1:
                valid = False
        
        if not valid:
            continue
        
        # Calculate energy delta
        # Old energy contribution of the segment
        # New energy contribution of the segment
        # We can just recalculate energy for the whole chain if it's fast enough?
        # N is up to 100. Recalculation is O(N). 
        # Given the time budget, O(N) per move is likely acceptable for a few thousand moves.
        # But let's try to be smart. 
        # Actually, _energy is O(N * 4). For N=100, that's 400 ops.
        # If we do 10,000 moves, that's 4 million ops. Python can do ~10-50 million simple ops/sec.
        # So 10k moves takes ~0.1-0.4s. This is very fast.
        # So we can just recalculate energy fully.
        
        # Construct new chain
        new_chain = list(current_chain)
        for k in range(i+1, j+1):
            new_chain[k] = new_points[k-i]
            
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if new_energy < current_energy or rng.random() < math.exp(-delta / T):
            # Accept
            # Update occupied dict
            for k in range(i+1, j+1):
                old_pt = current_chain[k]
                del occupied[old_pt]
                new_pt = new_chain[k]
                occupied[new_pt] = k
                
            current_chain = new_chain
            current_energy = new_energy
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(current_chain)
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_chain

def fold(sequence, dimension):
    """Best conformation using Hamiltonian skeleton and pivot search."""
    n = len(sequence)
    if n == 0:
        return []
    
    # Time management
    # The evaluator calls fold() 10 times. Total budget 120s.
    # We track global time to ensure we don't exceed it.
    if not hasattr(fold, '_initialized'):
        fold._start_time = time.time()
        fold._time_limit = 115.0 # Safety margin
        fold._initialized = True
        
    elapsed = time.time() - fold._start_time
    time_left = fold._time_limit - elapsed
    
    if time_left <= 0:
        # Return a valid trivial solution (straight line)
        return [list(p) for p in [(i, 0) for i in range(n)]]
    
    if not hasattr(fold, '_call_count'):
        fold._call_count = 0
    fold._call_count += 1
    calls_done = fold._call_count
    assumed_total = 10
    remaining_calls = max(1, assumed_total - calls_done + 1)
    time_per_call = time_left / remaining_calls
    
    # Ensure time_per_call is not too small
    if time_per_call < 0.1:
         time_per_call = 0.1

    rng = random.Random(12345 + len(sequence) + calls_done)
    
    # Generate serpentine skeleton
    chain = _generate_skeleton(n, dimension)
    
    # Run pivot search
    best_chain = _pivot_search(sequence, chain, rng, time_per_call)
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
