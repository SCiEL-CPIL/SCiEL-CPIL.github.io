# EVOLVE-BLOCK-START
import time
import math
import heapq

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _beam_search_fold(sequence, dimension, width, time_limit, start_pos=(0,0,0)):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [list(start_pos)]

    # Precompute remaining H count for heuristic
    rem_h = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        rem_h[i] = rem_h[i + 1] + (1 if sequence[i] == 'H' else 0)

    # Each state: (pos, occupied_set, contacts, path)
    # To save memory, we store path as tuple of positions.
    # For n up to 48, path length is 48. Occupied set size 48.
    # This is heavy. We will limit width.
    
    # Initial state
    init_pos = start_pos
    init_occ = {init_pos: 0}
    init_contacts = 0
    init_path = (init_pos,)
    
    # Heuristic: max possible future contacts
    # A crude admissible heuristic is 0. 
    # A better one: for each remaining H, it can form at most 4 contacts, 
    # but many will be with each other.
    # Let's use a simple priority: current contacts + potential.
    # Potential: sum of 1 for each remaining H residue that is adjacent to an existing H?
    # Actually, for beam search, we want to keep the states with highest current contacts.
    # We can use a greedy beam search: keep top K states by current energy (contacts).
    
    # State: (contacts, path_tuple, occupied_set)
    # We need to compare states. Since we want max contacts, we use -contacts as key for min-heap
    # or just sort and keep top K.
    
    beam = []
    # Add initial state
    # We need a unique ID for states to avoid duplicates in the beam?
    # Different paths can lead to the same (pos, occupied, contacts).
    # We should prune duplicates.
    
    # Let's use a list of (contacts, path, occupied)
    # And at each step, generate next states, and keep top W by contacts.
    
    # To handle duplicates: if we reach the same (pos, occupied) with lower contacts, discard.
    # If higher, replace.
    
    # Since occupied is a set of tuples, and path is a tuple of tuples, this is expensive.
    # For n=48, width=100, this might be slow in Python.
    
    # Optimization: 
    # 1. Use tuples for positions.
    # 2. Use frozenset for occupied? No, we need to map pos->index for energy calc.
    #    But we can just check neighbors in the set.
    
    # Let's try a simpler approach:
    # Store: (pos, occupied_dict, contacts, path_list)
    
    # Actually, let's just implement a basic beam search and see.
    
    current_beam = [(init_contacts, init_path, init_occ)]
    
    for i in range(1, n):
        next_beam = []
        seen = {} # key: (pos, frozenset(occupied)) -> (contacts, path, occ)
        
        for contacts, path, occ in current_beam:
            last_pos = path[-1]
            res = sequence[i]
            
            for nb in _neighbours(last_pos):
                if nb in occ:
                    continue
                
                new_occ = occ.copy()
                new_occ[nb] = i
                
                new_contacts = contacts
                if res == 'H':
                    # Check neighbors of nb for existing H's
                    for q in _neighbours(nb):
                        if q in new_occ:
                            j = new_occ[q]
                            if j < i - 1 and sequence[j] == 'H':
                                new_contacts += 1
                
                new_path = path + (nb,)
                
                # Key for deduplication
                # Using frozenset of occupied positions is a good unique identifier for the shape?
                # No, the path matters for the future, but the set of occupied positions
                # and the current position determine the future possibilities and energy.
                # Wait, the energy only depends on the set of occupied positions and the sequence.
                # The specific path doesn't matter for energy, only the set of positions.
                # However, the "self-avoiding" constraint is enforced by the set.
                # So if we reach the same (last_pos, occupied_set) with the same sequence prefix,
                # the future is identical. We only need to keep the one with max contacts.
                
                key = (nb, frozenset(new_occ.keys()))
                
                if key in seen:
                    if new_contacts > seen[key][0]:
                        seen[key] = (new_contacts, new_path, new_occ)
                else:
                    seen[key] = (new_contacts, new_path, new_occ)
        
        # Convert seen dict to list and sort by contacts
        candidates = list(seen.values())
        
        # Sort by contacts descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Keep top W
        current_beam = candidates[:width]
        
        if not current_beam:
            break
            
        # Time check
        # This is a bit expensive to check every step, but necessary.
        
    if not current_beam:
        return None
        
    # Return the best path
    best_contacts, best_path, _ = current_beam[0]
    return [list(p) for p in best_path]

def _local_optimize(sequence, chain, dimension, time_limit):
    """
    Simple local optimization using pivot moves, similar to seed but faster.
    """
    if not chain:
        return chain, 0
        
    current_chain = [tuple(p) for p in chain]
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    T = 1.0
    alpha = 0.99
    
    while time.time() < time_limit:
        # Pivot move
        n = len(current_chain)
        if n < 4:
            break
            
        pivot_idx = int(n * 0.5) + (time.time() % 1) # Pseudo-random pivot
        pivot_idx = max(1, min(n-2, int(pivot_idx)))
        
        p_prev = current_chain[pivot_idx - 1]
        p_pivot = current_chain[pivot_idx]
        
        bond_vector = [p_pivot[i] - p_prev[i] for i in range(dimension)]
        bond_axis = -1
        for i in range(dimension):
            if bond_vector[i] != 0:
                bond_axis = i
                break
        if bond_axis == -1:
            continue
            
        rot_axes = [i for i in range(dimension) if i != bond_axis]
        if not rot_axes:
            continue
            
        r_axis = rot_axes[int(time.time() * 1000) % len(rot_axes)]
        angle = 1 if int(time.time() * 10000) % 2 == 0 else -1
        
        head_set = set(current_chain[:pivot_idx + 1])
        new_tail = []
        valid = True
        occupied_tail = set()
        
        for i in range(pivot_idx + 1, n):
            orig = current_chain[i]
            dx = orig[0] - p_pivot[0]
            dy = orig[1] - p_pivot[1]
            dz = orig[2] - p_pivot[2]
            
            if r_axis == 0:
                if angle == 1: nx, ny, nz = dx, -dz, dy
                else: nx, ny, nz = dx, dz, -dy
            elif r_axis == 1:
                if angle == 1: nx, ny, nz = dz, dy, -dx
                else: nx, ny, nz = -dz, dy, dx
            else:
                if angle == 1: nx, ny, nz = -dy, dx, dz
                else: nx, ny, nz = dy, -dx, dz
                
            new_point = (p_pivot[0] + nx, p_pivot[1] + ny, p_pivot[2] + nz)
            
            if new_point in head_set or new_point in occupied_tail:
                valid = False
                break
                
            occupied_tail.add(new_point)
            new_tail.append(new_point)
            
        if not valid:
            continue
            
        new_chain = current_chain[:pivot_idx + 1] + new_tail
        new_energy = _energy(sequence, new_chain)
        delta_e = new_energy - current_energy
        
        if delta_e < 0:
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        else:
            prob = math.exp(-delta_e / T)
            if (time.time() * 1e9) % 1000000 / 1000000.0 < prob:
                current_chain = new_chain
                current_energy = new_energy
                
        T *= alpha
        
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Hybrid Beam Search + Local Optimization.
    """
    start_time = time.time()
    global_deadline = start_time + 280.0 # Safety margin
    
    n = len(sequence)
    
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    
    best_chain = None
    best_energy = 0
    
    # Strategy:
    # 1. For short chains (n <= 30), use Beam Search with wider beam.
    # 2. For medium chains (30 < n <= 60), use Beam Search with narrower beam.
    # 3. For long chains (n > 60), use Beam Search with very narrow beam or fallback to random+SA.
    # 4. Always follow up with local optimization.
    
    # Determine beam width
    if n <= 25:
        beam_width = 500
        beam_time = 10.0
    elif n <= 40:
        beam_width = 200
        beam_time = 15.0
    elif n <= 60:
        beam_width = 100
        beam_time = 20.0
    else:
        beam_width = 50
        beam_time = 30.0
        
    # Try a few different beam widths if time permits?
    # For now, just one beam search attempt per sequence.
    
    # Run Beam Search
    chain = _beam_search_fold(sequence, dimension, beam_width, beam_time)
    
    if chain is not None:
        # Local Optimize
        opt_chain, opt_energy = _local_optimize(sequence, chain, dimension, global_deadline - time.time() - 5.0)
        if opt_chain is not None:
            best_chain = opt_chain
            best_energy = opt_energy
    else:
        # Fallback: Random + SA
        # This is a simplified version of the seed's random fold
        import random
        rng = random.Random(12345 + len(sequence))
        for attempt in range(10):
            chain = [(0, 0, 0)]
            occupied = {chain[0]: 0}
            for i in range(1, n):
                neighbors = list(_neighbours(chain[-1]))
                rng.shuffle(neighbors)
                placed = False
                for nb in neighbors:
                    if nb not in occupied:
                        chain.append(nb)
                        occupied[nb] = i
                        placed = True
                        break
                if not placed:
                    break
            if len(chain) == n:
                opt_chain, opt_energy = _local_optimize(sequence, chain, dimension, global_deadline - time.time() - 5.0)
                if opt_chain is not None:
                    if best_chain is None or opt_energy < best_energy:
                        best_chain = opt_chain
                        best_energy = opt_energy
                break # Only one random attempt in fallback for simplicity? 
                # Actually, let's try a few if time permits.
                # But let's keep it simple.
                
    if best_chain is None:
        # Absolute fallback
        best_chain = [[i, 0, 0] for i in range(n)]
        
    return best_chain

# EVOLVE-BLOCK-END
