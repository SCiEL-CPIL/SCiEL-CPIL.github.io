"""HP lattice protein folding: simulated annealing with segment relocation and greedy growth."""
# EVOLVE-BLOCK-START
import random
import time
import math

_T0 = time.time()
_DEADLINE = 195.0  # wall-clock limit with safety margin (210s kill, leave 15s buffer)


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _neighbours(point):
    """Return 4-neighbours of a 2D lattice point."""
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def _compute_energy(sequence, chain):
    """Compute energy: -number of H-H non-bonded contacts."""
    n = len(chain)
    if n < 3:
        return 0
    
    pos_to_idx = {}
    for i, p in enumerate(chain):
        pos_to_idx[p] = i
    
    contacts = 0
    for i in range(n):
        if sequence[i] != 'H':
            continue
        x, y = chain[i]
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nb = (x + dx, y + dy)
            j = pos_to_idx.get(nb)
            if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts


def _grow(sequence, rng):
    """Generate a random self-avoiding walk as initial conformation with greedy bias."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        
        for nb in _neighbours(last):
            if nb in occupied:
                continue
            
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(nb):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            
            free = sum(1 for q in _neighbours(nb) if q not in occupied)
            
            score = gain * 10.0 + free * 0.1 + rng.random()
            options.append((score, nb))
        
        if not options:
            return None
        
        options.sort(key=lambda x: x[0], reverse=True)
        next_pos = options[0][1]
        
        chain.append(next_pos)
        occupied[next_pos] = i
    
    return chain


def _try_relocate_segment(sequence, chain, start_idx, end_idx, rng):
    """
    Attempt to relocate the segment chain[start_idx:end_idx+1] to a new position.
    Returns the new chain if successful, else None.
    """
    n = len(chain)
    seg_len = end_idx - start_idx + 1
    
    if start_idx == 0 or end_idx == n - 1:
        return None
    
    left_neighbor = chain[start_idx - 1]
    right_neighbor = chain[end_idx + 1]
    
    # Get candidate start positions (neighbors of left_neighbor)
    start_candidates = [p for p in _neighbours(left_neighbor)]
    
    # Build set of occupied positions excluding the segment
    occupied_by_others = set(chain[:start_idx] + chain[end_idx + 1:])
    
    valid_starts = [p for p in start_candidates if p not in occupied_by_others]
    if not valid_starts:
        return None
    
    rng.shuffle(valid_starts)
    
    # Try multiple random walks from each start candidate
    max_walks_per_start = 10
    
    for start_pos in valid_starts:
        # Quick check for segment length 1
        if seg_len == 1:
            if abs(start_pos[0] - right_neighbor[0]) + abs(start_pos[1] - right_neighbor[1]) == 1:
                new_chain = chain[:start_idx] + [start_pos] + chain[end_idx + 1:]
                if len(set(new_chain)) == n:
                    return new_chain
            continue
        
        # Try multiple random walks
        for _ in range(max_walks_per_start):
            path = [start_pos]
            visited = {start_pos}
            current = start_pos
            valid = True
            
            for step in range(1, seg_len):
                neighbors = [p for p in _neighbours(current) if p not in visited and p not in occupied_by_others]
                
                if not neighbors:
                    valid = False
                    break
                
                # Bias toward right_neighbor for longer segments
                if seg_len > 5 and step > seg_len // 2:
                    dists = [abs(p[0] - right_neighbor[0]) + abs(p[1] - right_neighbor[1]) for p in neighbors]
                    min_d = min(dists)
                    weights = [1.0 / (d - min_d + 1) for d in dists]
                    total_w = sum(weights)
                    r = rng.random() * total_w
                    cum = 0
                    chosen = 0
                    for idx, w in enumerate(weights):
                        cum += w
                        if r <= cum:
                            chosen = idx
                            break
                    next_pos = neighbors[chosen]
                else:
                    next_pos = rng.choice(neighbors)
                
                path.append(next_pos)
                visited.add(next_pos)
                current = next_pos
            
            if valid:
                # Check if last position is adjacent to right_neighbor
                if abs(path[-1][0] - right_neighbor[0]) + abs(path[-1][1] - right_neighbor[1]) == 1:
                    new_chain = chain[:start_idx] + path + chain[end_idx + 1:]
                    if len(set(new_chain)) == n:
                        # Verify connectivity
                        conn_valid = True
                        for i in range(n - 1):
                            if abs(new_chain[i][0] - new_chain[i+1][0]) + abs(new_chain[i][1] - new_chain[i+1][1]) != 1:
                                conn_valid = False
                                break
                        if conn_valid:
                            return new_chain
    
    return None


def _pivot_move(chain, rng):
    """Perform a pivot move: rotate a suffix of the chain around a pivot point."""
    n = len(chain)
    if n < 4:
        return None
    
    # Choose pivot index (not at ends)
    pivot = rng.randint(1, n - 2)
    pivot_pos = chain[pivot]
    suffix = chain[pivot + 1:]
    
    # Generate all 4 rotations of the suffix relative to pivot
    rotations = []
    prefix_set = set(chain[:pivot + 1])
    
    for angle in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        # Rotate vector (x, y) by 90 degrees
        # (0, 1): 90 deg CCW: (x, y) -> (-y, x)
        # (0, -1): 90 deg CW: (x, y) -> (y, -x)
        # (1, 0): 180 deg: (x, y) -> (-x, -y)
        # (-1, 0): identity (skip)
        if angle == (-1, 0):
            continue
            
        rotated = []
        valid = True
        
        for j, (x, y) in enumerate(suffix):
            rx, ry = x - pivot_pos[0], y - pivot_pos[1]
            
            if angle == (0, 1):  # 90 CCW
                new_x, new_y = -ry, rx
            elif angle == (0, -1):  # 90 CW
                new_x, new_y = ry, -rx
            else:  # 180
                new_x, new_y = -rx, -ry
            
            pos = (pivot_pos[0] + new_x, pivot_pos[1] + new_y)
            
            if pos in prefix_set:
                valid = False
                break
            
            rotated.append(pos)
        
        if valid and len(rotated) == len(suffix):
            new_chain = chain[:pivot + 1] + rotated
            if len(set(new_chain)) == n:
                rotations.append(new_chain)
    
    if not rotations:
        return None
    return rng.choice(rotations)


def _try_local_search(sequence, chain, rng, max_iters=1000):
    """Perform local search with simulated annealing."""
    n = len(chain)
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    T = 5.0
    cooling = 0.999
    
    for it in range(max_iters):
        if _remaining() < 5.0:
            break
        
        # Choose a move type
        move_type = rng.random()
        
        if move_type < 0.5:
            # Pivot move
            new_chain = _pivot_move(chain, rng)
        elif move_type < 0.8:
            # Segment relocation
            if n < 5:
                new_chain = None
            else:
                # Choose segment length
                seg_len = rng.choice([1, 2, 3, 4, 5])
                if n <= seg_len + 2:
                    new_chain = None
                else:
                    start = rng.randint(1, n - seg_len - 1)
                    end = start + seg_len - 1
                    new_chain = _try_relocate_segment(sequence, chain, start, end, rng)
        else:
            # Simple random move: try to move one residue
            idx = rng.randint(1, n - 1)
            new_chain = list(chain)
            x, y = chain[idx]
            moved = False
            occupied_without_idx = set(chain[:idx] + chain[idx+1:])
            
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                new_pos = (x + dx, y + dy)
                if new_pos in occupied_without_idx:
                    continue
                
                if idx > 0 and idx < n - 1:
                    prev = chain[idx - 1]
                    next_pos = chain[idx + 1]
                    if abs(new_pos[0] - prev[0]) + abs(new_pos[1] - prev[1]) != 1:
                        continue
                    if abs(new_pos[0] - next_pos[0]) + abs(new_pos[1] - next_pos[1]) != 1:
                        continue
                
                new_chain[idx] = new_pos
                moved = True
                break
            
            if not moved:
                continue
        
        if new_chain is None:
            continue
        
        # Verify self-avoidance
        if len(set(new_chain)) != n:
            continue
        
        # Verify connectivity
        valid = True
        for i in range(n - 1):
            if abs(new_chain[i][0] - new_chain[i+1][0]) + abs(new_chain[i][1] - new_chain[i+1][1]) != 1:
                valid = False
                break
        if not valid:
            continue
        
        new_energy = _compute_energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        # Simulated annealing acceptance
        if delta < 0 or rng.random() < math.exp(-delta / T):
            chain = new_chain
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(new_chain)
        
        T *= cooling
        if T < 0.1:
            T = 0.1
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Fold sequence into low-energy conformation using simulated annealing."""
    if dimension != 2:
        # Fallback for non-2D (shouldn't happen per task)
        return [[i] + [0] * (dimension - 1) for i in range(len(sequence))]
    
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0  # Start with unfolded (energy 0)
    
    # Time budget: use most of the available time
    start_time = time.time()
    time_budget = min(_remaining() - 5.0, 180.0)  # Leave 5s buffer
    
    # Number of restarts based on sequence length
    if n <= 30:
        num_restarts = 50
    elif n <= 50:
        num_restarts = 30
    elif n <= 70:
        num_restarts = 20
    else:
        num_restarts = 15
    
    iters_per_restart = max(200, int(1000000 / n))
    
    for restart in range(num_restarts):
        if _remaining() < 10.0:
            break
        
        # Generate initial conformation
        chain = _grow(sequence, rng)
        if chain is None or len(chain) != n:
            continue
        
        # Verify it's valid
        if len(set(chain)) != n:
            continue
        valid = True
        for i in range(n - 1):
            if abs(chain[i][0] - chain[i+1][0]) + abs(chain[i][1] - chain[i+1][1]) != 1:
                valid = False
                break
        if not valid:
            continue
        
        # Run local search
        chain, energy = _try_local_search(sequence, chain, rng, max_iters=iters_per_restart)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
        
        # Check time
        if time.time() - start_time > time_budget:
            break
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
    
    return [[x, y] for x, y in best_chain]
# EVOLVE-BLOCK-END
