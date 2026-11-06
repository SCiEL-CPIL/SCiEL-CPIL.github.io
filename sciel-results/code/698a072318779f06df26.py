"""HP lattice protein folding on the square lattice."""
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
    # Budget is 120s CPU for all sequences, 210s wall-clock kill.
    # Use 195s as a safe deadline to ensure we finish before kill.
    return 195.0 - (time.time() - _START_TIME)


def _neighbours(point):
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def _energy(sequence, chain):
    """Compute energy: -1 per non-adjacent H-H contact."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _is_valid(sequence, chain):
    """Check self-avoidance and unit steps."""
    n = len(chain)
    seen = set()
    for i, p in enumerate(chain):
        if p in seen:
            return False
        seen.add(p)
        if i > 0:
            prev = chain[i - 1]
            if abs(p[0] - prev[0]) + abs(p[1] - prev[1]) != 1:
                return False
    return True


def _grow(sequence, dimension, rng):
    """Randomized greedy chain growth."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
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
            score = gain * 10 + free * 0.1 + rng.random()
            options.append((score, site))
        
        if not options:
            return None
        
        options.sort(reverse=True)
        # Pick from top 3 with probability weighting
        top_k = min(3, len(options))
        weights = [1.0 / (k + 1) for k in range(top_k)]
        total_w = sum(weights)
        r = rng.random() * total_w
        cum = 0
        chosen = options[0][1]
        for k in range(top_k):
            cum += weights[k]
            if r <= cum:
                chosen = options[k][1]
                break
        
        chain.append(chosen)
        occupied[chosen] = i
    
    return chain


def _segment_move(sequence, chain, seg_start, seg_len, rng, occupied_set):
    """Try to relocate a segment of the chain to a new position via random walk."""
    n = len(chain)
    if seg_start + seg_len > n:
        return None
    
    # Extract segment
    seg = chain[seg_start:seg_start + seg_len]
    
    # Determine the "anchor" points: before seg_start and after seg_start+seg_len
    # The segment must connect from chain[seg_start-1] to chain[seg_start+seg_len]
    
    if seg_start == 0:
        # Segment starts at beginning, must end at chain[seg_start+seg_len]
        target = chain[seg_start + seg_len]
        start_point = None  # free start
    elif seg_start + seg_len == n:
        # Segment ends at end, must start from chain[seg_start-1]
        start_point = chain[seg_start - 1]
        target = None  # free end
    else:
        start_point = chain[seg_start - 1]
        target = chain[seg_start + seg_len]
    
    # Remove segment from occupied set temporarily
    for p in seg:
        occupied_set.discard(p)
    
    # Try to find a new path of length seg_len from start_point to target
    # using random walk, avoiding occupied cells
    
    best_new_seg = None
    best_gain = -1000
    
    num_walks = 5
    for _ in range(num_walks):
        # Random walk from start_point
        if start_point is None:
            # Start from a random neighbor of target
            candidates = [q for q in _neighbours(target) if q not in occupied_set]
            if not candidates:
                continue
            current = rng.choice(candidates)
            path = [current]
        else:
            current = start_point
            path = []
        
        success = True
        for step in range(seg_len):
            if start_point is not None:
                # We need to move step units from start_point
                # current is already at the position after 'step' moves
                if step == 0:
                    # First move
                    neighbors = [q for q in _neighbours(current) if q not in occupied_set and q != (path[-1] if path else None)]
                    if not neighbors:
                        success = False
                        break
                    current = rng.choice(neighbors)
                    path.append(current)
                else:
                    # Subsequent moves
                    neighbors = [q for q in _neighbours(current) if q not in occupied_set and q != (path[-1] if path else None)]
                    if not neighbors:
                        success = False
                        break
                    current = rng.choice(neighbors)
                    path.append(current)
            else:
                # Walking towards target
                if step == seg_len - 1:
                    # Last step must reach target
                    if current == target:
                        pass
                    elif current in _neighbours(target):
                        path.append(target)
                        current = target
                    else:
                        success = False
                        break
                else:
                    neighbors = [q for q in _neighbours(current) if q not in occupied_set and q != (path[-1] if path else None)]
                    if not neighbors:
                        success = False
                        break
                    # Bias towards target
                    target_dir = (target[0] - current[0], target[1] - current[1])
                    scored = []
                    for q in neighbors:
                        dist = abs(q[0] - target[0]) + abs(q[1] - target[1])
                        scored.append((dist, q))
                    scored.sort()
                    # Pick from top 2
                    if len(scored) >= 2:
                        r = rng.random()
                        if r < 0.7:
                            current = scored[0][1]
                        else:
                            current = scored[1][1]
                    else:
                        current = scored[0][1]
                    path.append(current)
        
        if not success:
            continue
        
        # Check if path ends at target (if target is not None)
        if target is not None and path[-1] != target:
            continue
        
        # Check if path starts correctly
        if start_point is not None and len(path) > 0 and path[0] != start_point:
            # path[0] should be the first move from start_point
            if abs(path[0][0] - start_point[0]) + abs(path[0][1] - start_point[1]) != 1:
                continue
        
        # Verify no collisions within path
        path_set = set(path)
        if len(path_set) != len(path):
            continue
        if any(p in occupied_set for p in path):
            continue
        
        # Calculate energy change
        old_energy = _energy(sequence, chain)
        new_chain = chain[:seg_start] + path + chain[seg_start + seg_len:]
        new_energy = _energy(sequence, new_chain)
        gain = new_energy - old_energy
        
        if gain > best_gain:
            best_gain = gain
            best_new_seg = path
    
    # Restore occupied set
    for p in seg:
        occupied_set.add(p)
    
    if best_new_seg is not None and best_gain > 0:
        return best_new_seg, best_gain
    
    return None


def _local_search(sequence, chain, rng, time_limit=5.0):
    """Perform local search with segment moves."""
    n = len(chain)
    if n < 4:
        return chain
    
    best_chain = chain[:]
    best_energy = _energy(sequence, best_chain)
    occupied_set = set(best_chain)
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < time_limit:
        if no_improve_count > max_no_improve:
            break
        
        # Try segment moves of various lengths
        improved = False
        for seg_len in [3, 4, 5, 7, 10, 15]:
            if seg_len >= n:
                continue
            # Try multiple random positions
            for _ in range(10):
                seg_start = rng.randint(0, n - seg_len - 1)
                result = _segment_move(sequence, best_chain, seg_start, seg_len, rng, occupied_set)
                if result is not None:
                    new_seg, gain = result
                    # Apply the move
                    old_seg = best_chain[seg_start:seg_start + seg_len]
                    for p in old_seg:
                        occupied_set.discard(p)
                    best_chain[seg_start:seg_start + seg_len] = new_seg
                    for p in new_seg:
                        occupied_set.add(p)
                    best_energy += gain
                    no_improve_count = 0
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation using growth + local search."""
    _init_time()
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0  # We want to minimize energy (more negative is better)
    
    n = len(sequence)
    
    # Time budget for this sequence
    time_per_seq = min(15.0, max(3.0, 120.0 / 10))  # Roughly 12s per sequence
    
    start_time = time.time()
    
    # Phase 1: Multiple growth attempts
    num_growths = min(300, max(50, int(time_per_seq * 20)))
    
    for attempt in range(num_growths):
        if time.time() - start_time > time_per_seq * 0.4:
            break
        
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        if not _is_valid(sequence, chain):
            continue
        
        energy = _energy(sequence, chain)
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)
    
    # Phase 2: Local search on the best chain
    remaining_time = time_per_seq - (time.time() - start_time)
    if remaining_time > 1.0:
        improved_chain = _local_search(sequence, best_chain, rng, time_limit=remaining_time)
        improved_energy = _energy(sequence, improved_chain)
        if improved_energy < best_energy:
            best_chain = improved_chain
            best_energy = improved_energy
    
    # Phase 3: More growth + local search if time remains
    if time.time() - start_time < time_per_seq * 0.8:
        for attempt in range(100):
            if time.time() - start_time > time_per_seq * 0.9:
                break
            
            chain = _grow(sequence, dimension, rng)
            if chain is None:
                continue
            
            if not _is_valid(sequence, chain):
                continue
            
            # Quick local search
            time_left = time_per_seq * 0.9 - (time.time() - start_time)
            if time_left > 0.5:
                chain = _local_search(sequence, chain, rng, time_limit=min(2.0, time_left))
            
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best_chain = chain
                best_energy = energy
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
