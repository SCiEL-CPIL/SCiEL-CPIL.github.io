"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time

_START_TIME = None
_DEADLINE = 195.0

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    _init_time()
    return _DEADLINE - (time.time() - _START_TIME)

def _neighbours(point):
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _energy(sequence, chain):
    """Compute HP contact energy: -1 for each H-H pair on neighbouring sites not adjacent in chain."""
    where = {p: i for i, p in enumerate(chain)}
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    contacts = 0
    for i in h_indices:
        x, y = chain[i]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = where.get((x + dx, y + dy))
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts

def _grow_beam(sequence, dimension, rng, beam_width=8):
    """
    Beam search growth.
    Keeps the top `beam_width` partial chains based on a heuristic score.
    Score combines immediate H-H contacts and free space to avoid trapping.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    start = (0, 0)
    initial_chain = [start]
    initial_occupied = {start: 0}
    # score is lower (more negative) is better
    beam = [(initial_chain, initial_occupied, 0.0)]
    
    for i in range(1, n):
        next_beam = []
        for chain, occupied, score in beam:
            current = chain[-1]
            # Generate all valid extensions
            for site in _neighbours(current):
                if site in occupied:
                    continue
                
                # Calculate immediate gain (contacts)
                gain = 0
                if sequence[i] == 'H':
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            gain += 1
                
                # Calculate free space (open neighbors)
                free = 0
                for q in _neighbours(site):
                    if q not in occupied:
                        free += 1
                
                # Heuristic: favor contacts strongly, free space weakly
                # Weighted sum: -2.0 * gain - 0.1 * free
                new_score = score - 2.0 * gain - 0.1 * free
                
                # Add to candidate list
                new_chain = chain + [site]
                new_occupied = dict(occupied)
                new_occupied[site] = i
                
                next_beam.append((new_chain, new_occupied, new_score))
        
        if not next_beam:
            return None
        
        # Sort by score (lower is better, i.e., more negative energy potential)
        # Tie-breaker: random to maintain diversity
        next_beam.sort(key=lambda x: (x[2], rng.random()))
        
        # Keep top beam_width
        beam = next_beam[:beam_width]
        
        # Early termination if time is low
        if _time_remaining() < 2.0:
             break

    if not beam:
        return None
    
    # Pick the best one
    best_chain, _, _ = beam[0]
    return best_chain

def _local_search(sequence, chain, rng, max_steps=200):
    """
    Local search using single-residue moves (kinks).
    Moves a residue i to a new position that maintains connectivity with i-1 and i+1.
    Accepts moves that improve energy (lower value).
    """
    n = len(chain)
    if n < 3:
        return chain, _energy(sequence, chain)
    
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _energy(sequence, chain)
    
    for _ in range(max_steps):
        if _time_remaining() < 1.0:
            break
        # Pick a random internal residue to move
        i = rng.randint(1, n - 2)
        prev = chain[i-1]
        curr = chain[i]
        next_ = chain[i+1]
        
        # Find valid positions for the new location of residue i
        # It must be a neighbor of prev AND a neighbor of next_
        # And it must not be the current position
        candidates = set(_neighbours(prev)) & set(_neighbours(next_))
        candidates.discard(curr)
        
        if not candidates:
            continue
            
        # Pick a random candidate from the valid ones
        new_pos = rng.choice(list(candidates))
        
        # Check if the new position is occupied by another residue
        if new_pos in occupied:
            continue
            
        # Perform the move
        del occupied[curr]
        occupied[new_pos] = i
        chain[i] = new_pos
        
        # Calculate new energy
        new_energy = _energy(sequence, chain)
        
        # Accept if better (more negative), otherwise revert
        if new_energy < current_energy:
            current_energy = new_energy
        else:
            # Revert
            del occupied[new_pos]
            occupied[curr] = i
            chain[i] = curr
    
    return chain, current_energy

def _segment_relocation(sequence, chain, rng, max_attempts=5, max_seg_len=10):
    """
    Attempt to relocate a segment of the chain to a new location.
    This is a more powerful move than single kinks.
    1. Pick a segment [a:b].
    2. Remove it.
    3. Try to find a new path of length L connecting a-1 and b (or a and b-1 depending on orientation) 
       that fits in the free space.
    For simplicity and speed, we will try to re-insert the segment by finding a valid path 
    between the two endpoints of the gap created by removal.
    """
    n = len(chain)
    if n < 5:
        return chain, _energy(sequence, chain)
    
    current_energy = _energy(sequence, chain)
    best_energy = current_energy
    best_chain = chain
    
    for _ in range(max_attempts):
        if _time_remaining() < 1.0:
            break
            
        # Pick segment length
        L = rng.randint(2, min(max_seg_len, n - 3))
        # Pick start index
        a = rng.randint(1, n - L - 1)
        b = a + L
        
        # The segment is chain[a:b]
        # The neighbors are chain[a-1] and chain[b]
        left_neighbor = chain[a-1]
        right_neighbor = chain[b]
        
        # Calculate Manhattan distance
        md = abs(right_neighbor[0] - left_neighbor[0]) + abs(right_neighbor[1] - left_neighbor[1])
        
        # Check if a path of length L is possible
        if md > L or (L - md) % 2 != 0:
            continue
            
        # Remove segment
        rest_chain = chain[:a] + chain[b:]
        rest_occupied = set(rest_chain)
        
        # We need to find a path of length L from left_neighbor to right_neighbor
        # that does not intersect rest_occupied (except endpoints which are in rest_occupied)
        # Note: left_neighbor and right_neighbor are in rest_occupied.
        
        def _find_path(start, end, length, occupied_set):
            # DFS with pruning
            if length == 0:
                return [start] if start == end else None
            
            def _dfs(current, remaining, path, occupied):
                if remaining == 0:
                    if current == end:
                        return list(path)
                    return None
                x, y = current
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    site = (x + dx, y + dy)
                    # Pruning: must be able to reach end in remaining steps
                    dist_to_end = abs(end[0] - site[0]) + abs(end[1] - site[1])
                    if dist_to_end > remaining or (remaining - dist_to_end) % 2 != 0:
                        continue
                    
                    # Check occupancy
                    # If site is start or end, it's allowed (it's in occupied_set but we are visiting it)
                    # Actually, start is already in path. 
                    # If site is in occupied_set and not end, it's blocked.
                    if site in occupied_set and site != end:
                        continue
                    
                    # Wait, if site is start, it's in path. 
                    # We are moving FROM current TO site.
                    # site must not be in occupied_set UNLESS it is the target 'end'.
                    # Also site must not be in current path (self-avoidance within path).
                    # Since path is small, checking `site in path` is fine.
                    
                    if site in path:
                        continue

                    path.append(site)
                    # We don't need to add to occupied_set for the sake of the global search 
                    # because we are only checking against `occupied_set` (the rest of the chain)
                    # and `path` (the new path).
                    # However, we must ensure we don't hit the rest of the chain.
                    # `site in occupied_set` check above handles that.
                    
                    result = _dfs(site, remaining - 1, path, occupied_set)
                    if result is not None:
                        return result
                    path.pop()
                return None

            return _dfs(start, length, [start], occupied_set)

        # The path length in terms of steps is L.
        # The number of vertices in the new segment is L.
        # The path function returns a list of vertices.
        # We need a path of L vertices connecting left_neighbor and right_neighbor.
        # Steps = L - 1.
        
        path = _find_path(left_neighbor, right_neighbor, L, rest_occupied)
        
        if path is None:
            continue
            
        if path[0] != left_neighbor or path[-1] != right_neighbor:
            continue
            
        # The new segment is path[1:-1] (excluding endpoints which are already in rest_chain)
        new_segment = path[1:-1]
        
        # Construct new chain
        new_chain = chain[:a] + new_segment + chain[b:]
        
        # Validate
        if len(new_chain) != n:
            continue
            
        # Check self-avoidance and connectivity
        # Connectivity is guaranteed by construction (path connects neighbors)
        # Self-avoidance:
        # - New segment doesn't intersect rest_chain (checked in _find_path)
        # - New segment is self-avoiding (checked in _dfs via `site in path`)
        # - Endpoints are shared correctly.
        
        new_energy = _energy(sequence, new_chain)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_chain = new_chain
            # If we found an improvement, we can restart the loop or continue
            # For efficiency, let's just accept and continue to try more moves in next call
            # Or break? The original code returns immediately on improvement in local search?
            # No, local search does max_steps.
            # Here we do max_attempts.
            # Let's just keep looking for better ones within the attempts.
            
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Fold an HP sequence on the square lattice.
    Uses beam search growth followed by local search and segment relocation refinement.
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
    max_iterations = 5000
    iterations = 0
    
    while _time_remaining() > 5.0 and iterations < max_iterations:
        iterations += 1
        
        # Generate a new conformation
        # Alternate between beam width 5 and 8 for diversity?
        # Or just use 8 as it's better.
        beam_width = 8
        if iterations % 2 == 0:
            beam_width = 5
            
        chain = _grow_beam(sequence, dimension, rng, beam_width=beam_width)
        if chain is None:
            continue
        
        # Refine with local search
        # Limit local search steps to ensure we can do many restarts
        chain, energy = _local_search(sequence, chain, rng, max_steps=50)
        
        # Try segment relocation if time permits
        if _time_remaining() > 10.0:
            chain, energy = _segment_relocation(sequence, chain, rng, max_attempts=2, max_seg_len=6)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
            
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
