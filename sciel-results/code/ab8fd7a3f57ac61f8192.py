import time
import random

N = 9
MAX_TIME = 115.0  # Safety margin before 120s/150s kill

# Precompute bit counts for all possible XOR results (0 to 2^9 - 1)
_BIT_COUNT = [i.bit_count() for i in range(1 << N)]
# Precompute neighbors for all 512 vertices
_NEIGHBORS = [[v ^ (1 << b) for b in range(N)] for v in range(1 << N)]

def _greedy_search(seed_offset=0, depth_limit=1000):
    """
    Optimized Greedy search using Forbidden Mask.
    """
    rng = random.Random(seed_offset)
    path = [0]
    # Forbidden mask: vertices that cannot be added next.
    # Forbidden(P) = {x | x in used(P) or dist(x, P[:-1]) == 1}
    # Initially P=[0], Forbidden={0}
    forbidden = [False] * (1 << N)
    forbidden[0] = True
    
    current = 0
    
    while len(path) < depth_limit:
        valid_neighbors = []
        for v in _NEIGHBORS[current]:
            if not forbidden[v]:
                valid_neighbors.append(v)
                
        if not valid_neighbors:
            break
            
        # Lookahead: score each neighbor by how many valid neighbors it has
        scored_neighbors = []
        for v in valid_neighbors:
            # If we pick v, the new path is path + [v].
            # The new forbidden set for the NEXT step will be:
            # Forbidden_new = Forbidden_old U Neighbors(current) U {v}
            
            # We want to count u in Neighbors(v) such that u is NOT in Forbidden_new.
            # u is in Forbidden_new if:
            # 1. forbidden[u] is True (from Forbidden_old)
            # 2. u is in Neighbors(current) (added to Forbidden_new)
            # 3. u == v (added to Forbidden_new, but u is neighbor of v so u!=v)
            
            count = 0
            for u in _NEIGHBORS[v]:
                if not forbidden[u] and _BIT_COUNT[u ^ current] != 1:
                    count += 1
            scored_neighbors.append((count, v))
            
        if not scored_neighbors:
            break

        # Sort by count descending, pick randomly from top ties
        max_count = max(c for c, v in scored_neighbors)
        candidates = [v for c, v in scored_neighbors if c == max_count]
        next_v = rng.choice(candidates)
        
        path.append(next_v)
        
        # Update Forbidden Mask
        # The 'current' variable is still the OLD current.
        # We need to add Neighbors(old_current) and next_v to forbidden.
        for u in _NEIGHBORS[current]:
            forbidden[u] = True
        forbidden[next_v] = True
        
        current = next_v
        
    return path

def _beam_search(seed_offset=0, beam_width=20, depth_limit=200):
    """
    Optimized Beam Search using Forbidden Mask.
    """
    rng = random.Random(seed_offset)
    # Beam entry: (path, forbidden_list, current)
    # forbidden_list is a list of bools of size 2^N
    
    init_forbidden = [False] * (1 << N)
    init_forbidden[0] = True
    
    beams = [([0], init_forbidden, 0)]
    
    for depth in range(depth_limit):
        new_beams = []
        for path, forbidden, current in beams:
            for v in _NEIGHBORS[current]:
                if not forbidden[v]:
                    # Create new state
                    new_path = path + [v]
                    new_forbidden = forbidden[:] # Copy list
                    # Update new_forbidden:
                    # Add neighbors of current (old current)
                    for u in _NEIGHBORS[current]:
                        new_forbidden[u] = True
                    # Add v
                    new_forbidden[v] = True
                    
                    new_beams.append((new_path, new_forbidden, v))
                    
        if not new_beams:
            break
            
        # Score and prune
        # Score: number of valid extensions (lookahead 1)
        scored_beams = []
        for path, forbidden, current in new_beams:
            count = 0
            # Valid neighbors of current under current forbidden state
            # The 'forbidden' list in new_beams is ALREADY updated for the next step.
            # So we just count neighbors of 'current' that are not forbidden.
            for v in _NEIGHBORS[current]:
                if not forbidden[v]:
                    count += 1
            scored_beams.append((count, path, forbidden, current))
            
        # Sort by count
        scored_beams.sort(key=lambda x: x[0], reverse=True)
        
        # Keep top beam_width
        beams = [(p, f, c) for _, p, f, c in scored_beams[:beam_width]]
        
        if not beams:
            break
            
    if not beams:
        return [0]
    
    best = max(beams, key=lambda x: len(x[0]))
    return best[0]

def construct():
    start_time = time.time()
    best_path = [0]
    best_len = 1
    
    # Initial deterministic greedy
    path = [0]
    forbidden = [False] * (1 << N)
    forbidden[0] = True
    current = 0
    while True:
        nxt = None
        for v in _NEIGHBORS[current]:
            if not forbidden[v]:
                nxt = v
                break
        if nxt is None:
            break
        path.append(nxt)
        # Update forbidden
        for u in _NEIGHBORS[current]:
            forbidden[u] = True
        forbidden[nxt] = True
        current = nxt
        
    best_path = path
    best_len = len(best_path)
    
    # Randomized greedy searches
    i = 0
    while time.time() - start_time < MAX_TIME:
        p = _greedy_search(i)
        if len(p) > best_len:
            best_path = p
            best_len = len(p)
        i += 1
        
        # Occasionally try beam search
        if i % 5 == 0 and time.time() - start_time < MAX_TIME - 10:
            bp = _beam_search(i, beam_width=20)
            if len(bp) > best_len:
                best_path = bp
                best_len = len(bp)
        
    return best_path
