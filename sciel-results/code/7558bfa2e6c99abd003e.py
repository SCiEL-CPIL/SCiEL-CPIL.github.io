import time
import random

N = 9
MAX_TIME = 115.0  # Safety margin before 120s/150s kill

# Precompute bit counts for all possible XOR results (0 to 2^9 - 1)
_BIT_COUNT = [i.bit_count() for i in range(1 << N)]
# Precompute neighbors for all 512 vertices
_NEIGHBORS = [[v ^ (1 << b) for b in range(N)] for v in range(1 << N)]

# Precompute neighbor masks for faster forbidden checks
_NEIGHBOR_MASKS = [0] * (1 << N)
for v in range(1 << N):
    mask = 0
    for u in _NEIGHBORS[v]:
        mask |= (1 << u)
    _NEIGHBOR_MASKS[v] = mask

def _greedy_search(seed_offset=0, depth_limit=1000):
    """
    Optimized Greedy search using Forbidden Mask (bitmask).
    """
    rng = random.Random(seed_offset)
    path = [0]
    # Forbidden mask as a bitmask: bit i is set if vertex i is forbidden
    forbidden = 1  # vertex 0 is forbidden
    current = 0
    
    while len(path) < depth_limit:
        valid_neighbors = []
        for v in _NEIGHBORS[current]:
            if not (forbidden >> v) & 1:
                valid_neighbors.append(v)
                
        if not valid_neighbors:
            break
            
        # Lookahead: score each neighbor by how many valid neighbors it has
        scored_neighbors = []
        for v in valid_neighbors:
            # If we pick v, the new forbidden will include neighbors of current and v itself
            # Count neighbors u of v that are NOT in (forbidden | neighbors_of_current | {v})
            count = 0
            for u in _NEIGHBORS[v]:
                if not (forbidden >> u) & 1 and _BIT_COUNT[u ^ current] != 1:
                    count += 1
            scored_neighbors.append((count, v))
            
        if not scored_neighbors:
            break

        # Sort by count descending, pick randomly from top ties
        max_count = max(c for c, v in scored_neighbors)
        candidates = [v for c, v in scored_neighbors if c == max_count]
        next_v = rng.choice(candidates)
        
        path.append(next_v)
        
        # Update Forbidden Mask: add neighbors of current and next_v
        forbidden |= _NEIGHBOR_MASKS[current]
        forbidden |= (1 << next_v)
        
        current = next_v
        
    return path

def _warnsdorff_search(seed_offset=0, max_depth=200, time_limit=5.0):
    """
    Warnsdorff-style backtracking: prefer vertices with fewest valid extensions.
    This is more effective for finding long induced paths.
    """
    rng = random.Random(seed_offset)
    start_time = time.time()
    
    best_path = [0]
    best_len = 1
    
    def dfs(path, forbidden, current, depth):
        nonlocal best_path, best_len
        nonlocal start_time
        
        if time.time() - start_time > time_limit:
            return
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
        
        if depth >= max_depth:
            return
        
        # Get valid neighbors with Warnsdorff scoring (fewest extensions first)
        valid_neighbors = []
        for v in _NEIGHBORS[current]:
            if not (forbidden >> v) & 1:
                # Count how many valid neighbors v would have after moving there
                count = 0
                for u in _NEIGHBORS[v]:
                    if not (forbidden >> u) & 1 and _BIT_COUNT[u ^ current] != 1:
                        count += 1
                valid_neighbors.append((count, v))
        
        if not valid_neighbors:
            return
        
        # Warnsdorff: sort by count ascending (prefer fewest options)
        valid_neighbors.sort(key=lambda x: x[0])
        
        # Add some randomness to avoid getting stuck in local patterns
        top_k = min(5, len(valid_neighbors))
        top_candidates = valid_neighbors[:top_k]
        if rng.random() < 0.3:
            rng.shuffle(top_candidates)
        
        for count, v in top_candidates:
            new_forbidden = forbidden | _NEIGHBOR_MASKS[current] | (1 << v)
            
            path.append(v)
            dfs(path, new_forbidden, v, depth + 1)
            path.pop()
            
            if time.time() - start_time > time_limit:
                return
    
    dfs([0], 1, 0, 0)
    return best_path

def _deep_greedy_with_backtrack(seed_offset=0):
    """
    Hybrid: greedy with occasional backtracking to escape dead ends.
    """
    rng = random.Random(seed_offset)
    path = [0]
    forbidden = 1
    current = 0
    backtrack_stack = []  # stack of (path_copy, forbidden_copy, current) for backtracking
    
    while True:
        valid_neighbors = []
        for v in _NEIGHBORS[current]:
            if not (forbidden >> v) & 1:
                count = 0
                for u in _NEIGHBORS[v]:
                    if not (forbidden >> u) & 1 and _BIT_COUNT[u ^ current] != 1:
                        count += 1
                valid_neighbors.append((count, v))
        
        if not valid_neighbors:
            # Dead end: try to backtrack
            if backtrack_stack:
                path, forbidden, current = backtrack_stack.pop()
            else:
                break
            continue
        
        # Warnsdorff: sort by count ascending (prefer fewest options)
        valid_neighbors.sort(key=lambda x: x[0])
        
        # With some probability, save state for backtracking and try a suboptimal choice
        if len(valid_neighbors) > 1 and rng.random() < 0.3:
            # Save current state for potential backtracking
            backtrack_stack.append((path[:], forbidden, current))
            # Cap stack size
            if len(backtrack_stack) > 20:
                backtrack_stack.pop(0)
            # Pick a random choice from top 3
            top_k = min(3, len(valid_neighbors))
            idx = rng.randint(0, top_k - 1)
            count, next_v = valid_neighbors[idx]
        else:
            count, next_v = valid_neighbors[0]
        
        path.append(next_v)
        forbidden |= _NEIGHBOR_MASKS[current]
        forbidden |= (1 << next_v)
        current = next_v
        
        if len(path) > 200:
            break
    
    return path

def construct():
    start_time = time.time()
    best_path = [0]
    best_len = 1
    
    # Initial deterministic greedy
    path = [0]
    forbidden = 1
    current = 0
    while True:
        nxt = None
        for v in _NEIGHBORS[current]:
            if not (forbidden >> v) & 1:
                nxt = v
                break
        if nxt is None:
            break
        path.append(nxt)
        forbidden |= _NEIGHBOR_MASKS[current]
        forbidden |= (1 << nxt)
        current = nxt
        
    best_path = path
    best_len = len(best_path)
    
    # Randomized searches with focus on Warnsdorff
    i = 0
    while time.time() - start_time < MAX_TIME:
        # Primary: Warnsdorff backtracking (most effective strategy)
        if time.time() - start_time < MAX_TIME - 8:
            bp = _warnsdorff_search(i, max_depth=150, time_limit=4.0)
            if len(bp) > best_len:
                best_path = bp
                best_len = len(bp)
        
        # Secondary: randomized greedy
        p = _greedy_search(i)
        if len(p) > best_len:
            best_path = p
            best_len = len(p)
        i += 1
        
        # Tertiary: deep greedy with backtrack
        if i % 4 == 0 and time.time() - start_time < MAX_TIME - 10:
            bp = _deep_greedy_with_backtrack(i)
            if len(bp) > best_len:
                best_path = bp
                best_len = len(bp)
        
        # Check time more frequently
        if time.time() - start_time > MAX_TIME:
            break
        
    return best_path
