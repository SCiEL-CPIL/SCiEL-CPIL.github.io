import time
import random

N = 9
MAX_TIME = 145.0  # Safety margin before 150s kill

# Precompute bit counts for all possible XOR results (0 to 2^9 - 1)
_BIT_COUNT = [i.bit_count() for i in range(1 << N)]
# Precompute neighbors for all 512 vertices
_NEIGHBORS = [[v ^ (1 << b) for b in range(N)] for v in range(1 << N)]
# Precompute neighbor bitmask for each vertex (bit v is set if v is a neighbor)
_NEIGHBOR_MASK = [0] * (1 << N)
for v in range(1 << N):
    mask = 0
    for b in range(N):
        mask |= 1 << (v ^ (1 << b))
    _NEIGHBOR_MASK[v] = mask

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
            # If we pick v, the new path is path + [v].
            # The new forbidden set for the NEXT step will be:
            # Forbidden_new = Forbidden_old | _NEIGHBOR_MASK[current] | (1 << v)
            
            # We want to count u in Neighbors(v) such that u is NOT in Forbidden_new.
            # Valid candidates: bits in _NEIGHBOR_MASK[v] that are NOT in Forbidden_new
            forbidden_new = forbidden | _NEIGHBOR_MASK[current] | (1 << v)
            valid_mask = _NEIGHBOR_MASK[v] & ~forbidden_new
            count = valid_mask.bit_count()
            scored_neighbors.append((count, v))
            
        if not scored_neighbors:
            break

        # Sort by count descending, pick randomly from top ties
        max_count = max(c for c, v in scored_neighbors)
        candidates = [v for c, v in scored_neighbors if c == max_count]
        next_v = rng.choice(candidates)
        
        path.append(next_v)
        
        # Update Forbidden Mask
        forbidden |= _NEIGHBOR_MASK[current]
        forbidden |= (1 << next_v)
        
        current = next_v
        
    return path

def _warnsdorff_search(seed_offset=0, start_time=0.0, time_budget=140.0):
    """
    Warnsdorff-style backtracking search for induced paths.
    Uses the "fewest remaining moves" heuristic with time-limited backtracking.
    """
    rng = random.Random(seed_offset)
    path = [0]
    forbidden = 1  # vertex 0 forbidden
    current = 0
    
    best_path = list(path)
    best_len = 1
    node_count = 0
    deadline = start_time + time_budget
    
    def extend(current, forbidden, path, depth):
        nonlocal best_path, best_len, node_count
        
        node_count += 1
        if node_count % 10000 == 0 and time.time() > deadline:
            return False
        
        if len(path) > best_len:
            best_path = list(path)
            best_len = len(path)
        
        # Get valid neighbors
        valid_neighbors = []
        for v in _NEIGHBORS[current]:
            if not (forbidden >> v) & 1:
                valid_neighbors.append(v)
                
        if not valid_neighbors:
            return False
            
        # Warnsdorff: order by fewest valid extensions (to avoid dead ends)
        # Use 2-level lookahead for better heuristic
        scored = []
        for v in valid_neighbors:
            new_forbidden = forbidden | _NEIGHBOR_MASK[current] | (1 << v)
            valid_mask = _NEIGHBOR_MASK[v] & ~new_forbidden
            count1 = valid_mask.bit_count()
            
            # 2-level lookahead: for the best continuation from v,
            # estimate how far we can go
            if count1 > 0 and count1 <= 3:  # Only do deep lookahead for tight positions
                # Find the best next move from v
                best_next_count = 0
                for u in _NEIGHBORS[v]:
                    if not (new_forbidden >> u) & 1:
                        ff = new_forbidden | _NEIGHBOR_MASK[v] | (1 << u)
                        valid_mask2 = _NEIGHBOR_MASK[u] & ~ff
                        c2 = valid_mask2.bit_count()
                        if c2 > best_next_count:
                            best_next_count = c2
                # Penalize if the best continuation is also tight
                score = count1 * 10 + best_next_count
            else:
                score = count1 * 10
            
            scored.append((score, v))
            
        # Sort by score ascending (Warnsdorff: fewest first)
        scored.sort(key=lambda x: x[0])
        
        # Try each neighbor in order
        for score, v in scored:
            if time.time() > deadline:
                return False
                
            new_forbidden = forbidden | _NEIGHBOR_MASK[current] | (1 << v)
            path.append(v)
            
            if extend(v, new_forbidden, path, depth + 1):
                path.pop()
                return True
                
            path.pop()
            
        return False
    
    extend(current, forbidden, path, 0)
    
    return best_path

def _try_local_improve(path, start_time=0.0, time_budget=140.0):
    """
    Try to extend or splice the path to make it longer.
    """
    deadline = start_time + time_budget
    best = list(path)
    best_len = len(best)
    
    # Try extending from both ends
    for end in [0, 1]:
        if time.time() > deadline:
            break
        if end == 0:
            # Try to prepend
            current = best[0]
            forbidden = 0
            for v in best:
                forbidden |= _NEIGHBOR_MASK[v]
            # Remove current from forbidden (we can go to its neighbors)
            forbidden &= ~(1 << current)
            
            for v in _NEIGHBORS[current]:
                if not (forbidden >> v) & 1:
                    new_path = [v] + best
                    # Verify it's valid
                    valid = True
                    for i in range(len(new_path)):
                        for j in range(i+2, len(new_path)):
                            if _BIT_COUNT[new_path[i] ^ new_path[j]] < 2:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        best = new_path
                        best_len = len(best)
                        break
        else:
            # Try to append
            current = best[-1]
            forbidden = 0
            for v in best:
                forbidden |= _NEIGHBOR_MASK[v]
            forbidden &= ~(1 << current)
            
            for v in _NEIGHBORS[current]:
                if not (forbidden >> v) & 1:
                    new_path = best + [v]
                    # Verify it's valid
                    valid = True
                    for i in range(len(new_path)):
                        for j in range(i+2, len(new_path)):
                            if _BIT_COUNT[new_path[i] ^ new_path[j]] < 2:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        best = new_path
                        best_len = len(best)
                        break
    
    return best

def construct():
    start_time = time.time()
    best_path = [0]
    best_len = 1
    
    # Initial deterministic greedy
    path = [0]
    forbidden = 1  # vertex 0 forbidden
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
        # Update forbidden
        forbidden |= _NEIGHBOR_MASK[current]
        forbidden |= (1 << nxt)
        current = nxt
        
    best_path = path
    best_len = len(best_path)
    
    # Try Warnsdorff backtracking searches with different seeds
    i = 0
    while time.time() - start_time < MAX_TIME - 5:
        elapsed = time.time() - start_time
        remaining = MAX_TIME - 5 - elapsed
        
        # Use more time for later searches as we get better heuristics
        bt_budget = min(remaining * 0.8, 30.0) if i < 10 else min(remaining * 0.9, 50.0)
        
        p = _warnsdorff_search(i, start_time=start_time, time_budget=bt_budget)
        if len(p) > best_len:
            # Try local improvement
            p = _try_local_improve(p, start_time=start_time, time_budget=min(5.0, remaining * 0.1))
            if len(p) > best_len:
                best_path = p
                best_len = len(p)
            else:
                best_path = p
                best_len = len(p)
        i += 1
        
        # Also try greedy to maintain diversity
        if i % 3 == 0:
            gp = _greedy_search(i)
            if len(gp) > best_len:
                best_path = gp
                best_len = len(gp)
        
        # Periodically try local improvement on best path
        if i % 5 == 0 and best_len > 50:
            improved = _try_local_improve(best_path, start_time=start_time, time_budget=min(3.0, remaining * 0.05))
            if len(improved) > best_len:
                best_path = improved
                best_len = len(improved)
        
        # Check time
        if time.time() - start_time > MAX_TIME - 2:
            break
    
    # Final local improvement pass
    if time.time() - start_time < MAX_TIME - 1:
        improved = _try_local_improve(best_path, start_time=start_time, time_budget=min(3.0, MAX_TIME - 1 - (time.time() - start_time)))
        if len(improved) > best_len:
            best_path = improved
            best_len = len(improved)
    
    return best_path
