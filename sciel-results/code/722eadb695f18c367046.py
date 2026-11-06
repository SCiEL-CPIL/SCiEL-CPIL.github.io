"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 9
TOTAL_VERTICES = 1 << N

# Precompute neighbors for each vertex
_NEIGHBORS = [None] * TOTAL_VERTICES
for _v in range(TOTAL_VERTICES):
    _NEIGHBORS[_v] = [_v ^ (1 << b) for b in range(N)]

# Precompute bit masks for quick adjacency checks
# adj_mask[v] = bitmask of all vertices adjacent to v
_ADJ_MASK = [0] * TOTAL_VERTICES
for _v in range(TOTAL_VERTICES):
    _m = 0
    for _nb in _NEIGHBORS[_v]:
        _m |= (1 << _nb)
    _ADJ_MASK[_v] = _m


def _greedy_seed():
    """Generate a greedy snake as initial seed."""
    path, used = [0], {0}
    used_mask = 1
    while True:
        cur = path[-1]
        nxt = None
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            # Check if v is adjacent to any non-consecutive vertex
            adj_to_used = _ADJ_MASK[v] & (used_mask ^ (1 << v))
            # Remove the previous vertex from check (it's consecutive)
            adj_to_used &= ~(1 << cur)
            if adj_to_used:
                continue
            nxt = v
            break
        if nxt is None:
            break
        path.append(nxt)
        used.add(nxt)
        used_mask |= (1 << nxt)
    return path


def _backtrack_extend(path, used_mask, deadline, max_depth):
    """
    Constraint-based backtracking with MRV heuristic.
    Tries to extend the path by finding a valid next vertex,
    using backtracking when stuck.
    
    Returns the extended path.
    """
    cur = path[-1]
    
    # Get candidate vertices: unvisited neighbors of current
    candidates = []
    for v in _NEIGHBORS[cur]:
        if used_mask & (1 << v):
            continue
        # Check induced constraint: v must not be adjacent to any path vertex except cur
        # (since v is adjacent to cur, we check against all others)
        conflict = _ADJ_MASK[v] & (used_mask ^ (1 << v) ^ (1 << cur))
        if conflict:
            continue
        candidates.append(v)
    
    if not candidates:
        return path
    
    # MRV heuristic: pick candidate that minimizes future branching factor
    # For each candidate, count how many valid extensions it would allow
    def count_future_options(v):
        """Count valid next vertices after adding v to path."""
        new_used = used_mask | (1 << v)
        count = 0
        for nv in _NEIGHBORS[v]:
            if new_used & (1 << nv):
                continue
            # nv adjacent to v (consecutive), check against rest
            conflict = _ADJ_MASK[nv] & (new_used ^ (1 << nv) ^ (1 << v))
            if not conflict:
                count += 1
        return count
    
    # Sort candidates by future options (ascending - MRV)
    candidates.sort(key=count_future_options)
    
    # Try candidates in order with limited backtracking
    for v in candidates:
        if time.time() > deadline:
            return path
        if len(path) >= max_depth:
            break
        
        new_path = path + [v]
        new_used = used_mask | (1 << v)
        
        # Recursively extend (with depth limit to control time)
        extended = _backtrack_extend(new_path, new_used, deadline, max_depth)
        if len(extended) > len(path):
            return extended
    
    return path


def _local_search(path, used_mask, deadline, max_iter=200):
    """
    Local search: try to improve path by finding alternative extensions
    at various positions using backtracking.
    """
    best_path = list(path)
    best_len = len(best_path)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        # Pick a random position to try re-extending from
        if len(best_path) <= 2:
            break
        pos = random.randint(1, len(best_path) - 2)
        
        # Truncate path at position pos
        sub_path = best_path[:pos + 1]
        sub_used = 0
        for v in sub_path:
            sub_used |= (1 << v)
        
        # Try to extend from this point
        extended = _backtrack_extend(sub_path, sub_used, deadline, len(best_path) + 5)
        
        if len(extended) > best_len:
            best_path = extended
            best_len = len(best_path)
            # Recompute used mask
            sub_used = 0
            for v in best_path:
                sub_used |= (1 << v)
            used_mask = sub_used
    
    return best_path


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    # Phase 1: Generate greedy seed
    path = _greedy_seed()
    used_mask = 0
    for v in path:
        used_mask |= (1 << v)
    
    best_path = list(path)
    best_len = len(best_path)
    
    # Phase 2: Try multiple random restarts with different greedy orders
    attempts = 0
    while time.time() < deadline - 10.0:
        attempts += 1
        
        # Generate a new greedy seed with randomized bit order
        bit_order = list(range(N))
        random.shuffle(bit_order)
        
        path2, used2 = [0], {0}
        used_mask2 = 1
        while True:
            cur = path2[-1]
            nxt = None
            for b in bit_order:
                v = cur ^ (1 << b)
                if v in used2:
                    continue
                adj_to_used = _ADJ_MASK[v] & (used_mask2 ^ (1 << v))
                adj_to_used &= ~(1 << cur)
                if adj_to_used:
                    continue
                nxt = v
                break
            if nxt is None:
                break
            path2.append(nxt)
            used2.add(nxt)
            used_mask2 |= (1 << nxt)
        
        if len(path2) > best_len:
            best_path = path2
            best_len = len(path2)
            used_mask = used_mask2
        
        # Phase 3: Backtracking extension on the best path
        if time.time() < deadline - 5.0:
            extended = _backtrack_extend(best_path, used_mask, deadline, best_len + 20)
            if len(extended) > best_len:
                best_path = extended
                best_len = len(extended)
                used_mask = 0
                for v in best_path:
                    used_mask |= (1 << v)
        
        # Phase 4: Local search improvements
        if time.time() < deadline - 3.0:
            improved = _local_search(best_path, used_mask, deadline, max_iter=50)
            if len(improved) > best_len:
                best_path = improved
                best_len = len(improved)
                used_mask = 0
                for v in best_path:
                    used_mask |= (1 << v)
    
    return best_path
# EVOLVE-BLOCK-END
