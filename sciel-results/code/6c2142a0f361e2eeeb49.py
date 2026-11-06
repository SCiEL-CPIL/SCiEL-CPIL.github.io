"""Snake-in-the-box in Q_9: improved construction via heuristic search."""
import random
import sys

N = 9
TOTAL_VERTICES = 1 << N

# EVOLVE-BLOCK-START

import time

def _precompute_neighbors():
    """Precompute neighbors and neighbor masks for all vertices."""
    neighbors = [[] for _ in range(TOTAL_VERTICES)]
    neighbor_masks = [0] * TOTAL_VERTICES
    for v in range(TOTAL_VERTICES):
        for b in range(N):
            nb = v ^ (1 << b)
            neighbors[v].append(nb)
            neighbor_masks[v] |= (1 << nb)
    return neighbors, neighbor_masks

# Global precomputed data
_NEIGHBORS, _NEIGHBOR_MASKS = _precompute_neighbors()

def _dfs_snake(start, time_limit=110.0):
    """
    Perform a depth-first search to find a long snake.
    Uses bitmask propagation for fast constraint checking.
    """
    best_path = []
    start_time = time.time()
    
    # State variables for recursion
    # To avoid recursion depth issues and overhead, we use an explicit stack or careful recursion.
    # Python recursion limit is usually 1000, which is fine for path length < 1000.
    # We will use recursion for clarity, but with a time check.
    
    path = [start]
    used_mask = 1 << start
    banned_mask = 0
    current = start
    prev = -1 # Not used for first step
    
    # We need to track the path to return it.
    # We'll use a list `path` that is modified in-place.
    
    def dfs(cur, prev, used, banned, depth):
        nonlocal best_path, start_time
        
        if time.time() - start_time > time_limit:
            return
        
        if depth > len(best_path):
            best_path = list(path)
        
        # Find candidates
        # Candidates must be neighbors of cur
        # Not in used
        # Not in banned
        cand_mask = _NEIGHBOR_MASKS[cur] & ~used & ~banned
        
        if cand_mask == 0:
            return
        
        # Heuristic: Order candidates by how many options they leave?
        # Or just iterate. For performance, iterating bits is fast.
        # We can try to pick the candidate that minimizes the number of future options (MRV)
        # or maximizes them. Let's try MRV (fewest options) as it's standard for CSP.
        
        candidates = []
        cm = cand_mask
        while cm:
            lsb = cm & -cm
            v = lsb.bit_length() - 1
            # Count available neighbors for v (excluding used and banned, and cur)
            # Actually, we just need to check if it's a good move.
            # Let's just collect them and sort by heuristic if list is small.
            candidates.append(v)
            cm ^= lsb
            
        # Simple heuristic: Try vertices with fewer remaining neighbors first (MRV)
        # This helps find dead ends earlier.
        def heuristic_score(v):
            # Number of valid next steps from v
            next_cand = _NEIGHBOR_MASKS[v] & ~used & ~banned & ~(1 << cur)
            return next_cand.bit_count()
            
        candidates.sort(key=heuristic_score)
        
        for v in candidates:
            if time.time() - start_time > time_limit:
                return
            
            # Update state
            path.append(v)
            new_used = used | (1 << v)
            # The vertex that becomes "banned source" is `cur` (the one we are leaving)
            # Because for the next step, `cur` will be non-consecutive to the new vertex.
            new_banned = banned | _NEIGHBOR_MASKS[cur]
            
            dfs(v, cur, new_used, new_banned, depth + 1)
            
            path.pop()
            
            if time.time() - start_time > time_limit:
                return

    dfs(start, -1, used_mask, banned_mask, 1)
    return best_path

def construct():
    """
    Construct a snake in Q_9 using a constraint-satisfaction style backtracking search.
    """
    # Run multiple searches with different random starts or orders if needed.
    # Since the graph is symmetric, starting at 0 is fine, but the *order* of exploration matters.
    # Our DFS uses a heuristic (MRV) which is deterministic.
    # To explore different parts of the search tree, we can randomize the tie-breaking in the heuristic.
    
    best_path = []
    time_limit = 115.0 # Leave 5s buffer
    start_time = time.time()
    
    # Try a few different random seeds for tie-breaking to diversify search
    for seed in range(10):
        if time.time() - start_time > time_limit:
            break
            
        # We can modify the heuristic slightly by randomizing the candidate order on ties
        # But for now, the MRV heuristic is strong.
        # Let's just run the DFS. If it's deterministic, it will return the same path.
        # To get different paths, we need to change the search order.
        
        # We can introduce randomness in the candidate selection.
        # Let's create a modified DFS that uses random tie-breaking.
        
        def dfs_randomized(cur, prev, used, banned, depth, rng):
            nonlocal best_path
            
            if time.time() - start_time > time_limit:
                return
            
            if depth > len(best_path):
                best_path = list(path)
            
            cand_mask = _NEIGHBOR_MASKS[cur] & ~used & ~banned
            if cand_mask == 0:
                return
            
            candidates = []
            cm = cand_mask
            while cm:
                lsb = cm & -cm
                v = lsb.bit_length() - 1
                candidates.append(v)
                cm ^= lsb
            
            # Randomize order to explore different branches
            rng.shuffle(candidates)
            
            # Apply MRV heuristic to sort, but keep some randomness?
            # Or just use MRV. MRV is usually better for finding *any* solution.
            # For *longest*, we might want to avoid dead ends.
            # Let's stick to MRV for speed, but randomize start vertices if we run out of time.
            
            def heuristic_score(v):
                next_cand = _NEIGHBOR_MASKS[v] & ~used & ~banned & ~(1 << cur)
                return next_cand.bit_count()
                
            candidates.sort(key=heuristic_score)
            
            for v in candidates:
                if time.time() - start_time > time_limit:
                    return
                
                path.append(v)
                new_used = used | (1 << v)
                new_banned = banned | _NEIGHBOR_MASKS[cur]
                
                dfs_randomized(v, cur, new_used, new_banned, depth + 1, rng)
                
                path.pop()
                
                if time.time() - start_time > time_limit:
                    return

        path = [0]
        used_mask = 1
        banned_mask = 0
        rng = random.Random(seed)
        dfs_randomized(0, -1, used_mask, banned_mask, 1, rng)

    return best_path

# EVOLVE-BLOCK-END
