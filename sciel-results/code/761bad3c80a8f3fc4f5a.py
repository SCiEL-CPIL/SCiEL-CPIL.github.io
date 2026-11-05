import time
import random
import numpy as np
import itertools

def _is_prime(p):
    if p < 2:
        return False
    if p % 2 == 0:
        return p == 2
    for i in range(3, int(np.sqrt(p)) + 1, 2):
        if p % i == 0:
            return False
    return True

def _paley_orientation(p):
    """Generate orientation bits for a Paley tournament of prime order p (p ≡ 3 mod 4)."""
    n = p
    squares = np.zeros(p, dtype=bool)
    for x in range(1, p):
        squares[(x * x) % p] = True
    
    orientation = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % p
            if squares[diff]:
                orientation.append(0)
            else:
                orientation.append(1)
    
    return n, orientation

def _build_masks_from_adj(adj, n):
    """Build in-neighborhood bitmasks from adjacency matrix."""
    masks = [0] * n
    for v in range(n):
        mask = 0
        for u in range(n):
            if u != v and adj[u][v]:
                mask |= (1 << u)
        masks[v] = mask
    return masks

def _count_bad_4sets(n, masks):
    """Count number of 4-sets without a common dominator using bitmask intersection."""
    bad = 0
    for a, b, c, d in itertools.combinations(range(n), 4):
        inter = masks[a] & masks[b] & masks[c] & masks[d]
        if inter == 0:
            bad += 1
    return bad

def _check_valid_fast(n, masks):
    """Check if every 4-set has a common dominator."""
    for a, b, c, d in itertools.combinations(range(n), 4):
        if (masks[a] & masks[b] & masks[c] & masks[d]) == 0:
            return False
    return True

def _local_search(n, max_time, seed=42):
    """
    Local search to find a valid tournament on n vertices.
    Uses bitmask-based property checking for efficiency.
    Returns (orientation, valid).
    """
    rng = random.Random(seed)
    start = time.time()
    deadline = start + max_time
    
    if n < 4:
        return None, False
    
    num_edges = n * (n - 1) // 2
    
    # Start with random tournament
    orientation = [rng.randint(0, 1) for _ in range(num_edges)]
    
    # Build adjacency matrix
    adj = np.zeros((n, n), dtype=bool)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if orientation[idx] == 0:
                adj[i][j] = True
            else:
                adj[j][i] = True
            idx += 1
    
    masks = _build_masks_from_adj(adj, n)
    current_bad = _count_bad_4sets(n, masks)
    
    if current_bad == 0:
        return orientation, True
    
    best_orientation = orientation[:]
    best_bad = current_bad
    
    # Precompute all 4-sets for repeated use
    all_4sets = list(itertools.combinations(range(n), 4))
    
    # For efficient local search, we need to quickly evaluate the effect of flipping an edge.
    # When we flip edge (i,j), the masks change for all vertices in 4-sets containing i or j.
    # This is complex, so for n <= 40, we do full recount.
    
    if n > 40:
        # For larger n, do limited random restarts
        for attempt in range(30):
            if time.time() > deadline:
                break
            orientation = [rng.randint(0, 1) for _ in range(num_edges)]
            adj = np.zeros((n, n), dtype=bool)
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if orientation[idx] == 0:
                        adj[i][j] = True
                    else:
                        adj[j][i] = True
                    idx += 1
            masks = _build_masks_from_adj(adj, n)
            bad = _count_bad_4sets(n, masks)
            if bad == 0:
                return orientation, True
            if bad < best_bad:
                best_bad = bad
                best_orientation = orientation[:]
        return best_orientation, best_bad == 0
    
    # For n <= 40, do hill-climbing with occasional restarts
    max_iter = 100000
    no_improve_count = 0
    stall_threshold = 2000
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        if no_improve_count > stall_threshold:
            # Restart with random tournament
            orientation = [rng.randint(0, 1) for _ in range(num_edges)]
            adj = np.zeros((n, n), dtype=bool)
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if orientation[idx] == 0:
                        adj[i][j] = True
                    else:
                        adj[j][i] = True
                    idx += 1
            masks = _build_masks_from_adj(adj, n)
            current_bad = _count_bad_4sets(n, masks)
            if current_bad == 0:
                return orientation, True
            no_improve_count = 0
            continue
        
        # Pick a random edge to flip
        edge_idx = rng.randint(0, num_edges - 1)
        i = 0
        remaining = edge_idx
        while remaining >= n - i - 1:
            remaining -= (n - i - 1)
            i += 1
        j = i + 1 + remaining
        
        # Flip the edge
        old_bit = orientation[edge_idx]
        orientation[edge_idx] = 1 - old_bit
        
        # Update adjacency
        if old_bit == 0:
            # Was i->j, now j->i
            adj[i][j] = False
            adj[j][i] = True
        else:
            # Was j->i, now i->j
            adj[j][i] = False
            adj[i][j] = True
        
        # Rebuild masks (only need to update masks for i and j)
        # Actually, flipping edge (i,j) changes masks for all vertices, 
        # but only the bits for i and j change.
        # masks[v] changes if v's in-neighborhood includes i or j.
        # Specifically:
        # - If i->j becomes j->i: 
        #   - masks[i] loses bit j (j no longer points to i)
        #   - masks[j] gains bit i (i now points to j)
        #   Wait, masks[v] has bit u set if u->v.
        #   So if edge changes from i->j to j->i:
        #     - j no longer has i in its in-neighborhood? No:
        #     - Previously i->j: masks[j] had bit i set
        #     - Now j->i: masks[i] has bit j set, masks[j] does NOT have bit i
        #   So: masks[i] gains bit j, masks[j] loses bit i
        #   But also, for any other vertex v, masks[v] is unchanged.
        
        if old_bit == 0:
            # Was i->j, now j->i
            # masks[j] loses bit i, masks[i] gains bit j
            masks[j] &= ~(1 << i)
            masks[i] |= (1 << j)
        else:
            # Was j->i, now i->j
            # masks[i] loses bit j, masks[j] gains bit i
            masks[i] &= ~(1 << j)
            masks[j] |= (1 << i)
        
        # Recount bad 4-sets
        new_bad = _count_bad_4sets(n, masks)
        
        if new_bad < current_bad:
            current_bad = new_bad
            no_improve_count = 0
            if new_bad == 0:
                return orientation, True
            if new_bad < best_bad:
                best_bad = new_bad
                best_orientation = orientation[:]
        else:
            # Revert
            orientation[edge_idx] = old_bit
            if old_bit == 0:
                masks[j] |= (1 << i)
                masks[i] &= ~(1 << j)
            else:
                masks[i] |= (1 << j)
                masks[j] &= ~(1 << i)
            no_improve_count += 1
    
    return best_orientation, best_bad == 0

def construct():
    """
    Construct a tournament on n vertices such that every 4 vertices have a common dominator.
    Minimizes n.
    Tries local search for n < 67, falls back to Paley tournament at n=67.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Try local search for n values from 31 to 66
    n_values_to_try = list(range(31, 67))
    
    # Allocate time: more for smaller n
    time_alloc = {}
    for n in n_values_to_try:
        if n <= 35:
            time_alloc[n] = 4.0
        elif n <= 45:
            time_alloc[n] = 2.0
        else:
            time_alloc[n] = 0.8
    
    for n in n_values_to_try:
        if time.time() > deadline - 10:
            break
        
        alloc = time_alloc[n]
        remaining = deadline - 10 - time.time()
        alloc = min(alloc, remaining)
        
        if alloc < 0.3:
            break
        
        orientation, valid = _local_search(n, alloc, seed=n * 1000 + 42)
        
        if valid:
            return {'n': n, 'orientation': orientation}
    
    # Fallback: Paley tournament at n=67
    p = 67
    n, orientation = _paley_orientation(p)
    return {'n': n, 'orientation': orientation}
