import time
import numpy as np
import random
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
    
    i_indices = []
    j_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            i_indices.append(i)
            j_indices.append(j)
            
    i_arr = np.array(i_indices, dtype=int)
    j_arr = np.array(j_indices, dtype=int)
    
    diffs = (j_arr - i_arr) % p
    bits = np.where(squares[diffs], 0, 1)
    
    return n, bits.tolist()

def _adj_to_orientation(n, adj):
    """Convert adjacency matrix to list of bits in lexicographic order."""
    orientation = []
    for i in range(n):
        for j in range(i+1, n):
            # 0 means i -> j, 1 means j -> i
            # adj[i][j] is 1 if i -> j
            if adj[i][j] == 1:
                orientation.append(0)
            else:
                orientation.append(1)
    return n, orientation

def _count_violations(n, adj):
    """
    Count the number of 4-sets that do NOT have a common dominator.
    adj is a 2D numpy array where adj[i][j] is 1 if i -> j, 0 if j -> i.
    Uses vectorized numpy operations for speed.
    """
    # For each 4-set (a,b,c,d), a dominator v is a vertex such that
    # adj[v,a] == 1, adj[v,b] == 1, adj[v,c] == 1, adj[v,d] == 1
    # AND v is not in {a,b,c,d}.
    
    # We can compute the intersection of in-neighborhoods for each 4-set.
    # in_neigh[v] is a bitmask or boolean array of vertices u such that v -> u.
    # Actually, adj[v, u] = 1 means v -> u. So for a fixed set S, we want v such that
    # all u in S have adj[v, u] == 1.
    
    # Let's use the property: 
    # A 4-set S is "good" if there exists v not in S such that v -> all u in S.
    # This is equivalent to: The intersection of the sets of dominators of each u in S 
    # (excluding S itself) is non-empty.
    
    # Efficient way:
    # Precompute for each vertex u, the set of dominators D(u) = {v | v -> u}.
    # In numpy terms, D(u) corresponds to the column u of adj, where adj[v, u] == 1.
    # So D(u) is the set of indices v where adj[v, u] is 1.
    
    # For a 4-set {a, b, c, d}, we need to check if (D(a) & D(b) & D(c) & D(d)) - {a,b,c,d} is not empty.
    
    # Since n <= 90, we can represent sets as Python integers (bitmasks) or numpy arrays.
    # Bitmasks are likely faster for intersection.
    
    # Let's convert adj to a list of bitmasks for in-neighborhoods (dominators).
    # dom[u] is an integer bitmask where bit v is set if v -> u.
    
    dom = [0] * n
    for u in range(n):
        mask = 0
        for v in range(n):
            if adj[v, u] == 1:
                mask |= (1 << v)
        dom[u] = mask
        
    # Iterate through all 4-combinations
    violations = 0
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        common = dom[a] & dom[b] & dom[c] & dom[d]
        # Mask out the vertices in the quad itself
        quad_mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
        if (common & ~quad_mask) == 0:
            violations += 1
            
    return violations

def _local_search(n, time_limit, seed=42):
    """
    Attempt to find a valid tournament on n vertices using local search.
    Returns (n, orientation_list) if successful, else None.
    """
    random.seed(seed)
    # Initialize random tournament
    # adj[i][j] = 1 if i -> j
    adj = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.5:
                adj[i][j] = 1
                adj[j][i] = 0
            else:
                adj[i][j] = 0
                adj[j][i] = 1
                
    start_time = time.time()
    current_violations = _count_violations(n, adj)
    
    # If already valid (unlikely for random n=31)
    if current_violations == 0:
        return _adj_to_orientation(n, adj)
        
    # Hill climbing / Simulated Annealing
    # We will try to flip edges that reduce violations.
    # To speed up, we can maintain a list of "bad" 4-sets and only check those affected by a flip.
    # But for n=31, full recount is ~0.05s.
    # Let's try a simpler approach: Random flips, accept if violations decrease.
    
    best_adj = adj.copy()
    best_violations = current_violations
    
    while time.time() - start_time < time_limit:
        # Pick random edge to flip
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        if i == j:
            continue
        
        # Flip edge i-j
        adj[i][j] = 1 - adj[i][j]
        adj[j][i] = 1 - adj[j][i]
        
        new_violations = _count_violations(n, adj)
        
        if new_violations < current_violations:
            current_violations = new_violations
            if new_violations < best_violations:
                best_violations = new_violations
                best_adj = adj.copy()
                if best_violations == 0:
                    return _adj_to_orientation(n, best_adj)
        else:
            # Revert
            adj[i][j] = 1 - adj[i][j]
            adj[j][i] = 1 - adj[j][i]
            
    if best_violations == 0:
        return _adj_to_orientation(n, best_adj)
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try local search for n=31
    # If this works, it's a huge win.
    # If not, fall back to Paley 67.
    
    result = _local_search(31, time_limit=100)
    
    if result is not None:
        return {'n': result[0], 'orientation': result[1]}
        
    # Fallback to Paley 67
    p = 67
    if not _is_prime(p) or p % 4 != 3:
        # Should not happen
        pass
        
    n, orientation = _paley_orientation(p)
    
    return {'n': n, 'orientation': orientation}
