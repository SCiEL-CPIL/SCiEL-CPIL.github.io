import time
import numpy as np
from itertools import combinations

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _paley_orientation(p):
    """
    Generate orientation for Paley tournament on p vertices.
    i -> j iff (j - i) mod p is a quadratic residue.
    Returns list of bits for pairs (i, j) with i < j in lexicographic order.
    """
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
        
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return orientation

def _check_property_fast(orientation, n):
    """
    Check if every set of 4 vertices has a common dominator.
    A common dominator for {a,b,c,d} is a vertex v in {a,b,c,d} such that
    v -> a, v -> b, v -> c, v -> d (excluding v itself, but since v is in the set,
    it must beat the other 3).
    Wait, the definition says "vertex with arcs to all four". 
    "Every set of 4 vertices must have a vertex with arcs to all four."
    This phrasing is slightly ambiguous. Usually "dominator" means a vertex in the set 
    that dominates the others. Let's re-read carefully.
    "Every set of 4 vertices must have a vertex with arcs to all four."
    Does "all four" mean the 4 vertices in the set? Or is it a typo for "the other three"?
    If v is in the set {v, a, b, c}, v cannot have an arc to itself.
    So it must mean v has arcs to the other 3 vertices in the set.
    
    Let's verify with the known result for p=67.
    For a tournament T on n vertices, the condition is that for every subset S of size 4,
    there exists v in S such that v -> u for all u in S \ {v}.
    
    We can check this efficiently by checking for each vertex v, which sets of 3 vertices
    it dominates. If a set of 4 {a,b,c,d} is "bad", then no vertex in it dominates the other 3.
    This is equivalent to saying: For every 4-set S, max(out-degree in S) >= 3.
    
    Alternative check:
    Iterate over all 4-sets? C(67, 4) is ~ 700,000. This is fast enough.
    For each 4-set, check if any vertex has out-degree 3 within the set.
    
    Let's implement a fast check.
    """
    # Build adjacency matrix or lookup
    # orientation is a list of bits for pairs (i,j) i<j.
    # Let's create a 2D array or a set of tuples for fast lookup.
    # n is up to 90, so a 90x90 boolean matrix is tiny.
    adj = [[0]*n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            if orientation[idx] == 0:
                adj[i][j] = 1
            else:
                adj[j][i] = 1
            idx += 1
            
    # Check all 4-combinations
    from itertools import combinations
    for combo in combinations(range(n), 4):
        a, b, c, d = combo
        # Check if any of a,b,c,d dominates the other 3
        # a dominates b,c,d?
        if adj[a][b] and adj[a][c] and adj[a][d]:
            continue
        # b dominates a,c,d?
        if adj[b][a] and adj[b][c] and adj[b][d]:
            continue
        # c dominates a,b,d?
        if adj[c][a] and adj[c][b] and adj[c][d]:
            continue
        # d dominates a,b,c?
        if adj[d][a] and adj[d][b] and adj[d][c]:
            continue
        
        # If we reach here, this 4-set has no common dominator
        return False
    return True

def _greedy_construct(max_n, deadline, rng_seed=42):
    """
    Greedy constructive algorithm:
    - Start with a small core (e.g., 5 vertices)
    - Add vertices one at a time
    - For each new vertex, choose orientations to minimize "bad" 4-sets
    - A 4-set is "bad" if no vertex in it dominates the other 3
    - Track all bad 4-sets; when adding a new vertex, try to fix as many as possible
    """
    rng = np.random.RandomState(rng_seed)
    start_time = time.time()
    
    # Start with n=5 vertices
    n = 5
    # adj[i][j] = 1 if i -> j, else 0
    adj = np.zeros((max_n, max_n), dtype=np.int8)
    
    # Initialize 5-vertex tournament randomly
    for i in range(n):
        for j in range(i+1, n):
            if rng.random() < 0.5:
                adj[i][j] = 1
                adj[j][i] = 0
            else:
                adj[i][j] = 0
                adj[j][i] = 1
    
    # Helper: check if a 4-set has a dominator
    def _has_dominator(a, b, c, d):
        # Check if a dominates b,c,d
        if adj[a][b] and adj[a][c] and adj[a][d]:
            return True
        # Check if b dominates a,c,d
        if adj[b][a] and adj[b][c] and adj[b][d]:
            return True
        # Check if c dominates a,b,d
        if adj[c][a] and adj[c][b] and adj[c][d]:
            return True
        # Check if d dominates a,b,c
        if adj[d][a] and adj[d][b] and adj[d][c]:
            return True
        return False
    
    # Helper: count bad 4-sets among current n vertices
    def _count_bad_4sets():
        bad_count = 0
        bad_sets = []
        for combo in combinations(range(n), 4):
            if not _has_dominator(*combo):
                bad_count += 1
                bad_sets.append(combo)
        return bad_count, bad_sets
    
    # Initial bad sets
    _, bad_sets = _count_bad_4sets()
    
    # Greedily add vertices
    while n < max_n:
        if time.time() > deadline:
            break
        
        # Try to add vertex n with best orientation
        best_score = float('inf')
        best_orient = None
        
        # For each possible orientation of new vertex vs existing vertices
        # We can use a heuristic: try a few random orientations and pick the best
        for trial in range(20):
            # Random orientation for new vertex
            new_orient = rng.randint(0, 2, size=n)
            
            # Temporarily set orientations
            old_vals = []
            for i in range(n):
                old_vals.append((adj[i][n], adj[n][i]))
                if new_orient[i] == 0:
                    adj[i][n] = 1
                    adj[n][i] = 0
                else:
                    adj[i][n] = 0
                    adj[n][i] = 1
            
            # Count how many previously bad 4-sets are now fixed
            # A bad 4-set {a,b,c,d} is fixed if the new vertex can dominate it
            # or if one of a,b,c,d now dominates the others (including new vertex)
            fixed_count = 0
            for s in bad_sets:
                a, b, c, d = s
                # Check if new vertex (n) dominates a,b,c,d
                if adj[n][a] and adj[n][b] and adj[n][c] and adj[n][d]:
                    fixed_count += 1
                    continue
                # Check if a dominates b,c,d,n
                if adj[a][b] and adj[a][c] and adj[a][d] and adj[a][n]:
                    fixed_count += 1
                    continue
                # Check if b dominates a,c,d,n
                if adj[b][a] and adj[b][c] and adj[b][d] and adj[b][n]:
                    fixed_count += 1
                    continue
                # Check if c dominates a,b,d,n
                if adj[c][a] and adj[c][b] and adj[c][d] and adj[c][n]:
                    fixed_count += 1
                    continue
                # Check if d dominates a,b,c,n
                if adj[d][a] and adj[d][b] and adj[d][c] and adj[d][n]:
                    fixed_count += 1
                    continue
            
            # Also count new bad 4-sets created (sets including new vertex)
            # This is expensive; approximate by sampling
            new_bad_count = 0
            sample_size = min(500, len(list(combinations(range(n), 3))))
            for _ in range(sample_size):
                sample = rng.choice(n, size=3, replace=False)
                a, b, c = sample
                # Check if {a,b,c,n} is bad
                if not _has_dominator(int(a), int(b), int(c), n):
                    new_bad_count += 1
            
            score = len(bad_sets) - fixed_count + new_bad_count
            
            # Restore old values
            for i in range(n):
                adj[i][n], adj[n][i] = old_vals[i]
            
            if score < best_score:
                best_score = score
                best_orient = new_orient
        
        # Apply best orientation
        if best_orient is not None:
            for i in range(n):
                if best_orient[i] == 0:
                    adj[i][n] = 1
                    adj[n][i] = 0
                else:
                    adj[i][n] = 0
                    adj[n][i] = 1
        
        n += 1
        
        # Update bad sets
        _, bad_sets = _count_bad_4sets()
        
        # If no bad sets, we have a valid tournament
        if len(bad_sets) == 0:
            break
    
    # Extract orientation for n vertices
    orientation = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j] == 1:
                orientation.append(0)  # i -> j
            else:
                orientation.append(1)  # j -> i
    
    return n, orientation

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # First, try greedy constructive algorithm to find a smaller n
    # Try multiple random seeds for robustness
    best_n = None
    best_orientation = None
    
    for seed in [42, 123, 456, 789, 1024]:
        if time.time() > deadline - 30:  # Leave time for verification
            break
        
        n_greedy, orient_greedy = _greedy_construct(67, deadline - 30, rng_seed=seed)
        
        # Verify the greedy result
        if _check_property_fast(orient_greedy, n_greedy):
            if best_n is None or n_greedy < best_n:
                best_n = n_greedy
                best_orientation = orient_greedy
    
    # If greedy found a valid result smaller than 67, use it
    if best_n is not None and best_n < 67:
        return {'n': best_n, 'orientation': best_orientation}
    
    # Fallback: Search for the smallest prime p >= 5 such that Paley tournament satisfies the property.
    p = 5
    while p <= 90:
        if time.time() > deadline:
            break
            
        if _is_prime(p):
            orientation = _paley_orientation(p)
            if _check_property_fast(orientation, p):
                return {'n': p, 'orientation': orientation}
        
        p += 1
    
    # Final fallback to 67
    p = 67
    orientation = _paley_orientation(p)
    return {'n': p, 'orientation': orientation}
