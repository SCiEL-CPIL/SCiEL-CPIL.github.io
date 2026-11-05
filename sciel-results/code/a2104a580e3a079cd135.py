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

def _count_violations(n, adj):
    """
    Count the number of 4-sets that do NOT have a common dominator.
    adj is a 2D numpy array where adj[i][j] is 1 if i -> j, 0 if j -> i.
    """
    violations = 0
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    # Check if there is a dominator for {a,b,c,d}
                    # Dominator must be outside {a,b,c,d}
                    has_dominator = False
                    for v in range(n):
                        if v == a or v == b or v == c or v == d:
                            continue
                        # Check if v -> a, v -> b, v -> c, v -> d
                        # adj[v][a] == 1 means v -> a
                        if adj[v][a] == 1 and adj[v][b] == 1 and adj[v][c] == 1 and adj[v][d] == 1:
                            has_dominator = True
                            break
                    if not has_dominator:
                        violations += 1
    return violations

def _count_violations_fast(n, adj):
    """
    Faster violation count using numpy vectorization.
    adj[i][j] = 1 if i -> j.
    """
    # adj is n x n
    # For each 4-set {a,b,c,d}, we need a vertex v not in {a,b,c,d}
    # such that adj[v][a] == 1 and adj[v][b] == 1 and adj[v][c] == 1 and adj[v][d] == 1.
    
    # This is equivalent to: for each 4-set, check if (adj[:, a] & adj[:, b] & adj[:, c] & adj[:, d]) 
    # has any 1 at an index v not in {a,b,c,d}.
    
    # We can precompute for all 4-sets? No, too many.
    # But we can vectorize over v for each 4-set?
    
    # Let's try a different approach:
    # A 4-set {a,b,c,d} is "bad" if for all v not in {a,b,c,d}, 
    # it is NOT the case that (v->a and v->b and v->c and v->d).
    # i.e., for all v not in {a,b,c,d}, at least one of adj[v][a], adj[v][b], adj[v][c], adj[v][d] is 0.
    
    # Let M be the matrix adj.
    # For a fixed 4-set S={a,b,c,d}, define vector V_S where V_S[v] = 1 if v dominates S.
    # V_S = M[:, a] & M[:, b] & M[:, c] & M[:, d]
    # S is bad if sum(V_S[v] for v not in S) == 0.
    
    # We can compute this for all 4-sets using combinations.
    # n=31, C(31,4) = 31465.
    # For each, we do 4 array lookups and an AND. This is fast in numpy.
    
    violations = 0
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    # Dominators of {a,b,c,d}
                    doms = adj[:, a] & adj[:, b] & adj[:, c] & adj[:, d]
                    # Exclude a,b,c,d themselves (though they can't dominate themselves in a tournament, 
                    # adj[x][x] is 0, so they are already 0 in doms)
                    # But just in case, or if we want to be strict about "outside":
                    # Actually, in a tournament, a vertex cannot point to itself.
                    # So adj[a][a] = 0. Thus doms[a] will be 0.
                    # So we just need to check if any doms[v] is 1 for v in 0..n-1.
                    # Since doms[a]=doms[b]=doms[c]=doms[d]=0, sum(doms) counts only outside vertices.
                    if np.any(doms):
                        pass # Good
                    else:
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
    current_violations = _count_violations_fast(n, adj)
    
    # If already valid (unlikely for random n=31)
    if current_violations == 0:
        return _adj_to_orientation(n, adj)
        
    best_adj = adj.copy()
    best_violations = current_violations
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        # Pick random edge to flip
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        if i == j:
            continue
        
        # Flip edge i-j
        adj[i][j] = 1 - adj[i][j]
        adj[j][i] = 1 - adj[j][i]
        
        new_violations = _count_violations_fast(n, adj)
        
        delta = new_violations - current_violations
        
        if delta < 0 or random.random() < np.exp(-delta / T):
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
            
        T *= cooling
        if T < T_min:
            T = T_min
            
    if best_violations == 0:
        return _adj_to_orientation(n, best_adj)
    return None

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

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try local search for n=31
    # If this works, it's a huge win.
    # If not, fall back to Paley 67.
    
    # Use multiple seeds/restarts if time permits?
    # For now, single run with SA.
    
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
