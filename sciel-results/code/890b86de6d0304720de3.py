import time
import numpy as np
from itertools import combinations

def construct():
    """
    Construct a tournament on n vertices where every set of 4 vertices has a common dominator.
    
    Strategy:
    1. Try to find a valid tournament with n < 67 using local search (greedy hill-climbing).
    2. If not found within time budget, fall back to the Paley tournament on p=67 vertices.
    
    The Paley tournament is defined on Z_p where p ≡ 3 (mod 4) is prime.
    For i < j, we orient i -> j iff (j-i) mod p is a quadratic residue mod p.
    For p=67, this gives a tournament where every 4-set has a common dominator.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # First, try local search to find a smaller tournament
    found_small = False
    best_n = 67
    best_orientation = None
    
    # Try n values from 31 up to 66
    for n in range(31, 67):
        if time.time() > deadline - 30:  # Leave 30s for fallback
            break
        
        # Try local search for this n
        result = _local_search(n, start_time, deadline - 30)
        if result is not None:
            found_small = True
            best_n = n
            best_orientation = result
            break
    
    if found_small:
        return {'n': best_n, 'orientation': best_orientation}
    
    # Fallback: Paley tournament on p=67
    p = 67
    
    # Compute quadratic residues modulo p
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    
    is_square = np.zeros(p, dtype=bool)
    for r in squares:
        is_square[r] = True
    
    # Construct orientation bits
    n_pairs = p * (p - 1) // 2
    orientation = np.zeros(n_pairs, dtype=np.uint8)
    
    idx = 0
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if is_square[diff]:
                orientation[idx] = 0  # i -> j
            else:
                orientation[idx] = 1  # j -> i
            idx += 1
    
    return {'n': p, 'orientation': orientation.tolist()}

def _count_violations(orientation, n):
    """Count the number of 4-sets that lack a common dominator."""
    # orientation is a list of bits for pairs (i,j) with i<j
    # 0 means i->j, 1 means j->i
    
    # Build adjacency matrix for quick lookup
    # adj[i][j] = 1 if i -> j, 0 if j -> i
    adj = np.zeros((n, n), dtype=np.uint8)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if orientation[idx] == 0:
                adj[i][j] = 1  # i -> j
            else:
                adj[j][i] = 1  # j -> i
            idx += 1
    
    violations = 0
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        # Check if any vertex dominates all others in the set
        # Vertex a dominates if a->b, a->c, a->d
        if adj[a][b] and adj[a][c] and adj[a][d]:
            continue
        if adj[b][a] and adj[b][c] and adj[b][d]:
            continue
        if adj[c][a] and adj[c][b] and adj[c][d]:
            continue
        if adj[d][a] and adj[d][b] and adj[d][c]:
            continue
        violations += 1
    
    return violations

def _flip_edge(orientation, n, i, j):
    """Flip the edge between i and j. Returns new orientation list."""
    if i > j:
        i, j = j, i
    # Find index of pair (i, j)
    # Pairs are in lexicographic order: (0,1), (0,2), ..., (0,n-1), (1,2), ..., (n-2,n-1)
    idx = 0
    for a in range(i):
        idx += (n - 1 - a)
    idx += (j - i - 1)
    
    new_orientation = orientation.copy()
    new_orientation[idx] = 1 - new_orientation[idx]
    return new_orientation

def _local_search(n, start_time, deadline):
    """
    Local search to find a valid tournament on n vertices.
    Returns orientation list if found, None otherwise.
    """
    np.random.seed(42)
    
    # Start with a random tournament
    n_pairs = n * (n - 1) // 2
    orientation = np.random.randint(0, 2, size=n_pairs, dtype=np.uint8)
    
    # Count initial violations
    current_violations = _count_violations(orientation.tolist(), n)
    if current_violations == 0:
        return orientation.tolist()
    
    best_orientation = orientation.copy()
    best_violations = current_violations
    
    # Greedy hill-climbing: try flipping each edge, keep if it reduces violations
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                
                new_orientation = _flip_edge(orientation.tolist(), n, i, j)
                new_violations = _count_violations(new_orientation, n)
                
                if new_violations < best_violations:
                    best_violations = new_violations
                    best_orientation = new_orientation
                    orientation = new_orientation
                    improved = True
                    if best_violations == 0:
                        return orientation.tolist()
    
    return None
