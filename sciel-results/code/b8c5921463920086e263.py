import time
import random
import numpy as np

def _is_c4_free(A):
    """Check if adjacency matrix A represents a C4-free graph."""
    n = A.shape[0]
    # C4-free iff for all u != v, |N(u) ∩ N(v)| ≤ 1
    # (A @ A)[i][j] = |N(i) ∩ N(j)| for i != j
    # Diagonal: (A @ A)[i][i] = deg(i), which we don't care about
    common = A @ A
    # Check off-diagonal
    for i in range(n):
        for j in range(i + 1, n):
            if common[i, j] > 1:
                return False
    return True

def _min_degree(A):
    return int(np.min(A.sum(axis=1)))

def _polarity_graph_from_form(q, form_type, n_override=None):
    """
    Construct a polarity graph from a quadratic form over GF(q).
    
    For a quadratic form Q over GF(q) (q odd), define graph on points of projective space
    or affine space. Here we use a simpler construction:
    
    For GF(q), q odd:
    - Vertices: elements of GF(q) (affine space), so n = q
    - Edge (a, b) if Q(a-b) is a non-square in GF(q) (or some other criterion)
    
    Actually, let's use the standard polarity graph construction:
    - Take the projective plane PG(2, q) or use a simpler vector space construction
    
    Simpler approach: Use the graph where vertices are elements of GF(q^2) or use
    a direct construction based on quadratic residues.
    
    For now, let's try a construction on GF(q) where:
    - n = q vertices (0 to q-1)
    - Edge (i, j) for i < j if (i-j)^2 mod q is a non-residue (or some criterion)
    
    Actually, the standard C4-free graph from finite fields:
    For q = p^k, p prime, the graph on Z_q where (i,j) is an edge if
    i-j is a quadratic non-residue mod q. This is C4-free when q is a prime power
    with certain properties.
    
    Let me use a well-known construction: the graph from a quadratic form.
    
    For GF(q), q odd prime:
    - Vertices: 0, 1, ..., q-1
    - Edge (i, j) iff (i-j) is a quadratic non-residue mod q
    This graph has degree (q-1)/2 for each vertex, and is C4-free.
    
    But we need min_degree >= n - 39, i.e., degree >= q - 39.
    With degree = (q-1)/2, we need (q-1)/2 >= q - 39
    => q - 1 >= 2q - 78
    => 77 >= q
    So q <= 77. And we want n = q as large as possible, so try q = 46, 47, etc.
    But q must be prime for the simple quadratic residue construction.
    
    Primes near 46-47: 43, 47.
    For q=47: degree = 23, need >= 47-39 = 8. 23 >= 8. Good!
    For q=43: degree = 21, need >= 43-39 = 4. 21 >= 4. Good!
    
    So q=47 gives n=47, min_degree=23 >= 8. This should work!
    
    Let me verify: is the graph on Z_47 with edges for quadratic non-residues C4-free?
    Yes, this is a standard result. The graph is C4-free because if there were a C4,
    it would imply a certain multiplicative relation that contradicts the properties
    of quadratic residues.
    """
    q = q
    # Check if q is prime
    if q < 2:
        return None
    
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    if not is_prime(q):
        return None
    
    # Compute quadratic residues mod q
    residues = set()
    for x in range(1, q):
        residues.add((x * x) % q)
    
    # Build adjacency matrix
    A = np.zeros((q, q), dtype=np.int8)
    for i in range(q):
        for j in range(i + 1, q):
            diff = (j - i) % q
            if diff in residues:
                # Edge if diff is a quadratic residue
                A[i, j] = 1
                A[j, i] = 1
    
    return A

def _construct_c4_free(n_target):
    """
    Try to construct a C4-free graph on n_target vertices with min_degree >= n_target - 39.
    Uses polarity graph construction from finite fields.
    """
    # Try prime field sizes
    # We want n = q where q is prime, and (q-1)/2 >= q - 39
    # => q <= 77
    
    # Try primes in descending order from min(n_target, 77) down
    candidates = []
    for q in range(min(n_target, 77), 1, -1):
        # Check if q is prime
        if q >= 2:
            is_p = True
            if q < 2:
                is_p = False
            elif q == 2:
                is_p = True
            elif q % 2 == 0:
                is_p = False
            else:
                for i in range(3, int(q**0.5) + 1, 2):
                    if q % i == 0:
                        is_p = False
                        break
            if is_p:
                # Check if degree requirement is met
                deg = (q - 1) // 2
                req = q - 39
                if deg >= req and q >= 2:
                    candidates.append(q)
    
    # Also try to construct graphs with n not equal to q by taking subgraphs
    # or using other constructions
    
    # First, try exact prime sizes
    for q in candidates:
        A = _polarity_graph_from_form(q, 'quadratic')
        if A is not None:
            if _is_c4_free(A):
                min_deg = _min_degree(A)
                if min_deg >= q - 39:
                    return A, q
    
    # If no exact prime works, try to find a C4-free subgraph of a larger prime graph
    # or use a different construction
    
    # Try construction on GF(q) with q a prime power, using a different edge rule
    # For now, fallback to random search with better initialization
    
    return None, None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Strategy 1: Try polarity graph constructions
    for n in range(47, 38, -1):
        if time.time() > deadline:
            break
        
        A, q = _construct_c4_free(n)
        if A is not None and q == n:
            # Valid graph found
            edges = []
            iu, ju = np.where(np.triu(A, k=1) == 1)
            for i, j in zip(iu, ju):
                edges.append([int(i), int(j)])
            return {'n': n, 'edges': edges}
    
    # Strategy 2: Random search with better initialization
    # Use the insight that we want high minimum degree, so start with dense graphs
    # and remove edges to break C4s
    
    for n in range(47, 38, -1):
        if time.time() > deadline:
            break
        
        target_min_deg = n - 39
        max_restarts = 50
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Initialize with a dense random graph
            A = np.zeros((n, n), dtype=np.int8)
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            
            # Start with high probability to get high degree
            p = 0.7
            rand_vals = (np.random.rand(n, n) < p).astype(np.int8)
            A = np.where(mask, rand_vals, 0)
            A += A.T
            
            # Local search: remove edges to break C4s while maintaining min degree
            improved = True
            max_steps = 200
            steps = 0
            while improved and steps < max_steps:
                if time.time() > deadline:
                    break
                steps += 1
                improved = False
                
                # Find C4 violations
                common = A @ A
                viol_idx = np.where(common > 1)
                
                valid_viol = [(u, v) for u, v in zip(viol_idx[0], viol_idx[1]) if u != v]
                
                if valid_viol:
                    # Pick a random violation and remove an edge
                    u, v = random.choice(valid_viol)
                    common_neighbors = np.where((A[u] == 1) & (A[v] == 1))[0]
                    if len(common_neighbors) > 0:
                        w = common_neighbors[random.randint(0, len(common_neighbors) - 1)]
                        A[u, w] = 0
                        A[w, u] = 0
                        improved = True
                else:
                    # No C4 violations, check min degree
                    degrees = A.sum(axis=1)
                    min_deg = int(np.min(degrees))
                    
                    if min_deg >= target_min_deg:
                        # Valid graph found
                        edges = []
                        iu, ju = np.where(np.triu(A, k=1) == 1)
                        for i, j in zip(iu, ju):
                            edges.append([int(i), int(j)])
                        return {'n': n, 'edges': edges}
                    else:
                        # Try to add edges to low-degree vertices
                        low_v = int(np.argmin(degrees))
                        candidates = np.where(A[low_v] == 0)[0]
                        candidates = candidates[candidates != low_v]
                        
                        if len(candidates) > 0:
                            random.shuffle(candidates)
                            added = False
                            for u in candidates:
                                # Try adding edge
                                A[low_v, u] = 1
                                A[u, low_v] = 1
                                
                                # Check if C4-free
                                common_new = A @ A
                                valid = True
                                if common_new[low_v, u] > 1:
                                    valid = False
                                else:
                                    row_lv = common_new[low_v, :].copy()
                                    row_lv[low_v] = 0
                                    if np.any(row_lv > 1):
                                        valid = False
                                    else:
                                        row_u = common_new[u, :].copy()
                                        row_u[u] = 0
                                        if np.any(row_u > 1):
                                            valid = False
                                
                                if valid:
                                    added = True
                                    improved = True
                                    break
                                else:
                                    A[low_v, u] = 0
                                    A[u, low_v] = 0
                            
                            if not added:
                                # Remove a random edge to break symmetry
                                if np.any(degrees > 0):
                                    v = random.choice(np.where(degrees > 0)[0])
                                    neighbors = np.where(A[v] == 1)[0]
                                    if len(neighbors) > 0:
                                        u = random.choice(neighbors)
                                        A[v, u] = 0
                                        A[u, v] = 0
                                        improved = True
    
    # Fallback: return a small valid graph
    return {'n': 39, 'edges': []}
