<<<<<<< SEARCH
import time

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges).
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # For k=5, g=9, try L-U-W with d=4 first
    if k == 5 and g == 9:
        # d=4 gives size 2*5^4 = 1250
        # We want the smallest n, so try d=4 first
        n, edges = _construct_luw(k, 4)
        # Check if girth is sufficient
        if _girth_at_least(n, edges, g):
            return n, edges
        
        # If d=4 fails girth, try d=5
        if time.time() < deadline - 10:
            n, edges = _construct_luw(k, 5)
            if _girth_at_least(n, edges, g):
                return n, edges
        
        # If d=5 fails, try d=6
        if time.time() < deadline - 5:
            n, edges = _construct_luw(k, 6)
            if _girth_at_least(n, edges, g):
                return n, edges
        
        # Fallback: use larger d to ensure girth
        d = 7
        n, edges = _construct_luw(k, d)
        return n, edges

    # General case: use seed logic
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    return _construct_luw(q, d)
=======
import time

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges).
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    if k == 5 and g == 9:
        # Try ABS construction first - typically smaller than L-U-W
        n, edges = _construct_abs(k, g)
        if _girth_at_least(n, edges, g):
            return n, edges
        
        # Fallback to L-U-W
        n, edges = _construct_luw(k, 4)
        if _girth_at_least(n, edges, g):
            return n, edges
        
        if time.time() < deadline - 10:
            n, edges = _construct_luw(k, 5)
            if _girth_at_least(n, edges, g):
                return n, edges
        
        if time.time() < deadline - 5:
            n, edges = _construct_luw(k, 6)
            if _girth_at_least(n, edges, g):
                return n, edges
        
        d = 7
        n, edges = _construct_luw(k, d)
        return n, edges

    # General case: try ABS first
    n, edges = _construct_abs(k, g)
    if _girth_at_least(n, edges, g):
        return n, edges
    
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    return _construct_luw(q, d)

def _construct_abs(k, g):
    """
    Alon-Babai-Suzuki construction: Cayley graph on upper triangular 
    unit matrices over GF(k) with elementary matrix generators.
    """
    if k == 5:
        q = 5
    else:
        q = k
    
    # For girth g, use depth d = floor((g-1)/2)
    d = (g - 1) // 2
    
    # Elements are d x d upper triangular unit matrices over GF(q)
    # Represent as tuple of entries above diagonal: (a12, a13, ..., a1d, a23, ..., a2d, ..., a(d-1)d)
    # Number of such entries = d*(d-1)//2
    num_entries = d * (d - 1) // 2
    
    # Build all group elements
    # Element is tuple of length num_entries
    # Index mapping: entry (i,j) for i<j maps to position
    # Position for (i,j): sum of (j-i-1) for i' in 0..i-1 plus (j-i-1)
    # Let's use a simpler flat representation
    
    def mat_to_tuple(mat):
        """Convert upper triangular unit matrix to flat tuple of off-diagonal entries."""
        result = []
        for i in range(d):
            for j in range(i+1, d):
                result.append(mat[i][j])
        return tuple(result)
    
    def tuple_to_mat(t):
        """Convert flat tuple back to matrix."""
        mat = [[0]*d for _ in range(d)]
        idx = 0
        for i in range(d):
            for j in range(i+1, d):
                mat[i][j] = t[idx]
                idx += 1
        return mat
    
    def mat_mul(A, B):
        """Multiply two upper triangular unit matrices over GF(q)."""
        C = [[0]*d for _ in range(d)]
        for i in range(d):
            C[i][i] = 1
            for j in range(i+1, d):
                s = 0
                for m in range(i+1, j+1):
                    s += A[i][m] * B[m][j]
                C[i][j] = s % q
        return C
    
    # Generate all group elements via BFS from identity
    identity = tuple([0] * num_entries)
    identity_mat = tuple_to_mat(identity)
    
    # Elementary generators: E(i,j,a) for i<j, a in 1..q-1
    # E(i,j,a) has 1 on diagonal, a at position (i,j), 0 elsewhere off-diagonal
    generators = []
    for i in range(d):
        for j in range(i+1, d):
            for a in range(1, q):
                gen_mat = [[0]*d for _ in range(d)]
                for m in range(d):
                    gen_mat[m][m] = 1
                gen_mat[i][j] = a
                generators.append(mat_to_tuple(gen_mat))
    
    # BFS to enumerate all group elements
    group_elements = [identity]
    index = {identity: 0}
    head = 0
    
    while head < len(group_elements):
        elem = group_elements[head]
        head += 1
        elem_mat = tuple_to_mat(elem)
        
        for gen in generators:
            gen_mat = tuple_to_mat(gen)
            prod = mat_mul(elem_mat, gen_mat)
            prod_tuple = mat_to_tuple(prod)
            
            if prod_tuple not in index:
                index[prod_tuple] = len(group_elements)
                group_elements.append(prod_tuple)
    
    n = len(group_elements)
    
    # Build edges: for each vertex, connect to vertex * generator
    edges = []
    seen_edges = set()
    
    for i, elem in enumerate(group_elements):
        elem_mat = tuple_to_mat(elem)
        for gen in generators:
            gen_mat = tuple_to_mat(gen)
            prod = mat_mul(elem_mat, gen_mat)
            prod_tuple = mat_to_tuple(prod)
            j = index[prod_tuple]
            
            if i < j:
                edge = (i, j)
                if edge not in seen_edges:
                    seen_edges.add(edge)
                    edges.append(edge)
    
    return n, edges
>>>>>>> REPLACE
