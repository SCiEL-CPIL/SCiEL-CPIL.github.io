import time
import random
from itertools import combinations

DEADLINE = 110.0

def _precompute_q3_signatures():
    M = [
        [1, 1, 1, 0],
        [1, 1, 0, 1],
        [1, 0, 1, 1],
        [0, 1, 1, 1]
    ]
    
    def to_int(matrix):
        val = 0
        for i in range(4):
            for j in range(4):
                if matrix[i][j]:
                    val |= (1 << (i * 4 + j))
        return val
        
    signatures = set()
    import itertools
    for row_perm in itertools.permutations(range(4)):
        for col_perm in itertools.permutations(range(4)):
            permuted = [[0]*4 for _ in range(4)]
            for i in range(4):
                for j in range(4):
                    permuted[i][j] = M[row_perm[i]][col_perm[j]]
            signatures.add(to_int(permuted))
    return signatures

Q3_SIGS = _precompute_q3_signatures()

def _find_q3(masks):
    """
    Returns a list of edges in a Q3, or None.
    """
    n = 16
    for verts in combinations(range(n), 8):
        for A in combinations(verts, 4):
            B = tuple(v for v in verts if v not in A)
            a_idx = {v: i for i, v in enumerate(A)}
            b_idx = {v: i for i, v in enumerate(B)}
            
            sig = 0
            edges_in_q3 = []
            for u in A:
                ui = a_idx[u]
                mask_u = masks[u]
                for v in B:
                    vi = b_idx[v]
                    if (mask_u >> v) & 1:
                        sig |= (1 << (ui * 4 + vi))
                        edges_in_q3.append((u, v))
                        
            if sig in Q3_SIGS:
                return edges_in_q3
    return None

def construct():
    start_time = time.time()
    n = 16
    deadline = start_time + DEADLINE
    
    best_edges = None
    best_count = 0
    
    # Try a few random restarts or just one greedy run?
    # One greedy run is likely sufficient given the small size.
    # But randomizing the edge removal might help escape local minima.
    
    masks = [0] * n
    for i in range(n):
        mask = (1 << n) - 1
        mask &= ~(1 << i)
        masks[i] = mask
        
    edge_count = n * (n - 1) // 2
    
    while time.time() < deadline - 5:
        q3_edges = _find_q3(masks)
        if q3_edges is None:
            break
        
        # Pick a random edge to remove
        u, v = random.choice(q3_edges)
        masks[u] &= ~(1 << v)
        masks[v] &= ~(1 << u)
        edge_count -= 1
        
    # Convert to list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (masks[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
