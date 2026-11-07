import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    # Build edge lookup
    edge_to_idx = {}
    for i, (u, v) in enumerate(edges):
        edge_to_idx[(u, v)] = i
        edge_to_idx[(v, u)] = i
        
    # Precompute edge endpoints for fast matrix building
    edge_u = np.array([u for u, v in edges], dtype=np.int32)
    edge_v = np.array([v for u, v in edges], dtype=np.int32)

    # Build adjacency list for faster traversal
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
    
    # Identify spanning tree and chords using BFS
    visited = [False] * n
    tree_edges = set()
    visited[0] = True
    queue = [0]
    
    while queue:
        u = queue.pop(0)
        for v, ei in adj[u]:
            if not visited[v]:
                visited[v] = True
                tree_edges.add(ei)
                queue.append(v)
    
    # Chords are edges not in the spanning tree
    chord_indices = [i for i in range(num_edges) if i not in tree_edges]
    num_chords = len(chord_indices)
    
    # Precompute base matrix with all +1
    A_base = np.zeros((n, n), dtype=np.float64)
    for u, v in edges:
        A_base[u, v] = 1.0
        A_base[v, u] = 1.0
        
    # Precompute chord edge positions in the matrix for quick flipping
    # We need to flip signs of specific entries
    chord_u = edge_u[chord_indices]
    chord_v = edge_v[chord_indices]

    best_radius = float('inf')
    best_signs = None
    
    # Helper: check if lift is connected (has a negative cycle)
    # Using parity assignment
    def is_lift_connected(signs):
        parity = [-1] * n
        parity[0] = 0
        queue = [0]
        
        while queue:
            u = queue.pop(0)
            pu = parity[u]
            for v, ei in adj[u]:
                s = signs[ei]
                # If s==1, parity same. If s==-1, parity flips.
                # expected_parity = pu if s==1 else 1-pu
                # Using XOR: expected = pu ^ (1 if s==-1 else 0)
                # s is 1 or -1. 
                # Let's use integer arithmetic for speed
                # s == -1 -> 1, s == 1 -> 0
                flip = 1 if s == -1 else 0
                expected_parity = pu ^ flip
                
                if parity[v] == -1:
                    parity[v] = expected_parity
                    queue.append(v)
                else:
                    if parity[v] != expected_parity:
                        return True
        
        return False

    # Iterate over all chord assignments
    # We construct signs array for each mask
    # To avoid allocating a new array every time, we can reuse one?
    # But we need to store the best one.
    
    for mask in range(1 << num_chords):
        if time.time() > deadline:
            break
            
        # Create signs array
        # Start with all +1
        signs = np.ones(num_edges, dtype=np.int8)
        
        # Set chord signs based on mask
        for j, idx in enumerate(chord_indices):
            if mask & (1 << j):
                signs[idx] = -1
                
        # Check validity
        if not is_lift_connected(signs):
            continue
            
        # Compute spectral radius
        # Start with base matrix
        A_signed = A_base.copy()
        
        # Flip signs for chords that are -1
        # We need to identify which chords are -1 in this mask
        # Instead of iterating over all chords, we can just build the matrix from signs
        # But copying A_base and flipping is faster if few chords are -1?
        # Actually, building from scratch is O(E). Copying is O(N^2).
        # N=18, N^2=324. E=27.
        # Building from scratch is faster!
        
        A_signed = np.zeros((n, n), dtype=np.float64)
        # Vectorized update
        # A_signed[edge_u, edge_v] = signs
        # A_signed[edge_v, edge_u] = signs
        # This works for numpy arrays
        A_signed[edge_u, edge_v] = signs.astype(np.float64)
        A_signed[edge_v, edge_u] = signs.astype(np.float64)
        
        eigvals = np.linalg.eigvalsh(A_signed)
        radius = np.max(np.abs(eigvals))
        
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
            
    if best_signs is None:
        # Fallback
        signs = np.ones(num_edges, dtype=np.int8)
        if num_chords > 0:
            signs[chord_indices[0]] = -1
        else:
            signs[0] = -1
        return signs
        
    return best_signs
