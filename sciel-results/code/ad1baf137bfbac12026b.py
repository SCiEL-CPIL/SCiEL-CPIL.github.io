import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    n_edges = len(edges)
    
    # Map edge to index
    edge_map = {}
    for i, (u, v) in enumerate(edges):
        edge_map[(u, v)] = i
        edge_map[(v, u)] = i
    
    # Identify tree edges: top path, bottom path, one matching edge (0, m)
    tree_edges = []
    for i in range(m - 1):
        tree_edges.append((i, i + 1))
        tree_edges.append((m + i, m + i + 1))
    tree_edges.append((0, m))
    
    # Chords: top closure, bottom closure, remaining matching
    chord_edges = []
    chord_edges.append((m - 1, 0))
    chord_edges.append((2 * m - 1, m))
    for i in range(1, m):
        chord_edges.append((i, m + i))
    
    tree_indices = [edge_map[e] for e in tree_edges]
    chord_indices = [edge_map[e] for e in chord_edges]
    num_chords = len(chord_indices)
    
    # Precompute base adjacency (tree edges = +1)
    A_base = np.zeros((n, n), dtype=float)
    for idx in tree_indices:
        u, v = edges[idx]
        A_base[u, v] = 1.0
        A_base[v, u] = 1.0
    
    # Precompute chord (u,v) pairs
    chord_uv = [edges[idx] for idx in chord_indices]
    
    # Precompute edge_uv for connectivity check
    edge_uv = list(edges)
    
    # Fast connectivity check using BFS on the 2-lift
    # Precompute for each vertex, which edge indices connect to it
    vertex_edges = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edge_uv):
        vertex_edges[u].append((idx, v))
        vertex_edges[v].append((idx, u))
    
    def lift_connected(signs):
        """Check if the 2-lift is connected using BFS."""
        n_lift = 2 * n
        visited = np.zeros(n_lift, dtype=bool)
        # Use a list as a queue for speed
        queue = [0]  # vertex (0, 0)
        visited[0] = True
        count = 1
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            v = node // 2
            s = node % 2
            for idx, other in vertex_edges[v]:
                sigma = signs[idx]
                ns = s if sigma == 1 else 1 - s
                nnode = other * 2 + ns
                if not visited[nnode]:
                    visited[nnode] = True
                    count += 1
                    queue.append(nnode)
        return count == n_lift
    
    best_radius = float('inf')
    best_signs = None
    
    # Enumerate all 2^(num_chords) sign assignments for chords
    # Tree edges are always +1
    base_signs = np.ones(n_edges, dtype=np.int8)
    
    total_masks = 1 << num_chords
    
    for mask in range(total_masks):
        if time.time() > deadline:
            break
        
        # Build signs array
        signs = base_signs.copy()
        for i, idx in enumerate(chord_indices):
            if (mask >> i) & 1:
                signs[idx] = -1
        
        # Check connectivity
        if not lift_connected(signs):
            continue
        
        # Build signed adjacency matrix
        A = A_base.copy()
        for i, idx in enumerate(chord_indices):
            u, v = chord_uv[i]
            sgn = -1.0 if ((mask >> i) & 1) else 1.0
            A[u, v] = sgn
            A[v, u] = sgn
        
        # Compute spectral radius
        try:
            ev = np.linalg.eigvalsh(A)
            radius = float(np.max(np.abs(ev)))
        except Exception:
            continue
        
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
    
    if best_signs is None:
        # Fallback: all +1
        best_signs = np.ones(n_edges, dtype=np.int8)
    
    return best_signs
