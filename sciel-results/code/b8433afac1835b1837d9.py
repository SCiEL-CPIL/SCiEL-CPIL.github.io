import numpy as np
import time
from collections import deque

def sign_edges(graph, budget):
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    E = len(edges)
    
    # Deadline guard: 85 seconds to be safe (evaluator kills at 90s)
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Precompute edge mapping
    edge_index = {}
    for i, (u, v) in enumerate(edges):
        edge_index[frozenset((u, v))] = i
    
    # Build adjacency structure for fast signed adjacency matrix construction
    # adj[u][v] = edge index, or -1 if no edge
    adj = [[-1] * n for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u][v] = i
        adj[v][u] = i
    
    # Precompute neighbor lists for connectivity check
    neighbors = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        neighbors[u].append(v)
        neighbors[v].append(u)
    
    def compute_radius(signs):
        """Compute spectral radius of signed adjacency matrix."""
        A = np.zeros((n, n), dtype=np.float64)
        for i, (u, v) in enumerate(edges):
            s = float(signs[i])
            A[u, v] = s
            A[v, u] = s
        eigvals = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(eigvals)))
    
    def is_connected(signs):
        """Check if the 2-lift is connected using BFS on the lift graph."""
        # Lift has 2*n vertices: (v, 0) and (v, 1)
        # Edge (u, v) with sign +1 connects (u,0)-(v,0) and (u,1)-(v,1)
        # Edge (u, v) with sign -1 connects (u,0)-(v,1) and (u,1)-(v,0)
        visited = [False] * (2 * n)
        queue = deque([0])  # Start from (0, 0)
        visited[0] = True
        
        while queue:
            node = queue.popleft()
            v = node % n
            side = node // n
            for nb in neighbors[v]:
                e_idx = adj[v][nb]
                s = signs[e_idx]
                if s == 1:
                    nb_side = side
                else:
                    nb_side = 1 - side
                nb_node = nb * 2 + nb_side
                if not visited[nb_node]:
                    visited[nb_node] = True
                    queue.append(nb_node)
        
        return all(visited)
    
    def evaluate(signs):
        """Evaluate signs: return radius if valid, else None."""
        if not is_connected(signs):
            return None
        return compute_radius(signs)
    
    # 1. Structured Enumeration
    # Define the spanning tree: matching (i, i+m) and bottom path (i, i+1) for i in 0..m-2
    tree_edges = set()
    for i in range(m):
        tree_edges.add(frozenset((i, m+i)))
    for i in range(m-1):
        tree_edges.add(frozenset((i, i+1)))
    
    chord_edges = []
    chord_edges.append(frozenset((m-1, 0)))
    for i in range(m):
        chord_edges.append(frozenset((m+i, m+(i+1)%m)))
        
    num_chords = len(chord_edges)
    best_signs = None
    best_radius = float('inf')
    
    # Enumerate 2^(m+1) patterns
    for pattern in range(2 ** num_chords):
        if time.time() > deadline:
            break
            
        sign_dict = {}
        for e in tree_edges:
            sign_dict[e] = 1
        for j, e in enumerate(chord_edges):
            sign_dict[e] = 1 if (pattern >> j) & 1 else -1
            
        signs_arr = np.ones(E, dtype=np.int8)
        for i, (u, v) in enumerate(edges):
            signs_arr[i] = sign_dict[frozenset((u, v))]
        
        r = evaluate(signs_arr)
        if r is not None and r < best_radius:
            best_radius = r
            best_signs = signs_arr.copy()

    if best_signs is None:
        # Fallback
        best_signs = np.ones(E, dtype=np.int8)
        r = evaluate(best_signs)
        if r is not None:
            best_radius = r
        else:
            best_signs = -np.ones(E, dtype=np.int8)
            r = evaluate(best_signs)
            if r is not None:
                best_radius = r
            else:
                return np.ones(E, dtype=np.int8)

    # 2. Local Search - aggressive
    current_signs = best_signs.copy()
    current_radius = best_radius
    
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        for i in range(E):
            if time.time() > deadline - 10.0:
                break
                
            old_sign = current_signs[i]
            current_signs[i] = -old_sign
            
            new_radius = evaluate(current_signs)
            if new_radius is not None and new_radius < current_radius:
                current_radius = new_radius
                best_radius = new_radius
                best_signs = current_signs.copy()
                improved = True
            else:
                current_signs[i] = old_sign

    # 3. Multi-start Local Search if time permits
    if time.time() < deadline - 5.0:
        rng = np.random.default_rng(3)
        num_starts = 200
        for _ in range(num_starts):
            if time.time() > deadline - 5.0:
                break
            
            # Generate random signs
            rand_signs = rng.choice([-1, 1], size=E).astype(np.int8)
            
            r = evaluate(rand_signs)
            if r is None:
                continue
                
            if r >= best_radius:
                continue
                
            # If better, do local search from here
            current_signs = rand_signs.copy()
            current_radius = r
            best_radius = r
            best_signs = rand_signs.copy()
            
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                for i in range(E):
                    if time.time() > deadline - 5.0:
                        break
                    old_sign = current_signs[i]
                    current_signs[i] = -old_sign
                    new_radius = evaluate(current_signs)
                    if new_radius is not None and new_radius < current_radius:
                        current_radius = new_radius
                        best_radius = new_radius
                        best_signs = current_signs.copy()
                        improved = True
                    else:
                        current_signs[i] = old_sign

    # 4. Final polish: try double-flip moves (flip two edges at once) with steepest descent
    if time.time() < deadline - 2.0:
        while time.time() < deadline - 2.0:
            best_improvement_radius = best_radius
            best_improvement_signs = best_signs
            best_i = -1
            best_j = -1
            
            for i in range(E):
                if time.time() > deadline - 2.0:
                    break
                for j in range(i+1, E):
                    if time.time() > deadline - 2.0:
                        break
                    
                    test_signs = best_signs.copy()
                    test_signs[i] = -test_signs[i]
                    test_signs[j] = -test_signs[j]
                    
                    new_radius = evaluate(test_signs)
                    if new_radius is not None and new_radius < best_improvement_radius:
                        best_improvement_radius = new_radius
                        best_improvement_signs = test_signs
                        best_i = i
                        best_j = j
            
            if best_i != -1:
                best_signs = best_improvement_signs
                best_radius = best_improvement_radius
            else:
                break

    return best_signs
