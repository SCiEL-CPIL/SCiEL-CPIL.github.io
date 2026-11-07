import numpy as np
import time

def sign_edges(graph, budget):
    """
    Returns an array of signs (-1 or 1) for each edge in the graph.
    Minimizes the spectral radius of the signed adjacency matrix.
    Ensures the 2-lift is connected.
    """
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    
    # Precompute adjacency list for faster BFS in validity check
    # adj[u] = list of (v, edge_index)
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))

    def build_adjacency(signs):
        A = np.zeros((n, n), dtype=np.float64)
        for i, (u, v) in enumerate(edges):
            A[u, v] = signs[i]
            A[v, u] = signs[i]
        return A

    def get_radius(signs):
        A = build_adjacency(signs)
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))

    def is_valid_lift(signs):
        """
        Returns True if the lift is connected (i.e., signed graph is NOT balanced).
        """
        visited = [False] * n
        color = [0] * n
        
        for start in range(n):
            if visited[start]:
                continue
            visited[start] = True
            color[start] = 0
            stack = [start]
            
            while stack:
                u = stack.pop()
                for v, i in adj[u]:
                    required_xor = 0 if signs[i] == 1 else 1
                    if not visited[v]:
                        visited[v] = True
                        color[v] = color[u] ^ required_xor
                        stack.append(v)
                    else:
                        if (color[u] ^ color[v]) != required_xor:
                            return True
        return False

    start_time = time.time()
    deadline = start_time + 85.0
    
    best_radius = float('inf')
    best_signs = None
    
    def try_update(signs, radius):
        nonlocal best_radius, best_signs
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
            
    # 1. Deterministic Initializations
    # All -1
    init_signs = np.array([-1] * num_edges, dtype=np.int8)
    if is_valid_lift(init_signs):
        r = get_radius(init_signs)
        try_update(init_signs, r)
    else:
        # Flip first edge
        test_signs = init_signs.copy()
        test_signs[0] = 1
        if is_valid_lift(test_signs):
            r = get_radius(test_signs)
            try_update(test_signs, r)
            
    # 2. Random Restart Hill Climbing
    rng = np.random.default_rng(3)
    
    while time.time() < deadline:
        # Random initialization
        current_signs = rng.choice([-1, 1], size=num_edges).astype(np.int8)
        
        if not is_valid_lift(current_signs):
            # Fix by flipping one edge
            current_signs[0] *= -1
            if not is_valid_lift(current_signs):
                # Extremely unlikely, but just in case, try another
                current_signs[1] *= -1
            
        current_radius = get_radius(current_signs)
        try_update(current_signs, current_radius)
        
        # Hill Climbing
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Random order of edges to avoid bias
            edge_order = rng.permutation(num_edges)
            for i in edge_order:
                if time.time() >= deadline:
                    break
                
                old_sign = current_signs[i]
                current_signs[i] *= -1
                
                if is_valid_lift(current_signs):
                    new_radius = get_radius(current_signs)
                    if new_radius < current_radius - 1e-12:
                        current_radius = new_radius
                        try_update(current_signs, current_radius)
                        improved = True
                        break
                    else:
                        current_signs[i] = old_sign
                else:
                    current_signs[i] = old_sign

    if best_signs is None:
        # Fallback
        s = np.array([-1] * num_edges, dtype=np.int8)
        s[0] = 1
        return s
        
    return best_signs
