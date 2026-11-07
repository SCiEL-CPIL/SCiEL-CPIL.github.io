# EVOLVE-BLOCK-START
import time
import numpy as np

def _is_connected_lift(graph, signs):
    n = graph['n']
    edges = graph['edges']
    # Build adjacency list for the lift (2n vertices)
    # Vertices 0..n-1 are side 0, n..2n-1 are side 1
    adj = [[] for _ in range(2 * n)]
    for i, (u, v) in enumerate(edges):
        s = signs[i]
        if s == 1:
            adj[u].append(v)
            adj[v].append(u)
            adj[u + n].append(v + n)
            adj[v + n].append(u + n)
        else:
            adj[u].append(v + n)
            adj[v].append(u + n)
            adj[u + n].append(v)
            adj[v + n].append(u)
    
    # BFS
    visited = [False] * (2 * n)
    queue = [0]
    visited[0] = True
    count = 1
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                count += 1
                queue.append(v)
    return count == 2 * n

def _spectral_radius_of_signed_adj(graph, signs):
    n = graph['n']
    edges = graph['edges']
    S = np.zeros((n, n))
    for i, (u, v) in enumerate(edges):
        s = signs[i]
        S[u, v] = s
        S[v, u] = s
    eigvals = np.linalg.eigvalsh(S)
    return np.max(np.abs(eigvals))

def _evaluate(graph, signs):
    if not _is_connected_lift(graph, signs):
        return float('inf')
    return _spectral_radius_of_signed_adj(graph, signs)

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    
    best_signs = None
    best_radius = float('inf')
    
    # Use multiple restarts with local search
    rng = np.random.default_rng(42)
    
    # Number of restarts depends on time.
    # Each restart involves local search.
    # We will run restarts until deadline.
    
    for restart in range(1000): # Cap restarts, but break on time
        if time.time() > deadline:
            break
            
        # Initialize with random signs
        signs = rng.choice([-1, 1], size=num_edges).astype(int)
        
        # Check if valid and get radius
        current_radius = _evaluate(graph, signs)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_signs = signs.copy()
            
        if current_radius == float('inf'):
            continue
            
        # Local search: Single edge flips
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for i in range(num_edges):
                if time.time() > deadline:
                    break
                # Try flipping edge i
                new_signs = signs.copy()
                new_signs[i] *= -1
                
                new_radius = _evaluate(graph, new_signs)
                
                if new_radius < current_radius:
                    signs = new_signs
                    current_radius = new_radius
                    improved = True
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_signs = signs.copy()
        
        # Optional: 2-edge flips if time remains
        # For now, stick to 1-edge to ensure we finish.
        
    if best_signs is None:
        # Fallback: all +1
        best_signs = np.ones(num_edges, dtype=int)
        # Check if all +1 is valid
        if not _is_connected_lift(graph, best_signs):
             # Try to find any valid one
             # If all +1 is disconnected (which it is if the graph is bipartite? No, prism is not bipartite for odd m? 
             # Prism Y_m is bipartite if m is even.
             # If m is even, all +1 lift is two disjoint copies of Y_m, which is connected? No, Y_m is connected, so lift is two copies, disconnected.
             # So all +1 is invalid for even m.
             # For odd m, Y_m is not bipartite? 
             # C_m is bipartite if m is even.
             # If m is odd, C_m is not bipartite.
             # The lift of a non-bipartite graph with all +1 signs is two disjoint copies of the graph.
             # So it is disconnected.
             # Thus, all +1 is ALWAYS disconnected for the lift?
             # Wait. "The lift must be connected".
             # If signs are all +1, the lift is two disjoint copies of the base graph.
             # So it is disconnected.
             # So we MUST have some -1 signs to connect the two sides?
             # Yes.
             
             # Random search fallback
             for _ in range(1000):
                 cand = rng.choice([-1, 1], size=num_edges).astype(int)
                 if _is_connected_lift(graph, cand):
                     best_signs = cand
                     break
             
    return best_signs
# EVOLVE-BLOCK-END
