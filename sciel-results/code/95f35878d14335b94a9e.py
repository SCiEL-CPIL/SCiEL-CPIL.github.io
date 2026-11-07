import numpy as np
import time
import sys

def sign_edges(graph, budget):
    """
    Assigns signs to edges of a prism graph to minimize the spectral radius
    of the signed adjacency matrix, ensuring the 2-lift is connected.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    
    # Build adjacency list for connectivity check
    adj = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
        
    # Precompute edge endpoints for fast matrix building
    edge_u = np.array([e[0] for e in edges])
    edge_v = np.array([e[1] for e in edges])
    
    def is_lift_connected(signs):
        """
        Returns True if the 2-lift is connected.
        The lift is connected iff the signed graph is NOT balanced.
        Balanced means there exists x in {-1, 1}^n such that s_uv = x_u * x_v for all edges.
        """
        x = [0] * n
        x[0] = 1
        stack = [0]
        
        while stack:
            u = stack.pop()
            xu = x[u]
            for v, idx in adj[u]:
                s = signs[idx]
                expected_xv = s * xu
                if x[v] == 0:
                    x[v] = expected_xv
                    stack.append(v)
                else:
                    if x[v] != expected_xv:
                        return True # Conflict -> Not Balanced -> Connected
        
        return False # No Conflict -> Balanced -> Disconnected

    def spectral_radius(signs):
        """
        Computes the spectral radius of the signed adjacency matrix.
        """
        A = np.zeros((n, n), dtype=float)
        # Vectorized update is tricky with sparse updates, but n is small.
        # For n=18, direct assignment is fast enough.
        # A[u, v] = sign
        # Since graph is undirected, A is symmetric.
        A[edge_u, edge_v] = signs
        A[edge_v, edge_u] = signs
        
        # eigvalsh is for symmetric matrices
        eigs = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigs))

    best_signs = None
    best_radius = float('inf')
    
    # Simulated Annealing Parameters
    T_start = 1.0
    T_end = 0.001
    alpha = 0.99
    max_iters = 5000
    num_restarts = 20
    
    rng = np.random.default_rng(3)
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize signs randomly
        signs = rng.choice([-1, 1], size=num_edges).astype(int)
        
        # Check initial validity
        if not is_lift_connected(signs):
            # Try to fix by flipping a random edge if it's balanced
            # If balanced, flipping any edge makes it unbalanced?
            # Not necessarily, but it changes the product on cycles containing that edge.
            # Just flip a random edge.
            flip_idx = rng.integers(0, num_edges)
            signs[flip_idx] *= -1
            if not is_lift_connected(signs):
                # Still balanced? Unlikely for random, but possible if structure is specific.
                # Continue trying or skip.
                continue

        current_radius = spectral_radius(signs)
        current_signs = signs.copy()
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_signs = current_signs.copy()
            
        T = T_start
        for i in range(max_iters):
            if time.time() > deadline:
                break
                
            # Pick a random edge to flip
            flip_idx = rng.integers(0, num_edges)
            old_sign = current_signs[flip_idx]
            new_sign = -old_sign
            
            # Check connectivity of new state
            # To speed up, we could update the BFS state incrementally, but n is small.
            # Just re-check.
            test_signs = current_signs.copy()
            test_signs[flip_idx] = new_sign
            
            if not is_lift_connected(test_signs):
                # Invalid move, skip
                T *= alpha
                continue
                
            new_radius = spectral_radius(test_signs)
            delta = new_radius - current_radius
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_signs = test_signs
                current_radius = new_radius
                
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_signs = current_signs.copy()
                    
            T *= alpha
            
    if best_signs is None:
        # Fallback: Return all -1s? Or random?
        # All -1s on a bipartite graph? Prism graphs are not bipartite for odd m?
        # m=4, 5, 7, 9.
        # If m is odd, prism is non-bipartite.
        # If m is even, prism is bipartite?
        # Y_4: 2 squares + rungs. It is bipartite.
        # If the graph is bipartite, assigning -1 to all edges makes it anti-regular?
        # Anyway, just return a random valid one.
        for _ in range(100):
            signs = rng.choice([-1, 1], size=num_edges).astype(int)
            if is_lift_connected(signs):
                best_signs = signs
                break
        if best_signs is None:
            best_signs = np.ones(num_edges, dtype=int) # Fallback, might be invalid but better than crash

    return best_signs
