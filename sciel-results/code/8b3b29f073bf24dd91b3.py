import numpy as np
import time

def sign_edges(graph, budget):
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    # Build adjacency list with edge indices
    adj = {i: [] for i in range(n)}
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
        
    # Find a spanning tree using BFS
    visited = [False] * n
    tree_edges = set()
    queue = [0]
    visited[0] = True
    
    while queue:
        u = queue.pop(0)
        for v, idx in adj[u]:
            if not visited[v]:
                visited[v] = True
                tree_edges.add(idx)
                queue.append(v)
                
    # Identify non-tree edges
    non_tree_edges = [i for i in range(num_edges) if i not in tree_edges]
    k = len(non_tree_edges)
    
    def check_lift_valid(signs):
        """Check if the lift is connected and 3-regular."""
        # Build the lift graph
        # Vertices: (v, 0) and (v, 1) for v in 0..n-1
        # For each edge (u, v) with sign s:
        #   If s = +1: (u,0)-(v,0) and (u,1)-(v,1)
        #   If s = -1: (u,0)-(v,1) and (u,1)-(v,0)
        
        lift_n = 2 * n
        lift_adj = [[] for _ in range(lift_n)]
        
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            if s == 1:
                # Equal sides
                lift_adj[u * 2].append(v * 2)
                lift_adj[v * 2].append(u * 2)
                lift_adj[u * 2 + 1].append(v * 2 + 1)
                lift_adj[v * 2 + 1].append(u * 2 + 1)
            else:
                # Opposite sides
                lift_adj[u * 2].append(v * 2 + 1)
                lift_adj[v * 2 + 1].append(u * 2)
                lift_adj[u * 2 + 1].append(v * 2)
                lift_adj[v * 2].append(u * 2 + 1)
        
        # Check 3-regular: each vertex should have degree 3
        for i in range(lift_n):
            if len(lift_adj[i]) != 3:
                return False
        
        # Check connectedness using BFS
        if lift_n == 0:
            return True
        visited_lift = [False] * lift_n
        queue = [0]
        visited_lift[0] = True
        count = 1
        
        while queue:
            u = queue.pop(0)
            for v in lift_adj[u]:
                if not visited_lift[v]:
                    visited_lift[v] = True
                    count += 1
                    queue.append(v)
        
        return count == lift_n
    
    def compute_radius(signs):
        """Compute spectral radius of signed adjacency matrix."""
        A = np.zeros((n, n))
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            A[u, v] = s
            A[v, u] = s
            
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    best_radius = float('inf')
    best_signs = None
    
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Phase 1: Try all 2^k combinations if k is small enough
    if k <= 12:
        for mask in range(1 << k):
            if time.time() > deadline:
                break
                
            # Skip the all-+1 case (mask == 0) which gives disconnected lift
            if mask == 0:
                continue
                
            signs = np.ones(num_edges, dtype=int)
            
            # Set non-tree edge signs based on mask
            for i, edge_idx in enumerate(non_tree_edges):
                if (mask >> i) & 1:
                    signs[edge_idx] = -1
                else:
                    signs[edge_idx] = 1
                    
            # Check if lift is valid
            if not check_lift_valid(signs):
                continue
                
            # Compute spectral radius
            radius = compute_radius(signs)
            
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
    else:
        # Phase 1: Local search for large k
        # Start with random valid configurations
        rng = np.random.default_rng(3)
        
        # Generate initial candidates
        candidates = []
        
        # Try some structured patterns
        for _ in range(100):
            signs = np.ones(num_edges, dtype=int)
            # Randomly flip some non-tree edges
            for edge_idx in non_tree_edges:
                if rng.random() < 0.5:
                    signs[edge_idx] = -1
            
            if check_lift_valid(signs):
                candidates.append(signs)
        
        # Also try single-flip configurations from all-+1
        for edge_idx in non_tree_edges:
            signs = np.ones(num_edges, dtype=int)
            signs[edge_idx] = -1
            if check_lift_valid(signs):
                candidates.append(signs)
        
        # Evaluate all candidates
        for signs in candidates:
            if time.time() > deadline:
                break
            radius = compute_radius(signs)
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
        
        # Local search: flip edges and keep if improvement
        if best_signs is not None:
            current_signs = best_signs.copy()
            current_radius = best_radius
            
            # Try flipping each edge
            for _ in range(50):  # Limited iterations
                if time.time() > deadline:
                    break
                    
                improved = False
                for edge_idx in range(num_edges):
                    if time.time() > deadline:
                        break
                        
                    # Flip this edge
                    old_sign = current_signs[edge_idx]
                    current_signs[edge_idx] = -old_sign
                    
                    if check_lift_valid(current_signs):
                        radius = compute_radius(current_signs)
                        if radius < current_radius - 1e-10:
                            current_radius = radius
                            best_radius = radius
                            best_signs = current_signs.copy()
                            improved = True
                        else:
                            # Revert
                            current_signs[edge_idx] = old_sign
                    else:
                        # Revert
                        current_signs[edge_idx] = old_sign
                
                if not improved:
                    break
    
    # Fallback if no valid lift found
    if best_signs is None:
        # Try to find any valid configuration
        # For prism graphs, we need at least one -1 sign
        # Try flipping each edge one at a time
        for edge_idx in range(num_edges):
            signs = np.ones(num_edges, dtype=int)
            signs[edge_idx] = -1
            if check_lift_valid(signs):
                best_signs = signs
                best_radius = compute_radius(signs)
                break
        
        if best_signs is None:
            # Last resort: try pairs of edges
            for i in range(num_edges):
                for j in range(i+1, num_edges):
                    if time.time() > deadline:
                        break
                    signs = np.ones(num_edges, dtype=int)
                    signs[i] = -1
                    signs[j] = -1
                    if check_lift_valid(signs):
                        best_signs = signs
                        best_radius = compute_radius(signs)
                        break
                if best_signs is not None:
                    break
        
        if best_signs is None:
            # Absolute fallback: all +1 except first edge
            signs = np.ones(num_edges, dtype=int)
            signs[0] = -1
            best_signs = signs
    
    return best_signs
