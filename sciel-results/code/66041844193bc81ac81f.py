import numpy as np
import time
import math

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 80  # Safety margin for 90s limit (10s buffer)
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    e_count = len(edges)
    
    # Precompute edge structure
    top_cycle_edges = []
    bottom_cycle_edges = []
    matching_edges = []
    
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_cycle_edges.append(i)
        elif u >= m and v >= m:
            bottom_cycle_edges.append(i)
        else:
            matching_edges.append(i)
            
    # Sort edges within groups to ensure consistent ordering if needed
    # The input edges are ordered, but we should map them correctly.
    # Assume top_cycle_edges[i] corresponds to edge (i, i+1 mod m)
    # We need to verify the vertex indices.
    
    def compute_radius(signs):
        """Compute spectral radius of signed adjacency matrix."""
        A = np.zeros((n, n), dtype=float)
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            A[u, v] = s
            A[v, u] = s
        eigvals = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(eigvals)))
    
    def is_valid_lift(signs):
        """Check if the lift is connected and 3-regular."""
        num_lift_vertices = 2 * n
        adj = [[] for _ in range(num_lift_vertices)]
        
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            if s == 1:
                adj[u * 2].append(v * 2)
                adj[v * 2].append(u * 2)
                adj[u * 2 + 1].append(v * 2 + 1)
                adj[v * 2 + 1].append(u * 2 + 1)
            else:
                adj[u * 2].append(v * 2 + 1)
                adj[v * 2 + 1].append(u * 2)
                adj[u * 2 + 1].append(v * 2)
                adj[v * 2].append(u * 2 + 1)
        
        # Check 3-regular
        for v in range(num_lift_vertices):
            if len(adj[v]) != 3:
                return False
        
        # Check connected (BFS from vertex 0)
        visited = [False] * num_lift_vertices
        queue = [0]
        visited[0] = True
        count = 1
        while queue:
            cur = queue.pop(0)
            for nb in adj[cur]:
                if not visited[nb]:
                    visited[nb] = True
                    count += 1
                    queue.append(nb)
        
        return count == num_lift_vertices

    # Strategy:
    # 1. If m is even, use the analytic solution: alternating cycle signs, +1 matching.
    # 2. If m is odd, use a heuristic initialization and local search.
    
    if m % 2 == 0:
        # Even m
        signs = np.ones(e_count, dtype=int)
        
        # Top cycle: alternate 1, -1
        for idx in top_cycle_edges:
            # Determine position in cycle
            # We need to know which vertex is the "start" of the edge
            u, v = edges[idx]
            # The edge connects i and i+1.
            # We can just alternate based on the smaller vertex index?
            # Since it's a cycle, we need a consistent pattern.
            # Let's map vertex index to position.
            # If u < v, pos = u. Else pos = v-1 (if v==0 and u==m-1).
            # Actually, simpler: just assign based on the index in the list if the list is ordered 0-1, 1-2...
            # The problem statement says "ordered list of (u, v)".
            # For a cycle, it's usually (0,1), (1,2), ..., (m-1, 0).
            
            # Let's determine the position k such that the edge is (k, k+1 mod m)
            if u < m and v < m:
                if v == (u + 1) % m:
                    k = u
                else:
                    k = v
            else:
                continue
                
            if k % 2 == 0:
                signs[idx] = 1
            else:
                signs[idx] = -1
                
        # Bottom cycle: alternate 1, -1
        for idx in bottom_cycle_edges:
            u, v = edges[idx]
            u_local = u - m
            v_local = v - m
            if v_local == (u_local + 1) % m:
                k = u_local
            else:
                k = v_local
                
            if k % 2 == 0:
                signs[idx] = 1
            else:
                signs[idx] = -1
                
        # Matching: all +1
        for idx in matching_edges:
            signs[idx] = 1
            
        # Validate
        if is_valid_lift(signs):
            return signs
        else:
            # Fallback to local search if invalid (should not happen for even m >= 4)
            pass
            
    # Odd m or fallback
    # Initialize with one -1 in each cycle, -1 in matching
    signs = np.ones(e_count, dtype=int)
    
    # Find the first edge in top cycle to flip
    if top_cycle_edges:
        signs[top_cycle_edges[0]] = -1
        
    # Find the first edge in bottom cycle to flip
    if bottom_cycle_edges:
        signs[bottom_cycle_edges[0]] = -1
        
    # Set all matching edges to -1
    for idx in matching_edges:
        signs[idx] = -1
        
    # Local Search
    current_signs = signs.copy()
    current_radius = compute_radius(current_signs)
    best_signs = current_signs.copy()
    best_radius = current_radius
    
    no_improve = 0
    while time.time() < deadline and no_improve < 10:
        improved = False
        for edge_idx in range(e_count):
            if time.time() > deadline:
                break
            
            new_signs = current_signs.copy()
            new_signs[edge_idx] = -new_signs[edge_idx]
            
            if not is_valid_lift(new_signs):
                continue
                
            try:
                r = compute_radius(new_signs)
                if r < current_radius - 1e-10:
                    current_signs = new_signs
                    current_radius = r
                    best_signs = new_signs.copy()
                    best_radius = r
                    improved = True
                    no_improve = 0
                    break
            except Exception:
                pass
        
        if not improved:
            no_improve += 1
    
    return best_signs
