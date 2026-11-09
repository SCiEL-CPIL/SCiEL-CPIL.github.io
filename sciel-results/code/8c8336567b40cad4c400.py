import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    
    # Identify edge types:
    # Prism graph: two m-cycles (0..m-1 and m..2m-1) joined by matching (i, i+m)
    # Cycle edges: (i, i+1) mod m within each cycle
    # Matching edges: (i, i+m)
    
    # We'll classify edges
    cycle_edges = []  # edges within the cycles
    matching_edges = []  # edges between the two cycles
    
    for i, (u, v) in enumerate(edges):
        if (u < m and v < m) or (u >= m and v >= m):
            cycle_edges.append(i)
        else:
            matching_edges.append(i)
    
    # Strategy: Enumerate sign assignments for matching edges
    # For a valid 2-lift, the product of signs around any cycle must be +1
    # The prism graph has cycles: top cycle, bottom cycle, and mixed cycles
    
    # Key insight: If we fix cycle edges to +1, then for the lift to be connected,
    # we need the matching edges to have an odd number of -1 signs (so the lift connects both sides)
    # Actually, let's think more carefully.
    
    # The signed adjacency matrix has eigenvalues that come in pairs ±λ.
    # The spectral radius is the max |eigenvalue|.
    
    # Let's try a different approach: enumerate all possible sign patterns for matching edges
    # and keep cycle edges at +1, then check validity and compute spectral radius.
    
    num_matching = len(matching_edges)
    best_radius = float('inf')
    best_signs = None
    
    # For small m (4, 5, 7, 9), num_matching = m, so 2^m is manageable
    # m=4: 16, m=5: 32, m=7: 128, m=9: 512
    
    for mask in range(1 << num_matching):
        if time.time() > deadline:
            if best_signs is not None:
                return best_signs
            break
        
        # Create signs array
        signs = np.ones(len(edges), dtype=int)
        
        # Cycle edges stay +1
        # Matching edges get signs based on mask
        for k, edge_idx in enumerate(matching_edges):
            if mask & (1 << k):
                signs[edge_idx] = -1
            else:
                signs[edge_idx] = 1
        
        # Check if this produces a valid connected 3-regular lift
        # For the 2-lift to be connected, the product of signs around any fundamental cycle
        # must not all be +1 in a way that disconnects.
        
        # Actually, for a prism graph with cycle edges +1 and matching edges signed,
        # the lift is connected if and only if there's an odd number of -1 matching edges.
        # Wait, that's not quite right either.
        
        # Let's just build the signed adjacency matrix and compute spectral radius.
        # We'll verify validity by checking the lift structure.
        
        # Build signed adjacency matrix
        A = np.zeros((n, n), dtype=float)
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            A[u, v] = s
            A[v, u] = s
        
        # Compute spectral radius
        eigvals = np.linalg.eigvalsh(A)
        radius = np.max(np.abs(eigvals))
        
        # Check validity: the 2-lift must be connected and 3-regular
        # For a signed graph, the 2-lift is connected if the "frustration" is non-trivial
        # Specifically, for our prism graph, if all matching edges are +1, the lift is disconnected
        # (two disconnected copies). If at least one matching edge is -1, it might be connected.
        
        # Actually, let's check connectivity of the lift directly.
        # The lift has 2n vertices: (v, 0) and (v, 1) for each vertex v.
        # Edge (u, v) with sign +1 connects (u, 0)-(v, 0) and (u, 1)-(v, 1)
        # Edge (u, v) with sign -1 connects (u, 0)-(v, 1) and (u, 1)-(v, 0)
        
        # For the lift to be connected, we need to be able to reach all 2n vertices.
        # Let's do a BFS on the lift.
        
        lift_adj = [[] for _ in range(2 * n)]
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            if s == 1:
                lift_adj[u * 2].append(v * 2)
                lift_adj[v * 2].append(u * 2)
                lift_adj[u * 2 + 1].append(v * 2 + 1)
                lift_adj[v * 2 + 1].append(u * 2 + 1)
            else:  # s == -1
                lift_adj[u * 2].append(v * 2 + 1)
                lift_adj[v * 2 + 1].append(u * 2)
                lift_adj[u * 2 + 1].append(v * 2)
                lift_adj[v * 2].append(u * 2 + 1)
        
        # BFS from (0, 0)
        visited_lift = [False] * (2 * n)
        visited_lift[0] = True
        queue = [0]
        while queue:
            node = queue.pop(0)
            for neighbor in lift_adj[node]:
                if not visited_lift[neighbor]:
                    visited_lift[neighbor] = True
                    queue.append(neighbor)
        
        if not all(visited_lift):
            # Disconnected lift, skip
            continue
        
        # Also check 3-regularity: each vertex in the lift should have degree 3
        # Since the base graph is 3-regular and each edge in the base creates exactly
        # 2 edges in the lift (one for each side), and each vertex in the lift has
        # degree equal to the degree of its base vertex, which is 3. So this is automatic.
        
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
    
    if best_signs is None:
        # Fallback: return all +1 except one matching edge set to -1
        # This should give a connected lift
        best_signs = np.ones(len(edges), dtype=int)
        if matching_edges:
            best_signs[matching_edges[0]] = -1
        else:
            # Shouldn't happen for prism graphs
            best_signs[0] = -1
    
    return best_signs
