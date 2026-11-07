import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    
    # Map edge indices to their logical role
    # Roles:
    # 'top_path': (i, i+1) for i in 0..m-2
    # 'bot_path': (i+m, i+1+m) for i in 0..m-2
    # 'match_0': (0, m)
    # 'top_close': (m-1, 0)
    # 'bot_close': (2m-1, m)
    # 'match_i': (i, i+m) for i in 1..m-1
    
    edge_roles = {}
    non_tree_indices = []
    non_tree_roles = []
    
    for idx, (u, v) in enumerate(edges):
        u, v = int(u), int(v)
        # Normalize u, v to be consistent? The graph is undirected.
        # But the edge list is ordered. We just need to identify the edge.
        
        # Check if it's a matching edge
        if abs(u - v) == m:
            top_v = min(u, v)
            if top_v == 0:
                edge_roles[idx] = 'match_0'
            else:
                edge_roles[idx] = f'match_{top_v}'
                non_tree_indices.append(idx)
                non_tree_roles.append(f'match_{top_v}')
        else:
            # It's a cycle edge
            if u < m and v < m:
                # Top cycle
                # Identify if it's path or close
                # Path: (i, i+1)
                # Close: (m-1, 0)
                # Note: (0, m-1) is the same as (m-1, 0)
                if (u == m-1 and v == 0) or (u == 0 and v == m-1):
                    edge_roles[idx] = 'top_close'
                    non_tree_indices.append(idx)
                    non_tree_roles.append('top_close')
                else:
                    edge_roles[idx] = 'top_path'
            elif u >= m and v >= m:
                # Bottom cycle
                if (u == 2*m-1 and v == m) or (u == m and v == 2*m-1):
                    edge_roles[idx] = 'bot_close'
                    non_tree_indices.append(idx)
                    non_tree_roles.append('bot_close')
                else:
                    edge_roles[idx] = 'bot_path'
            else:
                # Should not happen for prism
                edge_roles[idx] = 'unknown'
                
    # Sanity check: ensure we have all non-tree edges
    # Expected non-tree count: m + 1
    # My logic:
    # match_0 is tree.
    # match_i (i=1..m-1) are non-tree. Count: m-1.
    # top_close is non-tree. Count: 1.
    # bot_close is non-tree. Count: 1.
    # Total: m+1.
    
    # If any edge is 'unknown' or missing, fallback to random/heuristic
    if 'unknown' in edge_roles.values() or len(non_tree_indices) != m + 1:
        # Fallback: Simple heuristic
        signs = np.ones(n_edges, dtype=int)
        for i, (u, v) in enumerate(edges):
            if abs(u - v) == m:
                top = u if u < m else v
                if top % 2 == 0:
                    signs[i] = -1
        return signs

    # Precompute adjacency list for connectivity check
    # And build the signed adjacency matrix template
    
    n = 2 * m
    # Base adjacency matrix (0 or 1)
    A_base = np.zeros((n, n), dtype=int)
    for u, v in edges:
        A_base[u, v] = 1
        A_base[v, u] = 1
        
    # List of non-tree edge indices to iterate over
    # We will iterate over all 2^(m+1) combinations
    
    best_radius = float('inf')
    best_signs = None
    
    # For speed, we can precompute the eigenvalues for the tree part?
    # No, the signs change the matrix structure significantly.
    # But n is small (max 18). eigvalsh is fast.
    
    num_non_tree = len(non_tree_indices)
    total_combinations = 1 << num_non_tree
    
    # Optimization: We can break early if time is running out
    for mask in range(total_combinations):
        if time.time() > deadline:
            break
            
        # Construct signs
        signs = np.ones(n_edges, dtype=int)
        
        # Assign signs to non-tree edges based on mask
        for j, idx in enumerate(non_tree_indices):
            if (mask >> j) & 1:
                signs[idx] = -1
            else:
                signs[idx] = 1
                
        # Tree edges are +1 (already set by np.ones)
        
        # Check connectivity
        # Build signed adjacency matrix
        A_signed = np.zeros((n, n), dtype=float)
        for i, (u, v) in enumerate(edges):
            s = float(signs[i])
            A_signed[u, v] = s
            A_signed[v, u] = s
            
        # Check connectivity of the lift
        # The lift has 2n vertices.
        # We can check if the graph defined by A_signed is "connected" in the lift sense.
        # Actually, it's easier to check the cycle products.
        # But BFS on the lift is robust.
        # Lift vertices: (v, 0), (v, 1).
        # Edge (u, v) with sign s connects:
        # (u, 0) to (v, 0) if s=1, or (v, 1) if s=-1.
        # (u, 1) to (v, 1) if s=1, or (v, 0) if s=-1.
        
        # Since n is small, we can just do a BFS on the 2n nodes.
        lift_adj = [[] for _ in range(2 * n)]
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            if s == 1:
                lift_adj[2 * u].append(2 * v)
                lift_adj[2 * v].append(2 * u)
                lift_adj[2 * u + 1].append(2 * v + 1)
                lift_adj[2 * v + 1].append(2 * u + 1)
            else:
                lift_adj[2 * u].append(2 * v + 1)
                lift_adj[2 * v + 1].append(2 * u)
                lift_adj[2 * u + 1].append(2 * v)
                lift_adj[2 * v].append(2 * u + 1)
                
        # BFS from 0
        visited = [False] * (2 * n)
        queue = [0]
        visited[0] = True
        while queue:
            curr = queue.pop(0)
            for neighbor in lift_adj[curr]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
                    
        if not all(visited):
            continue
            
        # Compute spectral radius
        # eigvalsh is for symmetric matrices. A_signed is symmetric.
        try:
            eigenvalues = np.linalg.eigvalsh(A_signed)
            radius = np.max(np.abs(eigenvalues))
        except np.linalg.LinAlgError:
            continue
            
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
            
    if best_signs is None:
        # Fallback if no valid lift found (should not happen)
        signs = np.ones(n_edges, dtype=int)
        for i, (u, v) in enumerate(edges):
            if abs(u - v) == m:
                top = u if u < m else v
                if top % 2 == 0:
                    signs[i] = -1
        return signs
        
    return best_signs
