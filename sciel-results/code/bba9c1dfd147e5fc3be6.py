import numpy as np
import time

def sign_edges(graph, budget):
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    # Classify edges
    # Tree edges: All verticals, and bottom cycle edges except (0, m-1)
    # Chord edges: Bottom (0, m-1) and all top cycle edges
    
    chord_edge_indices = []
    chord_to_pos = {}
    
    for idx, (u, v) in enumerate(edges):
        # Normalize pair to (min, max) for easy comparison
        a, b = sorted((u, v))
        
        # Check if vertical
        if abs(u - v) == m:
            # This is a tree edge
            continue
            
        # Check if bottom cycle
        if u < m and v < m:
            # The only bottom cycle edge that is a chord is the one connecting 0 and m-1
            if a == 0 and b == m - 1:
                pos = len(chord_edge_indices)
                chord_edge_indices.append(idx)
                chord_to_pos[idx] = pos
            # Other bottom cycle edges are tree edges
            continue
            
        # Check if top cycle
        if u >= m and v >= m:
            # All top cycle edges are chords
            pos = len(chord_edge_indices)
            chord_edge_indices.append(idx)
            chord_to_pos[idx] = pos
            continue
            
    k = len(chord_edge_indices)
    
    # Pre-allocate signs array
    signs = np.ones(num_edges, dtype=int)
    
    # Pre-allocate adjacency matrix base (tree edges are +1)
    # Note: We only need the signed adjacency matrix of the BASE graph for spectral radius.
    # The connectivity check requires building the LIFT graph.
    
    # Base signed adjacency matrix A (n x n)
    # Initialize with 0
    A = np.zeros((n, n), dtype=float)
    
    # Fill in tree edges as +1
    for idx, (u, v) in enumerate(edges):
        if idx not in chord_to_pos:
            A[u, v] = 1.0
            A[v, u] = 1.0
            
    # Store endpoints for chord edges for quick updating
    chord_u = [edges[idx][0] for idx in chord_edge_indices]
    chord_v = [edges[idx][1] for idx in chord_edge_indices]
    
    best_radius = float('inf')
    best_signs_mask = 0
    
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin
    
    # Iterate over all 2^k combinations
    for mask in range(1 << k):
        if time.time() > deadline:
            break
            
        # Update A and signs for this mask
        # We can just overwrite the chord entries in A
        for i in range(k):
            s = 1.0 if (mask >> i) & 1 == 0 else -1.0
            u = chord_u[i]
            v = chord_v[i]
            A[u, v] = s
            A[v, u] = s
            
            # Also update the signs array for this chord
            signs[chord_edge_indices[i]] = int(s)
            
        # Check connectivity of the 2-lift
        # Lift nodes: 0..n-1 (side 0), n..2n-1 (side 1)
        # We can represent the lift adjacency implicitly or build a small graph.
        # Since n is small (max 18), we can build adjacency list for 36 nodes.
        
        # To speed up, we can use a bitset or simple array for BFS
        lift_adj = [[] for _ in range(2 * n)]
        
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            # u0 = u, u1 = u+n
            # v0 = v, v1 = v+n
            if s == 1:
                lift_adj[u].append(v)
                lift_adj[v].append(u)
                lift_adj[u + n].append(v + n)
                lift_adj[v + n].append(u + n)
            else:
                lift_adj[u].append(v + n)
                lift_adj[v + n].append(u)
                lift_adj[u + n].append(v)
                lift_adj[v].append(u + n)
                
        # BFS on lift
        visited_lift = [False] * (2 * n)
        q = [0]
        visited_lift[0] = True
        count = 1
        while q:
            cur = q.pop(0)
            for nei in lift_adj[cur]:
                if not visited_lift[nei]:
                    visited_lift[nei] = True
                    count += 1
                    q.append(nei)
                    
        if count < 2 * n:
            # Disconnected, skip
            # Note: signs array is now partially updated for this mask, but we update it next iteration
            # so it doesn't matter as long as we reset or overwrite correctly in next loop.
            # Actually, we overwrite all chord signs in the inner loop, so it's fine.
            continue
            
        # Compute spectral radius
        # A is symmetric, so eigvalsh is appropriate
        eigvals = np.linalg.eigvalsh(A)
        radius = np.max(np.abs(eigvals))
        
        if radius < best_radius:
            best_radius = radius
            best_signs_mask = mask
            
    # Reconstruct the best signs
    # We need to ensure the signs array reflects the best_mask
    # Reset signs to +1 (tree edges)
    final_signs = np.ones(num_edges, dtype=int)
    
    for i in range(k):
        s = 1 if (best_signs_mask >> i) & 1 == 0 else -1
        final_signs[chord_edge_indices[i]] = s
        
    return final_signs
