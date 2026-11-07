"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def sign_edges(graph, budget):
    from autoresearch_suite.science_tcs_expansion import check_lift
    
    deadline = time.time() + 85.0
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2  # prism has 2m vertices
    
    # Identify edge types: top cycle, bottom cycle, matching
    # Top cycle: edges between vertices 0..m-1
    # Bottom cycle: edges between vertices m..2m-1
    # Matching: edges (i, m+i)
    
    top_cycle_edges = []   # indices of edges in top cycle
    bottom_cycle_edges = [] # indices of edges in bottom cycle
    matching_edges = []     # indices of matching edges (i, m+i)
    top_cycle_order = []    # for each top cycle edge index, which position in cycle
    bottom_cycle_order = [] # for each bottom cycle edge index, which position in cycle
    matching_order = []     # for each matching edge index, which i it connects
    
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_cycle_edges.append(idx)
        elif u >= m and v >= m:
            bottom_cycle_edges.append(idx)
        else:
            matching_edges.append(idx)
    
    # Determine cycle ordering
    # Top cycle edges connect i to i+1 mod m
    top_adj = {}
    for idx in top_cycle_edges:
        u, v = edges[idx]
        top_adj.setdefault(u, []).append((v, idx))
        top_adj.setdefault(v, []).append((u, idx))
    
    # Walk the top cycle
    top_cycle_order = [-1] * len(top_cycle_edges)
    if top_cycle_edges:
        start = edges[top_cycle_edges[0]][0]
        current = start
        visited_edges = set()
        for pos in range(m):
            next_v = None
            for idx in top_cycle_edges:
                if idx in visited_edges:
                    continue
                u, v = edges[idx]
                if u == current:
                    next_v = v
                    visited_edges.add(idx)
                    top_cycle_order[top_cycle_edges.index(idx)] = pos
                    break
                elif v == current:
                    next_v = u
                    visited_edges.add(idx)
                    top_cycle_order[top_cycle_edges.index(idx)] = pos
                    break
            if next_v is None:
                break
            current = next_v
    
    # Same for bottom cycle
    bottom_cycle_order = [-1] * len(bottom_cycle_edges)
    if bottom_cycle_edges:
        start = edges[bottom_cycle_edges[0]][0]
        current = start
        visited_edges = set()
        for pos in range(m):
            next_v = None
            for idx in bottom_cycle_edges:
                if idx in visited_edges:
                    continue
                u, v = edges[idx]
                if u == current:
                    next_v = v
                    visited_edges.add(idx)
                    bottom_cycle_order[bottom_cycle_edges.index(idx)] = pos
                    break
                elif v == current:
                    next_v = u
                    visited_edges.add(idx)
                    bottom_cycle_order[bottom_cycle_edges.index(idx)] = pos
                    break
            if next_v is None:
                break
            current = next_v
    
    # Matching edges: (i, m+i)
    matching_order = [-1] * len(matching_edges)
    for idx in matching_edges:
        u, v = edges[idx]
        i = min(u, v)
        matching_order[matching_edges.index(idx)] = i
    
    # Fast connectivity check using sign products
    # The 2-lift is connected iff the monodromy is non-trivial.
    # For a prism, check: product around top cycle, bottom cycle, or a ladder cycle.
    def is_valid_lift(signs):
        """Check if the 2-lift is connected using sign product criterion."""
        # Product of signs around top cycle
        top_prod = 1
        for idx in top_cycle_edges:
            top_prod *= signs[idx]
        
        # Product of signs around bottom cycle
        bottom_prod = 1
        for idx in bottom_cycle_edges:
            bottom_prod *= signs[idx]
        
        # For the lift to be connected, we need the holonomy to be non-trivial
        # The key cycles are: top cycle, bottom cycle, and ladder cycles
        # A ladder cycle: top edge (i,i+1), matching (i+1, m+i+1), bottom edge (m+i+1, m+i), matching (m+i, i)
        # Sign product for ladder cycle at position i:
        # = top_sign(i) * matching_sign(i+1) * bottom_sign(i) * matching_sign(i)
        
        # If top_prod == -1 or bottom_prod == -1, the lift is connected
        if top_prod == -1 or bottom_prod == -1:
            return True
        
        # Otherwise check ladder cycles
        # Build lookup: for each position i in cycle, get the edge sign
        top_signs_by_pos = {}
        for idx in top_cycle_edges:
            pos = top_cycle_order[top_cycle_edges.index(idx)]
            if pos >= 0:
                top_signs_by_pos[pos] = signs[idx]
        
        bottom_signs_by_pos = {}
        for idx in bottom_cycle_edges:
            pos = bottom_cycle_order[bottom_cycle_edges.index(idx)]
            if pos >= 0:
                bottom_signs_by_pos[pos] = signs[idx]
        
        match_signs_by_i = {}
        for idx in matching_edges:
            i = matching_order[matching_edges.index(idx)]
            if i >= 0:
                match_signs_by_i[i] = signs[idx]
        
        # Check ladder cycle at position 0: 
        # top edge at pos 0 (connects 0-1), matching at 1, bottom edge at pos 0 (connects m-1 to m? need to check orientation)
        # This is getting complex. Let's just use BFS for safety but with a faster implementation.
        
        # Fast BFS on the 2-lift
        lift_n = 2 * n
        # Use bitset for visited
        visited = [False] * lift_n
        queue = [0]
        visited[0] = True
        count = 1
        
        # Pre-build adjacency as flat arrays for speed
        # Each vertex in lift has degree 3
        lift_adj = [[0, 0, 0] for _ in range(lift_n)]
        
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            if s == 1:
                lift_adj[u*2][0] = v*2
                lift_adj[v*2][0] = u*2
                lift_adj[u*2+1][0] = v*2+1
                lift_adj[v*2+1][0] = u*2+1
            else:
                lift_adj[u*2][0] = v*2+1
                lift_adj[v*2+1][0] = u*2
                lift_adj[u*2+1][0] = v*2
                lift_adj[v*2][0] = u*2+1
        
        # BFS
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            for neighbor in lift_adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    count += 1
                    queue.append(neighbor)
        
        return count == lift_n
    
    # Fast spectral radius using block structure
    # S = [[A_top, C], [C, A_bottom]] where C is diagonal matching
    # Use eigvalsh on the full matrix but build it fast
    def compute_radius_fast(signs):
        """Compute spectral radius of signed adjacency matrix."""
        S = np.zeros((n, n), dtype=np.float64)
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            S[u, v] = s
            S[v, u] = s
        eigvals = np.linalg.eigvalsh(S)
        return float(np.max(np.abs(eigvals)))
    
    # Start with random search
    rng = np.random.default_rng(3)
    
    best_signs = None
    best_radius = np.inf
    
    # Sample random sign vectors
    for _ in range(budget):
        signs = rng.choice([-1, 1], size=num_edges).astype(np.int8)
        
        if not is_valid_lift(signs):
            continue
        
        radius = compute_radius_fast(signs)
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
    
    if best_signs is None:
        # Fallback: try single -1 flips
        for i in range(num_edges):
            signs = np.ones(num_edges, dtype=np.int8)
            signs[i] = -1
            if is_valid_lift(signs):
                radius = compute_radius_fast(signs)
                if radius < best_radius:
                    best_radius = radius
                    best_signs = signs.copy()
    
    if best_signs is None:
        return np.ones(num_edges, dtype=np.int8)
    
    # Local search with 1-flip and 2-flip moves
    current_signs = best_signs.copy()
    current_radius = best_radius
    
    # Pre-compute for fast flip evaluation
    while time.time() < deadline - 2.0:
        improved = False
        
        # 1-flip moves
        for i in range(num_edges):
            if time.time() >= deadline - 2.0:
                break
            
            test_signs = current_signs.copy()
            test_signs[i] = -test_signs[i]
            
            if not is_valid_lift(test_signs):
                continue
            
            radius = compute_radius_fast(test_signs)
            if radius < current_radius - 1e-10:
                current_signs = test_signs
                current_radius = radius
                best_signs = test_signs.copy()
                best_radius = radius
                improved = True
        
        if not improved:
            # 2-flip moves
            two_flip_improved = False
            for i in range(num_edges):
                if time.time() >= deadline - 2.0:
                    break
                for j in range(i + 1, num_edges):
                    if time.time() >= deadline - 2.0:
                        break
                    
                    test_signs = current_signs.copy()
                    test_signs[i] = -test_signs[i]
                    test_signs[j] = -test_signs[j]
                    
                    if not is_valid_lift(test_signs):
                        continue
                    
                    radius = compute_radius_fast(test_signs)
                    if radius < current_radius - 1e-10:
                        current_signs = test_signs
                        current_radius = radius
                        best_signs = test_signs.copy()
                        best_radius = radius
                        two_flip_improved = True
                        break
                if two_flip_improved:
                    break
            
            if not two_flip_improved:
                # Random restarts
                found_improvement = False
                for _ in range(200):
                    if time.time() >= deadline - 2.0:
                        break
                    
                    signs = rng.choice([-1, 1], size=num_edges).astype(np.int8)
                    
                    if not is_valid_lift(signs):
                        continue
                    
                    radius = compute_radius_fast(signs)
                    if radius < current_radius - 1e-10:
                        current_signs = signs
                        current_radius = radius
                        best_signs = signs.copy()
                        best_radius = radius
                        found_improvement = True
                        break
                
                if not found_improvement:
                    break
    
    return best_signs
# EVOLVE-BLOCK-END
