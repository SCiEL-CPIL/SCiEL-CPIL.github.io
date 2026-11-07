"""Rewritten program: minimize spectral radius of signed prism graph 2-lifts."""
import time
import numpy as np

def _build_signed_adj(graph, signs):
    """Build signed adjacency matrix of the base graph."""
    n = graph['n']
    edges = graph['edges']
    A = np.zeros((n, n), dtype=float)
    for (u, v), s in zip(edges, signs):
        A[u, v] = s
        A[v, u] = s
    return A

def _spectral_radius(A):
    """Max |eigenvalue| of symmetric matrix A."""
    try:
        ev = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(ev)))
    except np.linalg.LinAlgError:
        return float('inf')

def _is_valid_lift(graph, signs):
    """Check if the 2-lift is connected (and 3-regular, which is automatic for prisms).
    Returns (is_valid, radius) where radius is inf if invalid."""
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    if len(signs) != len(edges):
        return False, float('inf')
    if not all(s in (-1, 1) for s in signs):
        return False, float('inf')
    
    # Build the 2-lift graph
    # Vertices: (v, side) for v in 0..n-1, side in {0,1}
    # Edge (u,v) with sign s: if s=+1, connect (u,0)-(v,0) and (u,1)-(v,1)
    #                       if s=-1, connect (u,0)-(v,1) and (u,1)-(v,0)
    num_lift_vertices = 2 * n
    adj = [[] for _ in range(num_lift_vertices)]
    
    for (u, v), s in zip(edges, signs):
        if s == 1:
            adj[u * 2].append(v * 2)
            adj[v * 2].append(u * 2)
            adj[u * 2 + 1].append(v * 2 + 1)
            adj[v * 2 + 1].append(u * 2 + 1)
        else:  # s == -1
            adj[u * 2].append(v * 2 + 1)
            adj[v * 2 + 1].append(u * 2)
            adj[u * 2 + 1].append(v * 2)
            adj[v * 2].append(u * 2 + 1)
    
    # Check connectivity via BFS
    if num_lift_vertices == 0:
        return True, float('inf')
    visited = [False] * num_lift_vertices
    queue = [0]
    visited[0] = True
    count = 1
    while queue:
        cur = queue.pop()
        for nb in adj[cur]:
            if not visited[nb]:
                visited[nb] = True
                count += 1
                queue.append(nb)
    
    if count != num_lift_vertices:
        return False, float('inf')
    
    # Compute spectral radius
    A = _build_signed_adj(graph, signs)
    radius = _spectral_radius(A)
    return True, radius

def _get_cycle_constraints(graph):
    """Return list of edge-index lists, one per independent cycle.
    For prism graph: top cycle, bottom cycle, and m 4-cycles."""
    m = graph['n'] // 2
    edges = graph['edges']
    edge_index = {}
    for i, (u, v) in enumerate(edges):
        edge_index[frozenset((u, v))] = i
    
    cycles = []
    # Top cycle: edges (i, i+1 mod m) for i in 0..m-1
    top_cycle = []
    for i in range(m):
        u, v = i, (i + 1) % m
        top_cycle.append(edge_index[frozenset((u, v))])
    cycles.append(top_cycle)
    
    # Bottom cycle: edges (m+i, m+(i+1) mod m)
    bottom_cycle = []
    for i in range(m):
        u, v = m + i, m + (i + 1) % m
        bottom_cycle.append(edge_index[frozenset((u, v))])
    cycles.append(bottom_cycle)
    
    # 4-cycles: matching_i + bottom_i + matching_{i+1} + top_{i+1}
    # matching_i = (i, m+i), bottom_i = (m+i, m+i+1), matching_{i+1} = (i+1, m+i+1), top_{i+1} = (i+1, i+2)
    # Wait, let me reconsider. The 4-cycle goes: i -> m+i -> m+i+1 -> i+1 -> i
    # Edges: (i, m+i), (m+i, m+i+1), (m+i+1, i+1), (i+1, i)
    for i in range(m):
        i_next = (i + 1) % m
        mi = m + i
        mi_next = m + i_next
        cycle_edges = []
        # (i, mi)
        cycle_edges.append(edge_index[frozenset((i, mi))])
        # (mi, mi_next)
        cycle_edges.append(edge_index[frozenset((mi, mi_next))])
        # (mi_next, i_next)
        cycle_edges.append(edge_index[frozenset((mi_next, i_next))])
        # (i_next, i)
        cycle_edges.append(edge_index[frozenset((i_next, i))])
        cycles.append(cycle_edges)
    
    return cycles

def _parity_ok(signs, cycles):
    """Check that product of signs on each cycle is +1."""
    for cycle in cycles:
        prod = 1
        for idx in cycle:
            prod *= signs[idx]
        if prod != 1:
            return False
    return True

def sign_edges(graph, budget):
    deadline = time.time() + 85  # 85s out of 90s limit
    
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    # Build edge index
    edge_index = {}
    for i, (u, v) in enumerate(edges):
        edge_index[frozenset((u, v))] = i
    
    # Identify edge groups: top cycle, bottom cycle, matching
    top_edge_idx = []
    bottom_edge_idx = []
    matching_edge_idx = []
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_edge_idx.append(i)
        elif u >= m and v >= m:
            bottom_edge_idx.append(i)
        else:
            matching_edge_idx.append(i)
    
    # Get cycles for connectivity check
    cycles = _get_cycle_constraints(graph)
    
    # Initialize: all +1
    best_signs = np.ones(num_edges, dtype=int)
    best_radius = float('inf')
    
    # Try all +1 first
    valid, radius = _is_valid_lift(graph, best_signs)
    if valid:
        best_radius = radius
    
    # For even m, try to improve by flipping pairs
    if m % 2 == 0:
        # We can flip (top_edge_j, bottom_edge_j) pair to preserve parity
        # Let's do a local search: start from all +1, try flipping pairs
        current = best_signs.copy()
        current_radius = best_radius
        
        # Random restarts with different initial configurations
        rng = np.random.default_rng(42)
        
        for restart in range(50):
            if time.time() > deadline:
                break
            
            # Start from a random valid configuration
            if restart == 0:
                cur = np.ones(num_edges, dtype=int)
            else:
                cur = best_signs.copy()
                # Randomly flip some pairs
                for _ in range(rng.integers(0, m // 2 + 1)):
                    j = rng.integers(0, m)
                    cur[top_edge_idx[j]] *= -1
                    cur[bottom_edge_idx[j]] *= -1
            
            # Local search: try flipping individual pairs
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Shuffle order for diversity
                order = rng.permutation(m)
                for j in order:
                    if time.time() > deadline:
                        break
                    # Try flipping pair j
                    old_top = cur[top_edge_idx[j]]
                    old_bot = cur[bottom_edge_idx[j]]
                    cur[top_edge_idx[j]] *= -1
                    cur[bottom_edge_idx[j]] *= -1
                    
                    valid, r = _is_valid_lift(graph, cur)
                    if valid and r < current_radius - 1e-12:
                        current_radius = r
                        improved = True
                    else:
                        # Revert
                        cur[top_edge_idx[j]] = old_top
                        cur[bottom_edge_idx[j]] = old_bot
            
            # Update best
            valid, r = _is_valid_lift(graph, cur)
            if valid and r < best_radius - 1e-12:
                best_radius = r
                best_signs = cur.copy()
        
        # Also try flipping matching edges in pairs with appropriate adjustments
        # Actually, let's also try a different approach: 
        # For even m, we can have an even number of -1s in top and bottom independently
        # Let's try configurations with specific numbers of -1s
        
        # Try: k -1s in top, k -1s in bottom (same positions)
        for k in range(0, m // 2 + 1):
            if time.time() > deadline:
                break
            # Try all combinations would be too expensive for large m
            # Instead, try a few random ones
            for trial in range(20):
                if time.time() > deadline:
                    break
                cur = np.ones(num_edges, dtype=int)
                if k > 0:
                    positions = rng.choice(m, size=k, replace=False)
                    for j in positions:
                        cur[top_edge_idx[j]] = -1
                        cur[bottom_edge_idx[j]] = -1
                valid, r = _is_valid_lift(graph, cur)
                if valid and r < best_radius - 1e-12:
                    best_radius = r
                    best_signs = cur.copy()
    
    # For odd m, all +1 is the only 3-regular lift (or at least, it's hard to find others)
    # But let's still try some random searches in case we missed something
    if m % 2 == 1:
        # For odd m, a 3-regular lift requires even number of -1s in top and bottom
        # But m is odd, so we can have 2, 4, ... -1s
        # However, the 4-cycle constraints may force all +1
        # Let's verify: if top has even number of -1s and bottom has even number of -1s,
        # and the 4-cycles are satisfied, is the lift connected?
        # For m odd, let's try a few configurations
        rng = np.random.default_rng(123)
        for trial in range(100):
            if time.time() > deadline:
                break
            cur = np.ones(num_edges, dtype=int)
            # Try flipping 2 pairs
            if m >= 2:
                pos1 = rng.integers(0, m)
                pos2 = rng.integers(0, m)
                if pos1 != pos2:
                    cur[top_edge_idx[pos1]] = -1
                    cur[bottom_edge_idx[pos1]] = -1
                    cur[top_edge_idx[pos2]] = -1
                    cur[bottom_edge_idx[pos2]] = -1
            valid, r = _is_valid_lift(graph, cur)
            if valid and r < best_radius - 1e-12:
                best_radius = r
                best_signs = cur.copy()
    
    return best_signs
