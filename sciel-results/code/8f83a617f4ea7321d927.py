"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_signed_adjacency(graph, signs):
    """Build the n x n signed adjacency matrix."""
    n = graph['n']
    edges = graph['edges']
    S = np.zeros((n, n))
    for i, (u, v) in enumerate(edges):
        s = signs[i]
        S[u, v] = s
        S[v, u] = s
    return S

def _spectral_radius(S):
    """Compute spectral radius of symmetric matrix S."""
    eigvals = np.linalg.eigvalsh(S)
    return np.max(np.abs(eigvals))

def _is_connected_lift_fast(graph, signs):
    """Fast connectivity check for the 2-lift using cycle products."""
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    
    # The lift is connected iff the signing is not a coboundary.
    # For a prism graph, we can check: the product of signs around any fundamental cycle
    # must not all be +1.
    # 
    # Actually, for the 2-lift of a graph G, the lift is connected iff there exists
    # at least one cycle in G whose product of edge signs is -1.
    # 
    # For a prism graph, the fundamental cycles are:
    # - The top m-cycle
    # - The bottom m-cycle
    # - For each i, the 4-cycle: i -> i+1 -> m+i+1 -> m+i -> i
    # 
    # But we just need at least one cycle with product -1.
    # 
    # Simplest check: build the lift adjacency and do BFS. But let's do it smarter.
    # 
    # Alternative: the 2-lift is disconnected iff there exists a vertex labeling phi: V -> {0,1}
    # such that for every edge (u,v) with sign sigma, phi(u) XOR phi(v) = 0 if sigma=+1, 
    # phi(u) XOR phi(v) = 1 if sigma=-1.
    # This is a system of linear equations over GF(2). The lift is connected iff this system
    # has no solution (or equivalently, the constraints are inconsistent).
    # 
    # Actually, the lift is disconnected iff the system IS consistent (has a solution).
    # If consistent, vertices split into two components based on phi.
    # If inconsistent, the lift is connected.
    # 
    # So: lift is connected iff the GF(2) system is inconsistent.
    
    # Build the system: for each edge (u,v) with sign sigma:
    # x_u XOR x_v = 1 if sigma = -1, else 0
    # 
    # Use union-find with parity, or just solve the linear system.
    # For small graphs, let's use a simple BFS-based approach on the constraint graph.
    
    # x_u XOR x_v = b_e where b_e = 0 if sigma=+1, 1 if sigma=-1
    # 
    # Start with x_0 = 0, propagate. If we get a contradiction, system is inconsistent -> connected.
    
    parent = list(range(n))
    rank = [0] * n
    parity = [0] * n  # parity[i] = x_i XOR x_parent[i]
    
    def find(x):
        if parent[x] != x:
            root = find(parent[x])
            parity[x] ^= parity[parent[x]]
            parent[x] = root
        return parent[x]
    
    def union(x, y, b):
        """Enforce x XOR y = b."""
        rx = find(x)
        ry = find(y)
        if rx == ry:
            # Check consistency
            return (parity[x] ^ parity[y]) == b
        # Union by rank
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
            b = b ^ 1  # Swap x and y
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        # Attach ry under rx
        # x XOR y = b, and x is in rx, y is in ry
        # We want parity[ry] such that the constraint holds
        # After union, find(y) will return rx, and parity[y] will be updated
        # Currently: x XOR find(x) = parity[x], y XOR find(y) = parity[y]
        # We want: x XOR y = b
        # x = parity[x] XOR rx (conceptually)
        # y = parity[y] XOR ry
        # x XOR y = parity[x] XOR parity[y] XOR rx XOR ry
        # But rx and ry are just labels, what matters is the parity chain
        # Set parent[ry] = rx, and parity[ry] = parity[x] XOR parity[y] XOR b
        parent[ry] = rx
        parity[ry] = parity[x] ^ parity[y] ^ b
        return True
    
    for i, (u, v) in enumerate(edges):
        b = 0 if signs[i] == 1 else 1
        if not union(u, v, b):
            # Inconsistent -> lift is connected
            return True
    
    # Consistent -> lift is disconnected
    return False

def _local_search(graph, signs, deadline):
    """Local search: flip single edges to reduce spectral radius while maintaining connectivity."""
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    
    S = _build_signed_adjacency(graph, signs)
    best_radius = _spectral_radius(S)
    best_signs = signs.copy()
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(num_edges):
            if time.time() > deadline:
                break
            # Try flipping edge i
            old_sign = signs[i]
            signs[i] = -old_sign
            
            # Check connectivity
            if not _is_connected_lift_fast(graph, signs):
                signs[i] = old_sign
                continue
            
            S = _build_signed_adjacency(graph, signs)
            radius = _spectral_radius(S)
            
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
                improved = True
            else:
                signs[i] = old_sign
    
    return best_signs, best_radius

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    # Identify edge types
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
    
    top_cycle_edges.sort()
    bottom_cycle_edges.sort()
    matching_edges.sort()
    
    # Gauge fixing: set matching edges to +1, first top cycle edge to +1
    fixed_signs = np.ones(num_edges, dtype=int)
    for i in matching_edges:
        fixed_signs[i] = 1
    if top_cycle_edges:
        fixed_signs[top_cycle_edges[0]] = 1
    
    free_edges = []
    for i in top_cycle_edges[1:]:
        free_edges.append(i)
    for i in bottom_cycle_edges:
        free_edges.append(i)
    
    num_free = len(free_edges)
    
    best_signs = None
    best_radius = float('inf')
    
    # For small m, do exhaustive search; for larger m, use random + local search
    if num_free <= 18:
        # Exhaustive search over free variables
        for mask in range(1 << num_free):
            if time.time() > deadline:
                break
            
            signs = fixed_signs.copy()
            for j, edge_idx in enumerate(free_edges):
                signs[edge_idx] = -1 if (mask >> j) & 1 else 1
            
            if not _is_connected_lift_fast(graph, signs):
                continue
            
            S = _build_signed_adjacency(graph, signs)
            radius = _spectral_radius(S)
            
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
    else:
        # Random initialization + local search
        rng = np.random.default_rng(3)
        for attempt in range(min(budget, 200)):
            if time.time() > deadline:
                break
            
            signs = fixed_signs.copy()
            for edge_idx in free_edges:
                signs[edge_idx] = rng.choice([-1, 1])
            
            if not _is_connected_lift_fast(graph, signs):
                continue
            
            S = _build_signed_adjacency(graph, signs)
            radius = _spectral_radius(S)
            
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
        
        # Local search refinement
        if best_signs is not None:
            best_signs, best_radius = _local_search(graph, best_signs, deadline)
    
    # If we still don't have a valid solution, try random without gauge fixing
    if best_signs is None:
        rng = np.random.default_rng(3)
        for _ in range(budget):
            if time.time() > deadline:
                break
            signs = rng.choice([-1, 1], size=num_edges)
            if _is_connected_lift_fast(graph, signs):
                S = _build_signed_adjacency(graph, signs)
                radius = _spectral_radius(S)
                if radius < best_radius:
                    best_radius = radius
                    best_signs = signs.copy()
        
        if best_signs is not None:
            best_signs, best_radius = _local_search(graph, best_signs, deadline)
    
    if best_signs is None:
        # Last resort: all +1 (check if valid, if not try all -1)
        best_signs = np.ones(num_edges, dtype=int)
        if not _is_connected_lift_fast(graph, best_signs):
            best_signs = np.ones(num_edges, dtype=int)
            best_signs[0] = -1  # Flip one edge to try to make it connected
    
    return best_signs
# EVOLVE-BLOCK-END
