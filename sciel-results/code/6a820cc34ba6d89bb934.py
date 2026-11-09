import numpy as np
import time
from collections import deque

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit

    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)

    # Identify edge types
    c1_edges = []
    c2_edges = []
    m_edges = []

    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            c1_edges.append(i)
        elif u >= m and v >= m:
            c2_edges.append(i)
        else:
            m_edges.append(i)

    # Precompute edge endpoints
    edge_u = np.array([u for u, v in edges], dtype=int)
    edge_v = np.array([v for u, v in edges], dtype=int)

    # Precompute lift adjacency contributions for fast connectivity check
    # For connectivity, we can use a faster check based on the structure.
    # A 2-lift of a connected graph is connected iff the product of signs 
    # around every cycle is not all -1? No, that's for 2-cover.
    # Actually, the lift is disconnected iff there is a subset of vertices S 
    # such that the number of edges crossing the cut (S, V\S) with sign -1 
    # is even? No.
    # The lift is disconnected iff the signed graph is "antibalanced" in a specific way?
    # Simpler: Just do BFS on the lift graph. It's small (2n nodes, 6m edges).
    # n=18 max, so 36 nodes, 54 edges. BFS is very fast.

    def check_connectivity(signs):
        lift_n = 2 * n
        visited = [False] * lift_n
        queue = deque([0])
        visited[0] = True
        
        # Build adjacency on the fly or precompute?
        # Precomputing full adjacency for every sign change is expensive.
        # Let's precompute the "neighbor" structure.
        # For each vertex v in base, neighbors are fixed.
        # In lift, (v, 0) is connected to (u, 0) if sign(e)=1 and (u,1) if sign(e)=-1.
        
        # We can just iterate edges.
        # This is O(E) per BFS step if we do it naively, but we can just build the list.
        # Since n is small, building the list is fine.
        
        adj_lift = [[] for _ in range(lift_n)]
        for i in range(num_edges):
            u = edge_u[i]
            v = edge_v[i]
            s = signs[i]
            if s == 1:
                adj_lift[u*2].append(v*2)
                adj_lift[v*2].append(u*2)
                adj_lift[u*2+1].append(v*2+1)
                adj_lift[v*2+1].append(u*2+1)
            else:
                adj_lift[u*2].append(v*2+1)
                adj_lift[v*2+1].append(u*2)
                adj_lift[u*2+1].append(v*2)
                adj_lift[v*2].append(u*2+1)
        
        while queue:
            u = queue.popleft()
            for v in adj_lift[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        
        return all(visited)

    def evaluate_signs(signs):
        if not check_connectivity(signs):
            return float('inf')
            
        A = np.zeros((n, n), dtype=float)
        A[edge_u, edge_v] = signs
        A[edge_v, edge_u] = signs
        
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))

    best_radius = float('inf')
    best_signs = None

    def try_signs(signs):
        nonlocal best_radius, best_signs
        r = evaluate_signs(signs)
        if r < best_radius:
            best_radius = r
            best_signs = signs.copy()

    def get_cycle_edges_indices(edge_indices, offset=0):
        pos_map = {}
        for idx in edge_indices:
            u, v = edges[idx]
            u = u - offset
            v = v - offset
            if (u + 1) % m == v:
                pos = u
            elif (v + 1) % m == u:
                pos = v
            else:
                continue
            pos_map[pos] = idx
        return sorted(pos_map.items())

    c1_ordered = get_cycle_edges_indices(c1_edges, 0)
    c2_ordered = get_cycle_edges_indices(c2_edges, m)
    m_ordered = sorted(m_edges, key=lambda i: edges[i][0] if edges[i][0] < m else edges[i][1])

    # Generate initial candidates
    candidates = []
    
    # Pattern 1: Alt C1, Alt C2, -1 M
    s1 = np.ones(num_edges, dtype=int)
    for pos, idx in c1_ordered: s1[idx] = 1 if pos % 2 == 0 else -1
    for pos, idx in c2_ordered: s1[idx] = 1 if pos % 2 == 0 else -1
    for idx in m_edges: s1[idx] = -1
    candidates.append(s1)
    
    # Pattern 2: Alt C1, Anti-Alt C2, -1 M
    s2 = s1.copy()
    for pos, idx in c2_ordered: s2[idx] = -1 if pos % 2 == 0 else 1
    candidates.append(s2)
    
    # Pattern 3: Alt C1, Alt C2, +1 M
    s3 = s1.copy()
    for idx in m_edges: s3[idx] = 1
    candidates.append(s3)
    
    # Pattern 4: All +1 C1, All +1 C2, Alt M
    s4 = np.ones(num_edges, dtype=int)
    for pos, idx in enumerate(m_ordered): s4[idx] = 1 if pos % 2 == 0 else -1
    candidates.append(s4)
    
    # Pattern 5: Alt C1, Alt C2, Alt M
    s5 = s1.copy()
    for pos, idx in enumerate(m_ordered): s5[idx] = 1 if pos % 2 == 0 else -1
    candidates.append(s5)
    
    for s in candidates:
        try_signs(s)

    # Phase 1: Single Flip Hill Climb
    current_signs = best_signs.copy()
    current_radius = best_radius

    max_passes = 20
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        for i in range(num_edges):
            if time.time() > deadline:
                break
            current_signs[i] *= -1
            r = evaluate_signs(current_signs)
            if r < current_radius - 1e-9:
                current_radius = r
                best_radius = r
                best_signs = current_signs.copy()
                improved = True
            else:
                current_signs[i] *= -1
        if not improved:
            break

    # Phase 2: Pairwise Flip Search
    current_signs = best_signs.copy()
    current_radius = best_radius
    
    max_pair_passes = 5
    for _ in range(max_pair_passes):
        if time.time() > deadline:
            break
        improved = False
        for i in range(num_edges):
            if time.time() > deadline:
                break
            for j in range(i + 1, num_edges):
                if time.time() > deadline:
                    break
                current_signs[i] *= -1
                current_signs[j] *= -1
                r = evaluate_signs(current_signs)
                if r < current_radius - 1e-9:
                    current_radius = r
                    best_radius = r
                    best_signs = current_signs.copy()
                    improved = True
                    current_signs = best_signs.copy()
                    break
                else:
                    current_signs[i] *= -1
                    current_signs[j] *= -1
            if improved:
                break
        if not improved:
            break

    # Phase 3: Final Single Flip Polish
    current_signs = best_signs.copy()
    current_radius = best_radius
    for _ in range(10):
        if time.time() > deadline:
            break
        improved = False
        for i in range(num_edges):
            if time.time() > deadline:
                break
            current_signs[i] *= -1
            r = evaluate_signs(current_signs)
            if r < current_radius - 1e-9:
                current_radius = r
                best_radius = r
                best_signs = current_signs.copy()
                improved = True
            else:
                current_signs[i] *= -1
        if not improved:
            break

    # Phase 1: Single Flip Hill Climb
    current_signs = best_signs.copy()
    current_radius = best_radius

    max_passes = 20
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        for i in range(num_edges):
            if time.time() > deadline:
                break
            current_signs[i] *= -1
            r = evaluate_signs(current_signs)
            if r < current_radius - 1e-9:
                current_radius = r
                best_radius = r
                best_signs = current_signs.copy()
                improved = True
            else:
                current_signs[i] *= -1
        if not improved:
            break

    # Phase 2: Pairwise Flip Search
    current_signs = best_signs.copy()
    current_radius = best_radius
    
    max_pair_passes = 5
    for _ in range(max_pair_passes):
        if time.time() > deadline:
            break
        improved = False
        for i in range(num_edges):
            if time.time() > deadline:
                break
            for j in range(i + 1, num_edges):
                if time.time() > deadline:
                    break
                current_signs[i] *= -1
                current_signs[j] *= -1
                r = evaluate_signs(current_signs)
                if r < current_radius - 1e-9:
                    current_radius = r
                    best_radius = r
                    best_signs = current_signs.copy()
                    improved = True
                    current_signs = best_signs.copy()
                    break
                else:
                    current_signs[i] *= -1
                    current_signs[j] *= -1
            if improved:
                break
        if not improved:
            break

    # Phase 3: Final Single Flip Polish
    current_signs = best_signs.copy()
    current_radius = best_radius
    for _ in range(10):
        if time.time() > deadline:
            break
        improved = False
        for i in range(num_edges):
            if time.time() > deadline:
                break
            current_signs[i] *= -1
            r = evaluate_signs(current_signs)
            if r < current_radius - 1e-9:
                current_radius = r
                best_radius = r
                best_signs = current_signs.copy()
                improved = True
            else:
                current_signs[i] *= -1
        if not improved:
            break

    return best_signs
