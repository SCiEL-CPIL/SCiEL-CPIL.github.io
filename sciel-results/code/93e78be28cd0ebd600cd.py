import numpy as np
import time
from collections import deque

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit

    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)

    # Identify edge types: cycle 1, cycle 2, matching
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

    # Helper to get cycle edge order
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

    # Fast connectivity check using union-find
    def is_connected(signs):
        parent = list(range(2 * n))
        rank = [0] * (2 * n)
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
        
        for i in range(num_edges):
            u = edge_u[i]
            v = edge_v[i]
            s = signs[i]
            if s == 1:
                union(u, v)
                union(u + n, v + n)
            else:
                union(u, v + n)
                union(u + n, v)
        
        root = find(0)
        for i in range(1, 2 * n):
            if find(i) != root:
                return False
        return True

    # Fast spectral radius using signed adjacency matrix
    def spectral_radius(signs):
        A = np.zeros((n, n), dtype=float)
        A[edge_u, edge_v] = signs
        A[edge_v, edge_u] = signs
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))

    def evaluate(signs):
        if not is_connected(signs):
            return float('inf')
        return spectral_radius(signs)

    best_radius = float('inf')
    best_signs = None

    def try_signs(signs):
        nonlocal best_radius, best_signs
        r = evaluate(signs)
        if r < best_radius:
            best_radius = r
            best_signs = signs.copy()

    # Generate diverse initial candidates
    candidates = []
    
    # Pattern 1: Alt C1, Alt C2, -1 M
    s1 = np.ones(num_edges, dtype=int)
    for pos, idx in c1_ordered: s1[idx] = 1 if pos % 2 == 0 else -1
    for pos, idx in c2_ordered: s2_idx = idx; s1[idx] = 1 if pos % 2 == 0 else -1
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
    
    # Pattern 6: All -1 (works for odd m)
    if m % 2 == 1:
        candidates.append(np.full(num_edges, -1, dtype=int))
    
    # Pattern 7: Random starts
    rng = np.random.default_rng(3)
    for _ in range(15):
        s_rand = rng.choice([-1, 1], size=num_edges).astype(int)
        candidates.append(s_rand)

    # Evaluate initial candidates
    for s in candidates:
        try_signs(s)

    if best_signs is None:
        return np.ones(num_edges, dtype=int)

    # Simulated annealing from best initial
    current_signs = best_signs.copy()
    current_radius = best_radius

    T = 0.3
    cooling = 0.998
    max_steps = 5000

    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_signs = current_signs.copy()
        
        # Pick a random edge to flip
        idx = rng.integers(0, num_edges)
        current_signs[idx] *= -1
        
        r = evaluate(current_signs)
        
        if r < float('inf'):
            delta = r - current_radius
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_radius = r
            else:
                current_signs[idx] *= -1
        else:
            current_signs[idx] *= -1
        
        T *= cooling

    # Final hill climbing polish
    current_signs = best_signs.copy()
    current_radius = best_radius
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(num_edges):
            if time.time() > deadline - 5:
                break
            current_signs[i] *= -1
            r = evaluate(current_signs)
            if r < current_radius - 1e-10:
                current_radius = r
                best_radius = r
                best_signs = current_signs.copy()
                improved = True
            else:
                current_signs[i] *= -1

    return best_signs
