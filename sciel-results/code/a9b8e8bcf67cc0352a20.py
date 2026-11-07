import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    DEADLINE = 85

    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n_vertices = 2 * m

    c1_idx, c2_idx, vert_idx = [], [], []
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            c1_idx.append(i)
        elif u >= m and v >= m:
            c2_idx.append(i)
        else:
            vert_idx.append(i)

    c1_edges = [edges[i] for i in c1_idx]
    c2_edges = [edges[i] for i in c2_idx]
    vert_edges = [edges[i] for i in vert_idx]

    # Map vertex i (0..m-1) to index in vert_idx
    vert_map = [-1] * m
    for j, (u, v) in enumerate(vert_edges):
        if u < m:
            vert_map[u] = j
        else:
            vert_map[v - m] = j

    def build_cycle_matrix(s, edge_list):
        C = np.zeros((m, m), dtype=float)
        for j, (u, v) in enumerate(edge_list):
            C[u, v] = s[j]
            C[v, u] = s[j]
        return C

    n_c = 1 << m
    C1_list = np.zeros((n_c, m, m), dtype=float)
    C2_list = np.zeros((n_c, m, m), dtype=float)
    
    for code in range(n_c):
        s = np.array([1 if (code >> j) & 1 else -1 for j in range(m)], dtype=float)
        C1_list[code] = build_cycle_matrix(s, c1_edges)
        C2_list[code] = build_cycle_matrix(s, c2_edges)

    I_m = np.eye(m)
    candidates = []
    for a in range(n_c):
        C1 = C1_list[a]
        for b in range(n_c):
            C2 = C2_list[b]
            A_proxy = np.block([
                [C1, I_m],
                [I_m, C2]
            ])
            eigvals = np.linalg.eigvalsh(A_proxy)
            rad = float(np.max(np.abs(eigvals)))
            candidates.append((rad, a, b))
            
    candidates.sort(key=lambda x: x[0])
    
    def is_lift_connected(signs):
        n_lift = 4 * m
        adj = [[] for _ in range(n_lift)]
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            u0, u1 = 2*u, 2*u+1
            v0, v1 = 2*v, 2*v+1
            if s == 1:
                adj[u0].append(v0); adj[v0].append(u0)
                adj[u1].append(v1); adj[v1].append(u1)
            else:
                adj[u0].append(v1); adj[v1].append(u0)
                adj[u1].append(v0); adj[v0].append(u1)
        visited = [False] * n_lift
        queue = [0]
        visited[0] = True
        count = 1
        while queue:
            node = queue.pop(0)
            for nei in adj[node]:
                if not visited[nei]:
                    visited[nei] = True
                    count += 1
                    queue.append(nei)
        return count == n_lift

    def compute_radius(signs):
        A = np.zeros((2*m, 2*m), dtype=float)
        for i, (u, v) in enumerate(edges):
            A[u, v] = signs[i]
            A[v, u] = signs[i]
        eigvals = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(eigvals)))

    best_radius = float('inf')
    best_signs = None
    K = 2000
    
    for rad, a, b in candidates[:K]:
        if time.time() > DEADLINE:
            break
        C1 = C1_list[a]
        C2 = C2_list[b]
        
        for v_code in range(n_c):
            if time.time() > DEADLINE:
                break
            s3 = np.array([1 if (v_code >> j) & 1 else -1 for j in range(m)], dtype=int)
            
            signs = np.empty(n_edges, dtype=int)
            for j, idx in enumerate(c1_idx):
                signs[idx] = 1 if (a >> j) & 1 else -1
            for j, idx in enumerate(c2_idx):
                signs[idx] = 1 if (b >> j) & 1 else -1
            for k in range(m):
                signs[vert_idx[vert_map[k]]] = s3[k]
                
            if not is_lift_connected(signs):
                continue
                
            rad_true = compute_radius(signs)
            if rad_true < best_radius:
                best_radius = rad_true
                best_signs = signs.copy()
                
    if best_signs is None:
        # Fallback to random search if no connected lift found in top K
        rng = np.random.default_rng(42)
        for _ in range(10000):
            signs = rng.choice([-1, 1], size=n_edges)
            if is_lift_connected(signs):
                rad_true = compute_radius(signs)
                if rad_true < best_radius:
                    best_radius = rad_true
                    best_signs = signs
            if time.time() > DEADLINE:
                break
                
    return best_signs
