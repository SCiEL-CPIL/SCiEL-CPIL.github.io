import numpy as np
import time
from collections import deque

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 90s limit (40s buffer)
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    e_count = len(edges)
    
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
    
    # Precompute edge endpoints as arrays for speed
    edge_u = np.array([e[0] for e in edges], dtype=int)
    edge_v = np.array([e[1] for e in edges], dtype=int)
    
    # Precompute lift adjacency contributions
    lift_adj_plus = [None] * e_count
    lift_adj_minus = [None] * e_count
    
    for i in range(e_count):
        u = edge_u[i]
        v = edge_v[i]
        u0, u1 = u * 2, u * 2 + 1
        v0, v1 = v * 2, v * 2 + 1
        
        # Sign +1: (u,0)-(v,0) and (u,1)-(v,1)
        lift_adj_plus[i] = [(u0, v0), (v0, u0), (u1, v1), (v1, u1)]
        
        # Sign -1: (u,0)-(v,1) and (u,1)-(v,0)
        lift_adj_minus[i] = [(u0, v1), (v1, u0), (u1, v0), (v0, u1)]
    
    def evaluate_signs(signs):
        # signs is a numpy array of 1s and -1s
        # 1. Check connectivity of the lift
        lift_n = 2 * n
        adj_lift = [[] for _ in range(lift_n)]
        for i in range(e_count):
            if signs[i] == 1:
                for u_l, v_l in lift_adj_plus[i]:
                    adj_lift[u_l].append(v_l)
            else:
                for u_l, v_l in lift_adj_minus[i]:
                    adj_lift[u_l].append(v_l)
        
        visited = [False] * lift_n
        queue = deque([0])
        visited[0] = True
        while queue:
            u = queue.popleft()
            for v in adj_lift[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        
        if not all(visited):
            return float('inf')
            
        # 2. Compute spectral radius
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
    
    # Generate initial candidates using structured patterns
    candidates = []
    
    # Pattern 1: Alt C1, Alt C2, -1 M
    s1 = np.ones(e_count, dtype=int)
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
    s4 = np.ones(e_count, dtype=int)
    for pos, idx in enumerate(m_ordered): s4[idx] = 1 if pos % 2 == 0 else -1
    candidates.append(s4)
    
    # Pattern 5: Alt C1, Alt C2, Alt M
    s5 = s1.copy()
    for pos, idx in enumerate(m_ordered): s5[idx] = 1 if pos % 2 == 0 else -1
    candidates.append(s5)
    
    # Evaluate initial candidates
    for s in candidates:
        try_signs(s)
    
    if best_signs is None:
        # Fallback: all +1
        best_signs = np.ones(e_count, dtype=int)
        best_radius = evaluate_signs(best_signs)
    
    # Phase-based optimization with smooth surrogate
    def phase_to_signs(theta):
        diffs = theta[edge_u] - theta[edge_v]
        cos_vals = np.cos(diffs)
        return np.where(cos_vals >= 0, 1, -1).astype(int)
    
    def spectral_radius_from_phases(theta):
        signs = phase_to_signs(theta)
        A = np.zeros((n, n), dtype=float)
        A[edge_u, edge_v] = signs
        A[edge_v, edge_u] = signs
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def soft_spectral_radius(theta, temperature=0.1):
        signs = phase_to_signs(theta)
        A = np.zeros((n, n), dtype=float)
        A[edge_u, edge_v] = signs
        A[edge_v, edge_u] = signs
        eigvals = np.linalg.eigvalsh(A)
        abs_eigs = np.abs(eigvals)
        max_eig = np.max(abs_eigs)
        log_sum_exp = max_eig + np.log(np.sum(np.exp((abs_eigs - max_eig) / temperature))) * temperature
        return log_sum_exp
    
    def soft_spectral_radius_grad(theta, temperature=0.1):
        eps = 1e-4
        grad = np.zeros(n)
        base_val = soft_spectral_radius(theta, temperature)
        for j in range(n):
            theta_plus = theta.copy()
            theta_plus[j] += eps
            val_plus = soft_spectral_radius(theta_plus, temperature)
            grad[j] = (val_plus - base_val) / eps
        return grad
    
    # Multiple random restarts for phase optimization
    rng = np.random.default_rng(42)
    num_restarts = 15
    steps_per_restart = 40
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Initialize phases
        if restart == 0:
            theta = np.zeros(n)
        else:
            theta = rng.uniform(0, 2 * np.pi, size=n)
        
        current_radius = spectral_radius_from_phases(theta)
        
        # Gradient descent with adaptive step size
        step_size = 0.5
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
            
            grad = soft_spectral_radius_grad(theta, temperature=0.1)
            
            # Update phases
            theta_new = theta - step_size * grad
            theta_new = np.mod(theta_new, 2 * np.pi)
            
            new_radius = spectral_radius_from_phases(theta_new)
            
            if new_radius < current_radius:
                theta = theta_new
                current_radius = new_radius
                step_size = min(step_size * 1.05, 2.0)
            else:
                step_size *= 0.9
            
            if step_size < 0.01:
                break
        
        # Get signs from optimized phases
        signs = phase_to_signs(theta)
        r = evaluate_signs(signs)
        if r < best_radius:
            best_radius = r
            best_signs = signs.copy()
    
    # Local search polish: single flips
    if time.time() < deadline:
        improved = True
        max_polish_iters = 50
        for _ in range(max_polish_iters):
            if time.time() > deadline:
                break
            improved = False
            for i in range(e_count):
                if time.time() > deadline:
                    break
                old_sign = best_signs[i]
                best_signs[i] = -old_sign
                new_radius = evaluate_signs(best_signs)
                if new_radius < best_radius - 1e-10:
                    best_radius = new_radius
                    improved = True
                else:
                    best_signs[i] = old_sign
            if not improved:
                break
    
    # Pairwise flip search
    if time.time() < deadline:
        current_signs = best_signs.copy()
        current_radius = best_radius
        max_pair_passes = 3
        for _ in range(max_pair_passes):
            if time.time() > deadline:
                break
            improved = False
            for i in range(e_count):
                if time.time() > deadline:
                    break
                for j in range(i + 1, e_count):
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
                if time.time() > deadline:
                    break
            if not improved:
                break
    
    return best_signs
