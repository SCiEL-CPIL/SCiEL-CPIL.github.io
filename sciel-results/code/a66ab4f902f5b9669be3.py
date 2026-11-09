import numpy as np
import time
from collections import deque
from scipy.optimize import minimize

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit

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

    # Precompute lift adjacency for each sign
    lift_adj_plus = [None] * num_edges
    lift_adj_minus = [None] * num_edges

    for i in range(num_edges):
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
        lift_n = 2 * n
        adj_lift = [[] for _ in range(lift_n)]
        for i in range(num_edges):
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

        # Compute spectral radius
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

    # Generate diverse initial candidates
    candidates = []

    # Pattern 1: Alt C1, Alt C2, -1 M
    s = np.ones(num_edges, dtype=int)
    for pos, idx in c1_ordered: s[idx] = 1 if pos % 2 == 0 else -1
    for pos, idx in c2_ordered: s[idx] = 1 if pos % 2 == 0 else -1
    for idx in m_edges: s[idx] = -1
    candidates.append(s.copy())

    # Pattern 2: Alt C1, Anti-Alt C2, -1 M
    s = np.ones(num_edges, dtype=int)
    for pos, idx in c1_ordered: s[idx] = 1 if pos % 2 == 0 else -1
    for pos, idx in c2_ordered: s[idx] = -1 if pos % 2 == 0 else 1
    for idx in m_edges: s[idx] = -1
    candidates.append(s.copy())

    # Pattern 3: Anti-Alt C1, Alt C2, -1 M
    s = np.ones(num_edges, dtype=int)
    for pos, idx in c1_ordered: s[idx] = -1 if pos % 2 == 0 else 1
    for pos, idx in c2_ordered: s[idx] = 1 if pos % 2 == 0 else -1
    for idx in m_edges: s[idx] = -1
    candidates.append(s.copy())

    # Pattern 4: Alt C1, Alt C2, +1 M
    s = np.ones(num_edges, dtype=int)
    for pos, idx in c1_ordered: s[idx] = 1 if pos % 2 == 0 else -1
    for pos, idx in c2_ordered: s[idx] = 1 if pos % 2 == 0 else -1
    for idx in m_edges: s[idx] = 1
    candidates.append(s.copy())

    # Pattern 5: All +1 C1, All +1 C2, Alt M
    s = np.ones(num_edges, dtype=int)
    for pos, idx in enumerate(m_ordered): s[idx] = 1 if pos % 2 == 0 else -1
    candidates.append(s.copy())

    # Pattern 6: Alt C1, Alt C2, Alt M
    s = np.ones(num_edges, dtype=int)
    for pos, idx in c1_ordered: s[idx] = 1 if pos % 2 == 0 else -1
    for pos, idx in c2_ordered: s[idx] = 1 if pos % 2 == 0 else -1
    for pos, idx in enumerate(m_ordered): s[idx] = 1 if pos % 2 == 0 else -1
    candidates.append(s.copy())

    # Pattern 7: All -1
    s = -np.ones(num_edges, dtype=int)
    candidates.append(s.copy())

    # Pattern 8: All +1
    s = np.ones(num_edges, dtype=int)
    candidates.append(s.copy())

    # Phase synchronization approach: parameterize vertices with complex phases
    # z_v = exp(i * theta_v), edge sign = sign(Re(z_u * conj(z_v)))
    def phases_to_signs(theta):
        """Convert vertex phases to edge signs"""
        z = np.exp(1j * theta)  # complex unit vectors for each vertex
        signs = np.zeros(num_edges, dtype=int)
        for i in range(num_edges):
            u = edge_u[i]
            v = edge_v[i]
            val = np.real(z[u] * np.conj(z[v]))
            signs[i] = 1 if val >= 0 else -1
        return signs

    def spectral_radius_from_phases(theta):
        """Compute spectral radius from vertex phases using smooth surrogate"""
        signs = phases_to_signs(theta)
        # Check validity
        lift_n = 2 * n
        adj_lift = [[] for _ in range(lift_n)]
        for i in range(num_edges):
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
        
        A = np.zeros((n, n), dtype=float)
        A[edge_u, edge_v] = signs
        A[edge_v, edge_u] = signs
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def spectral_radius_smooth(theta):
        """Smooth surrogate: use soft sign values for spectral radius optimization"""
        # Compute soft signs: sigmoid of Re(z_u * conj(z_v))
        z = np.exp(1j * theta)
        soft_signs = np.zeros(num_edges, dtype=float)
        for i in range(num_edges):
            u = edge_u[i]
            v = edge_v[i]
            val = np.real(z[u] * np.conj(z[v]))
            # Use tanh for smooth approximation of sign
            soft_signs[i] = np.tanh(5.0 * val)
        
        A = np.zeros((n, n), dtype=float)
        A[edge_u, edge_v] = soft_signs
        A[edge_v, edge_u] = soft_signs
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))

    # Evaluate initial candidates
    for s in candidates:
        try_signs(s)

    # Phase 0: Continuous optimization via phase synchronization
    if time.time() < deadline - 10:
        # Try multiple random initializations of phases
        rng = np.random.default_rng(42)
        best_phase_radius = float('inf')
        best_phase_theta = None
        
        for init_iter in range(8):
            if time.time() > deadline - 10:
                break
            
            # Random initial phases
            theta0 = rng.uniform(-np.pi, np.pi, n)
            
            # Optimize using smooth surrogate first, then refine with discrete
            result = minimize(
                spectral_radius_smooth,
                theta0,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            # Convert to discrete signs and evaluate
            if np.isfinite(result.fun):
                discrete_signs = phases_to_signs(result.x)
                discrete_radius = evaluate_signs(discrete_signs)
                if discrete_radius < best_phase_radius and np.isfinite(discrete_radius):
                    best_phase_radius = discrete_radius
                    best_phase_theta = result.x.copy()
        
        if best_phase_theta is not None:
            phase_signs = phases_to_signs(best_phase_theta)
            try_signs(phase_signs)

    if best_signs is None:
        # Fallback: all +1
        return np.ones(num_edges, dtype=int)

    # Phase 1: Single Flip Hill Climb
    current_signs = best_signs.copy()
    current_radius = best_radius
    max_passes = 30
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

    # Phase 2: Pairwise Flip Search (2-opt style)
    current_signs = best_signs.copy()
    current_radius = best_radius
    max_pair_passes = 10
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
                    # Restart the pairwise search from the new state
                    current_signs = best_signs.copy()
                    current_radius = best_radius
                    break
                else:
                    current_signs[i] *= -1
                    current_signs[j] *= -1
            if improved:
                break
        if not improved:
            break

    # Phase 3: Triple flip search if time permits
    if time.time() < deadline - 5:
        current_signs = best_signs.copy()
        current_radius = best_radius
        for _ in range(5):
            if time.time() > deadline:
                break
            improved = False
            for i in range(num_edges):
                if time.time() > deadline:
                    break
                for j in range(i + 1, num_edges):
                    if time.time() > deadline:
                        break
                    for k in range(j + 1, num_edges):
                        if time.time() > deadline:
                            break
                        current_signs[i] *= -1
                        current_signs[j] *= -1
                        current_signs[k] *= -1
                        r = evaluate_signs(current_signs)
                        if r < current_radius - 1e-9:
                            current_radius = r
                            best_radius = r
                            best_signs = current_signs.copy()
                            improved = True
                            current_signs = best_signs.copy()
                            current_radius = best_radius
                            break
                        else:
                            current_signs[i] *= -1
                            current_signs[j] *= -1
                            current_signs[k] *= -1
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break

    # Phase 4: Final continuous optimization polish from best discrete solution
    if time.time() < deadline - 3:
        # Try to find phases consistent with current best signs
        best_theta = None
        best_theta_radius = float('inf')
        
        # Try several random phase initializations
        rng = np.random.default_rng(123)
        for init_iter in range(6):
            if time.time() > deadline - 3:
                break
            
            # Start with random phases
            theta0 = rng.uniform(-np.pi, np.pi, n)
            
            # Use smooth surrogate for optimization
            result = minimize(
                spectral_radius_smooth,
                theta0,
                method='L-BFGS-B',
                options={'maxiter': 80, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            # Evaluate discrete version
            if np.isfinite(result.fun):
                discrete_signs = phases_to_signs(result.x)
                discrete_radius = evaluate_signs(discrete_signs)
                if discrete_radius < best_theta_radius and np.isfinite(discrete_radius):
                    best_theta_radius = discrete_radius
                    best_theta = result.x.copy()
        
        if best_theta is not None:
            phase_signs = phases_to_signs(best_theta)
            try_signs(phase_signs)

    return best_signs
