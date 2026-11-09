import numpy as np
import time
import math

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 80  # Safety margin for 90s limit (10s buffer)
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    e_count = len(edges)
    
    # Precompute edge structure
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
    
    def compute_radius(signs):
        """Compute spectral radius of signed adjacency matrix."""
        A = np.zeros((n, n), dtype=float)
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            A[u, v] = s
            A[v, u] = s
        eigvals = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(eigvals)))
    
    def is_valid_lift(signs):
        """Check if the lift is connected and 3-regular."""
        # Build the lift graph
        # Vertices: (v, 0) and (v, 1) for v in 0..n-1
        # Edge (u,v) with sign s: if s=+1, connects (u,0)-(v,0) and (u,1)-(v,1)
        #                       if s=-1, connects (u,0)-(v,1) and (u,1)-(v,0)
        num_lift_vertices = 2 * n
        adj = [[] for _ in range(num_lift_vertices)]
        
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            if s == 1:
                adj[u * 2].append(v * 2)
                adj[v * 2].append(u * 2)
                adj[u * 2 + 1].append(v * 2 + 1)
                adj[v * 2 + 1].append(u * 2 + 1)
            else:
                adj[u * 2].append(v * 2 + 1)
                adj[v * 2 + 1].append(u * 2)
                adj[u * 2 + 1].append(v * 2)
                adj[v * 2].append(u * 2 + 1)
        
        # Check 3-regular
        for v in range(num_lift_vertices):
            if len(adj[v]) != 3:
                return False
        
        # Check connected (BFS from vertex 0)
        visited = [False] * num_lift_vertices
        queue = [0]
        visited[0] = True
        count = 1
        while queue:
            cur = queue.pop(0)
            for nb in adj[cur]:
                if not visited[nb]:
                    visited[nb] = True
                    count += 1
                    queue.append(nb)
        
        return count == num_lift_vertices
    
    def generate_phase_signs(phases):
        """Convert complex unit phases to edge signs."""
        signs = np.zeros(e_count, dtype=int)
        for i, (u, v) in enumerate(edges):
            val = phases[u] * np.conj(phases[v])
            real_part = np.real(val)
            signs[i] = 1 if real_part >= 0 else -1
        return signs
    
    # Phase synchronization approach
    # Parameterize each vertex with a phase in [0, 2*pi)
    # Edge sign = sign(Re(z_u * conj(z_v))) where z_v = exp(i * phase_v)
    
    def optimize_phases(init_phases, max_iters=2000, lr=0.1):
        """Gradient descent on spectral radius using phase parameterization."""
        phases = init_phases.copy()
        best_radius = float('inf')
        best_phases = phases.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute current signs and radius
            signs = generate_phase_signs(np.exp(1j * phases))
            try:
                r = compute_radius(signs)
            except Exception:
                break
            
            if r < best_radius:
                best_radius = r
                best_phases = phases.copy()
            
            # Compute gradient via finite differences
            grad = np.zeros(n)
            eps = 0.01
            for v in range(n):
                phases_plus = phases.copy()
                phases_plus[v] += eps
                signs_plus = generate_phase_signs(np.exp(1j * phases_plus))
                try:
                    r_plus = compute_radius(signs_plus)
                except Exception:
                    r_plus = r + eps  # fallback
                grad[v] = (r_plus - r) / eps
            
            # Gradient descent step
            phases -= lr * grad
            # Wrap phases to [0, 2*pi)
            phases = phases % (2 * math.pi)
            
            # Adaptive learning rate
            if it % 100 == 0:
                lr *= 0.95
        
        return best_phases, best_radius
    
    # Generate diverse initial phase configurations
    initial_phases_list = []
    
    # 1. All zeros (all +1 signs)
    initial_phases_list.append(np.zeros(n))
    
    # 2. Alternating phases
    phases_alt = np.zeros(n)
    for v in range(n):
        phases_alt[v] = math.pi if v % 2 == 1 else 0
    initial_phases_list.append(phases_alt)
    
    # 3. Top/bottom split
    phases_split = np.zeros(n)
    for v in range(m, 2 * m):
        phases_split[v] = math.pi
    initial_phases_list.append(phases_split)
    
    # 4. Random phases
    rng = np.random.default_rng(42)
    for _ in range(5):
        initial_phases_list.append(rng.uniform(0, 2 * math.pi, n))
    
    # 5. Structured phases based on cycle position
    phases_cycle = np.zeros(n)
    for v in range(m):
        phases_cycle[v] = 2 * math.pi * v / m
        phases_cycle[v + m] = 2 * math.pi * (v + 1) / m
    initial_phases_list.append(phases_cycle)
    
    # Run phase optimization from each initialization
    best_signs = np.ones(e_count, dtype=int)
    best_radius = float('inf')
    
    for init_phases in initial_phases_list:
        if time.time() > deadline:
            break
        
        opt_phases, opt_radius = optimize_phases(init_phases, max_iters=500, lr=0.1)
        
        if opt_radius < best_radius:
            best_radius = opt_radius
            best_signs = generate_phase_signs(np.exp(1j * opt_phases))
    
    # Validate and refine with discrete local search
    # First, ensure we have a valid lift
    if not is_valid_lift(best_signs):
        # Try to find a valid lift by flipping edges
        for attempt in range(100):
            if time.time() > deadline:
                break
            # Randomly flip a few edges
            flip_indices = rng.choice(e_count, size=min(3, e_count), replace=False)
            new_signs = best_signs.copy()
            for idx in flip_indices:
                new_signs[idx] = -new_signs[idx]
            if is_valid_lift(new_signs):
                r = compute_radius(new_signs)
                if r < best_radius:
                    best_radius = r
                    best_signs = new_signs
                break
    
    # Discrete local search to polish
    current_signs = best_signs.copy()
    current_radius = best_radius
    no_improve_count = 0
    
    while time.time() < deadline and no_improve_count < 30:
        improved = False
        
        for edge_idx in range(e_count):
            if time.time() > deadline:
                break
            
            new_signs = current_signs.copy()
            new_signs[edge_idx] = -new_signs[edge_idx]
            
            if not is_valid_lift(new_signs):
                continue
                
            try:
                r = compute_radius(new_signs)
                if r < current_radius - 1e-10:
                    current_signs = new_signs
                    current_radius = r
                    best_signs = new_signs.copy()
                    best_radius = r
                    improved = True
                    no_improve_count = 0
                    break
            except Exception:
                pass
        
        if not improved:
            no_improve_count += 1
            # Try 2-flip moves
            if no_improve_count % 3 == 0:
                for i in range(e_count):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, e_count):
                        if time.time() > deadline:
                            break
                        new_signs = current_signs.copy()
                        new_signs[i] = -new_signs[i]
                        new_signs[j] = -new_signs[j]
                        
                        if not is_valid_lift(new_signs):
                            continue
                            
                        try:
                            r = compute_radius(new_signs)
                            if r < current_radius - 1e-10:
                                current_signs = new_signs
                                current_radius = r
                                best_signs = new_signs.copy()
                                best_radius = r
                                improved = True
                                no_improve_count = 0
                                break
                        except Exception:
                            pass
                    if improved:
                        break
    
    return best_signs
