import time
import numpy as np
from collections import deque

def sign_edges(graph, budget):
    """
    Returns an array of signs (-1 or +1) for each edge in the graph.
    
    Strategy:
    1. Exhaustive search for small m (m <= 6, i.e., edges <= 18) to guarantee global optimum.
    2. For larger m (7, 9):
       a. Continuous Relaxation: Map signs to phases theta in [0, pi].
          Objective: Minimize spectral radius of A(theta) where A_ij = cos(theta_k) for edge k=(i,j).
          This creates a smooth landscape.
       b. Gradient Descent: Use power iteration / eigenvector gradients to minimize the max eigenvalue magnitude.
       c. Projection & Polish: Project continuous solution to discrete signs {-1, 1}, then refine with discrete hill-climbing.
    3. Efficient connectivity check using BFS on the signed graph (coboundary check).
    """
    
    # Safety margin: The evaluator kills at 90s. We aim to finish well before.
    deadline = time.time() + 80.0
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2
    
    # Precompute edge endpoints for fast matrix construction
    u_arr = np.array([e[0] for e in edges])
    v_arr = np.array([e[1] for e in edges])
    
    # Precompute adjacency list for connectivity check
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))

    def is_lift_connected(signs):
        """
        Returns True if the 2-lift is connected.
        Equivalent to the signing NOT being a coboundary.
        """
        f = [-1] * n
        f[0] = 0
        queue = deque([0])
        
        while queue:
            u = queue.popleft()
            for v, idx in adj[u]:
                s = signs[idx]
                flip = 1 if s == -1 else 0
                required_f_v = f[u] ^ flip
                
                if f[v] == -1:
                    f[v] = required_f_v
                    queue.append(v)
                else:
                    if f[v] != required_f_v:
                        # Contradiction found: No coboundary exists.
                        # Therefore, the lift IS connected.
                        return True
        
        # No contradiction found: A coboundary exists.
        # Therefore, the lift is DISCONNECTED.
        return False

    def get_radius(signs):
        """
        Computes the spectral radius of the signed adjacency matrix.
        """
        S = np.zeros((n, n), dtype=np.float64)
        # Vectorized assignment
        S[u_arr, v_arr] = signs.astype(float)
        S[v_arr, u_arr] = signs.astype(float)
        eigvals = np.linalg.eigvalsh(S)
        return np.max(np.abs(eigvals))

    def discrete_hill_climb(signs, time_limit):
        """
        Standard discrete hill climbing to polish a candidate.
        """
        current_signs = signs.copy()
        if not is_lift_connected(current_signs):
            # Try to fix connectivity
            for _ in range(50):
                idx = np.random.randint(num_edges)
                current_signs[idx] *= -1
                if is_lift_connected(current_signs):
                    break
            if not is_lift_connected(current_signs):
                return current_signs, np.inf

        current_radius = get_radius(current_signs)
        improved = True
        start_time = time.time()
        
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            for i in range(num_edges):
                if time.time() > deadline:
                    return current_signs, current_radius
                
                old_sign = current_signs[i]
                current_signs[i] = -old_sign
                
                if not is_lift_connected(current_signs):
                    current_signs[i] = old_sign
                    continue
                
                new_radius = get_radius(current_signs)
                
                if new_radius < current_radius - 1e-9:
                    current_radius = new_radius
                    improved = True
                else:
                    current_signs[i] = old_sign
                    
        return current_signs, current_radius

    best_signs = None
    best_radius = np.inf
    
    # --- Strategy 1: Exhaustive Search for small m ---
    if num_edges <= 18:
        max_combo = 1 << num_edges
        # Optimization: Pre-allocate arrays
        signs_buf = np.empty(num_edges, dtype=np.int8)
        
        for k in range(max_combo):
            if time.time() > deadline - 2.0:
                break
            
            # Extract signs
            # Using bit manipulation is fast
            # Map bit 0 -> -1, bit 1 -> +1
            # Actually, let's just use the standard trick
            signs = np.where((k >> np.arange(num_edges)) & 1, 1, -1).astype(np.int8)
            
            if not is_lift_connected(signs):
                continue
                
            radius = get_radius(signs)
            if radius < best_radius:
                best_radius = radius
                best_signs = signs.copy()
        
        if best_signs is not None:
            return best_signs

    # --- Strategy 2: Continuous Phase Optimization for larger m ---
    
    # We optimize phases theta. s_i = cos(theta_i).
    # Initial phases: random or structured.
    rng = np.random.default_rng(42)
    
    def continuous_optimize(init_thetas, time_limit):
        """
        Minimizes the spectral radius of the matrix with entries cos(theta_i).
        Returns the optimal thetas.
        """
        thetas = init_thetas.copy()
        # We want to minimize lambda_max(A(theta)).
        # Gradient descent step.
        
        # To speed up convergence, we can use a simple iterative refinement.
        # However, computing full eigenvectors every step is expensive.
        # For n=14 or 18, it's okay.
        
        current_val = np.inf
        # We track the top eigenvalue magnitude.
        
        # Simple Gradient Descent
        # grad_theta_i = -sin(theta_i) * (v_u * v_v) where v is the eigenvector of max |lambda|.
        # We want to move theta in direction -grad.
        
        lr = 0.1
        # Momentum? Maybe just simple GD is enough for this small scale.
        
        for step in range(500):
            if time.time() > deadline - 5.0:
                break
                
            # Compute A
            S = np.zeros((n, n), dtype=np.float64)
            vals = np.cos(thetas)
            S[u_arr, v_arr] = vals
            S[v_arr, u_arr] = vals
            
            # Get top eigenvalue and eigenvector
            # eigvalsh is stable for symmetric
            eigvals, eigvecs = np.linalg.eigh(S)
            
            # Find index of max absolute eigenvalue
            max_idx = np.argmax(np.abs(eigvals))
            lam = eigvals[max_idx]
            v = eigvecs[:, max_idx]
            
            current_val = np.abs(lam)
            
            # Compute gradient
            # d/dtheta_i (v^T S v) = v^T (dS/dtheta_i) v
            # dS/dtheta_i has -sin(theta_i) at (u_i, v_i) and (v_i, u_i).
            # So grad_i = -sin(theta_i) * 2 * v[u_i] * v[v_i]
            
            grads = -np.sin(thetas) * 2.0 * v[u_arr] * v[v_arr]
            
            # Update
            thetas -= lr * grads
            
            # Project thetas to [0, pi] to keep cos in [-1, 1]
            # Actually, cos is periodic. We can just let it flow, but 
            # restricting to [0, pi] ensures we stay in the principal branch.
            thetas = np.clip(thetas, 0, np.pi)
            
            # Check convergence
            if np.max(np.abs(grads)) < 1e-5:
                break
                
        return thetas

    # Generate initializations
    initializations = []
    
    # 1. All 0 (All +1)
    initializations.append(np.zeros(num_edges))
    # 2. All pi (All -1)
    initializations.append(np.full(num_edges, np.pi))
    
    # 3. Structured: Alternating on cycles
    # Prism: 0..m-1 (C1), m..2m-1 (C2), 2m..3m-1 (Match)
    # Try to balance signs on cycles
    init_struct1 = np.zeros(num_edges)
    # C1: alternate 0, pi/2? No, we want to end up near 0 or pi.
    # Let's just use random starts and some heuristic starts.
    
    # Random starts
    for _ in range(20):
        init_rand = rng.uniform(0, np.pi, size=num_edges)
        initializations.append(init_rand)
    
    # Heuristic: Try to make cycles have product -1 (frustrated) to reduce radius?
    # Actually, for bipartite-like structures, balanced is good.
    # Prism graph is not bipartite for odd m.
    
    # Run continuous optimization for each init
    time_per_init = (deadline - time.time() - 5.0) / len(initializations)
    if time_per_init < 1.0:
        time_per_init = 1.0
        
    for init in initializations:
        if time.time() > deadline - 5.0:
            break
            
        opt_thetas = continuous_optimize(init, time_per_init)
        
        # Project to discrete signs
        # s_i = 1 if cos(theta_i) > 0 else -1
        # This is equivalent to theta_i in (-pi/2, pi/2) -> +1, else -1
        signs = np.where(np.cos(opt_thetas) > 0, 1, -1).astype(np.int8)
        
        # Polish with discrete hill climbing
        # Give it a bit of time, but not too much, as we have multiple inits
        polish_time = min(2.0, (deadline - time.time() - 2.0) / 2)
        res_signs, res_radius = discrete_hill_climb(signs, polish_time)
        
        if res_signs is not None and res_radius < best_radius:
            best_radius = res_radius
            best_signs = res_signs.copy()

    # If we still have time, do a "Shake and Local Search" from the best found
    if best_signs is not None and time.time() < deadline - 5.0:
        # Perform several rounds of perturbation
        max_shakes = 20
        for _ in range(max_shakes):
            if time.time() > deadline - 5.0:
                break
            
            # Start from current best
            perturbed = best_signs.copy()
            
            # Randomly flip 1-3 edges
            num_flips = rng.integers(1, 4)
            for _ in range(num_flips):
                idx = rng.integers(0, num_edges)
                perturbed[idx] *= -1
            
            # Run local search from perturbed state
            res_signs, res_radius = discrete_hill_climb(perturbed, 2.0)
            
            if res_signs is not None and res_radius < best_radius:
                best_radius = res_radius
                best_signs = res_signs.copy()

    # Fallback
    if best_signs is None:
        best_signs = np.ones(num_edges, dtype=np.int8)
        if not is_lift_connected(best_signs):
            # Try to find any valid connected signing
            for _ in range(1000):
                idx = np.random.randint(num_edges)
                best_signs[idx] *= -1
                if is_lift_connected(best_signs):
                    break

    return best_signs
