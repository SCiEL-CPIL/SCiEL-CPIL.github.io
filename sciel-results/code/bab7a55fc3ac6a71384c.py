"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def sign_edges(graph, budget):
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2
    
    # Precompute edge endpoints as arrays for fast matrix construction
    edge_u = np.array([e[0] for e in edges], dtype=int)
    edge_v = np.array([e[1] for e in edges], dtype=int)
    
    # Precompute adjacency structure for fast connectivity checks
    # For each vertex, store which edges are incident
    vertex_edges = [[] for _ in range(n)]
    for i in range(num_edges):
        u, v = edges[i]
        vertex_edges[u].append(i)
        vertex_edges[v].append(i)

    def spectral_radius(signs):
        A = np.zeros((n, n))
        A[edge_u, edge_v] = signs
        A[edge_v, edge_u] = signs
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))

    def is_connected_lift(signs):
        # Fast BFS on the 2-lift using precomputed structure
        # Lift vertex (v, side) has index v*2 + side
        # Edge i connects (u, s) to (v, s*signs[i]) where s in {0,1}
        # Represent side as 0 or 1, and s*sign means: if sign=+1, same side; if sign=-1, flip side
        # side2 = side if signs[i]==1 else 1-side = side ^ (1 if signs[i]==-1 else 0)
        
        # Precompute flip mask: flip[i] = 0 if signs[i]==1, 1 if signs[i]==-1
        flip = (signs == -1).astype(np.int8)
        
        visited = np.zeros(2 * n, dtype=bool)
        visited[0] = True
        queue = [0]
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            v = node // 2
            side = node % 2
            for ei in vertex_edges[v]:
                eu, ev = edges[ei]
                if eu == v:
                    other = ev
                else:
                    other = eu
                other_side = side ^ flip[ei]
                other_node = other * 2 + other_side
                if not visited[other_node]:
                    visited[other_node] = True
                    queue.append(other_node)
        
        return visited.all()

    def is_valid(signs):
        return is_connected_lift(signs)

    rng = np.random.default_rng(42)
    
    best_signs = None
    best_radius = float('inf')
    
    # Phase 1: Continuous Optimization to find a good initial basin
    # We minimize the spectral radius of A(x) where x in [-1, 1]
    # x_i corresponds to the sign of edge i, relaxed to continuous.
    
    def continuous_spectral_radius_and_grad(x):
        A = np.zeros((n, n))
        A[edge_u, edge_v] = x
        A[edge_v, edge_u] = x
        eigvals, eigvecs = np.linalg.eigh(A)
        
        # Find max absolute eigenvalue
        abs_eigvals = np.abs(eigvals)
        idx_max = np.argmax(abs_eigvals)
        val_max = abs_eigvals[idx_max]
        
        # Gradient calculation
        # If eigvals[idx_max] > 0, lambda = eigvals[idx_max], grad = v v^T
        # If eigvals[idx_max] < 0, lambda = -eigvals[idx_max], grad = -v v^T
        v = eigvecs[:, idx_max]
        if eigvals[idx_max] > 0:
            grad_matrix = np.outer(v, v)
        else:
            grad_matrix = -np.outer(v, v)
            
        # Map gradient matrix back to edge variables
        # d(lambda)/d(x_i) = grad_matrix[u_i, v_i]
        # Since A is symmetric and x_i appears in A[u,v] and A[v,u],
        # and lambda is a function of A, d lambda / d x_i = d lambda / d A[u,v] * d A[u,v] / d x_i + d lambda / d A[v,u] * d A[v,u] / d x_i
        # d A[u,v] / d x_i = 1, d A[v,u] / d x_i = 1
        # So grad_x[i] = grad_matrix[u_i, v_i] + grad_matrix[v_i, u_i] = 2 * grad_matrix[u_i, v_i]
        # Wait, if we define A[x] such that A[u,v]=x, then the chain rule applies.
        # Let's verify: lambda = v^T A v. d lambda / d x_i = v^T (d A / d x_i) v.
        # d A / d x_i has 1 at (u,v) and (v,u).
        # So v^T (d A / d x_i) v = v_u v_v + v_v v_u = 2 v_u v_v.
        # My grad_matrix is v v^T (or -v v^T). So grad_matrix[u,v] is v_u v_v.
        # So grad_x[i] = 2 * grad_matrix[u_i, v_i].
        
        grad_x = 2.0 * grad_matrix[edge_u, edge_v]
        
        return val_max, grad_x

    # Run gradient descent for a fixed number of steps or time
    # We'll do a few restarts of continuous GD
    num_cont_restarts = 5
    steps_per_cont = 200
    
    initial_signs_list = []
    
    for r in range(num_cont_restarts):
        if time.time() > deadline:
            break
            
        # Random continuous start
        x = rng.uniform(-1, 1, size=num_edges)
        
        # Gradient Descent
        lr = 0.1
        for step in range(steps_per_cont):
            if time.time() > deadline:
                break
            val, grad = continuous_spectral_radius_and_grad(x)
            # Project gradient onto feasible set? No, just step and clip
            x = x - lr * grad
            x = np.clip(x, -1, 1)
            
            # Adaptive learning rate? Keep it simple.
            if step % 50 == 0:
                lr *= 0.9
                
        # Round to signs
        signs = np.sign(x)
        # Handle zeros (unlikely with uniform)
        signs[signs == 0] = 1
        
        # Check validity
        if is_valid(signs):
            initial_signs_list.append(signs.copy())
            # Also check radius for tracking
            rad = spectral_radius(signs)
            if rad < best_radius:
                best_radius = rad
                best_signs = signs.copy()
        else:
            # If invalid, try to fix with a few random flips
            valid = False
            for _ in range(100):
                if time.time() > deadline:
                    break
                flip_idx = rng.integers(num_edges)
                signs[flip_idx] *= -1
                if is_valid(signs):
                    valid = True
                    initial_signs_list.append(signs.copy())
                    rad = spectral_radius(signs)
                    if rad < best_radius:
                        best_radius = rad
                        best_signs = signs.copy()
                    break

    # Phase 2: Discrete Local Search (Simulated Annealing)
    # Use the best initial signs found in Phase 1 as a warm start, plus some random restarts
    
    # If we found good initial signs, use them. Otherwise use random.
    start_signs_pool = initial_signs_list
    if not start_signs_pool:
        # Fallback to random
        s = np.ones(num_edges, dtype=int)
        if not is_valid(s):
             # Try to find any valid
             for _ in range(1000):
                 s = rng.choice([-1, 1], size=num_edges)
                 if is_valid(s):
                     break
        start_signs_pool = [s]

    # Simulated Annealing parameters
    # We have ~50s left.
    T_start = 0.05
    T_end = 0.0005
    
    # Run SA for multiple restarts, using the pool as starting points
    # We will alternate between random starts and pool starts
    num_sa_restarts = 20
    
    for restart in range(num_sa_restarts):
        if time.time() > deadline:
            break
        
        # Choose start
        if restart < len(start_signs_pool):
            signs = start_signs_pool[restart].copy()
        else:
            signs = rng.choice([-1, 1], size=num_edges)
            if not is_valid(signs):
                valid = False
                for _ in range(300):
                    if time.time() > deadline: break
                    flip_idx = rng.integers(num_edges)
                    signs[flip_idx] *= -1
                    if is_valid(signs):
                        valid = True
                        break
                if not valid:
                    continue

        cur_radius = spectral_radius(signs)
        if cur_radius < best_radius:
            best_radius = cur_radius
            best_signs = signs.copy()
        
        # SA Loop
        # Estimate steps per restart based on time
        time_left = deadline - time.time()
        # Assume ~1ms per step? eigvalsh on 18x18 is fast.
        # Let's just run until time runs out for this restart, but allow early exit if T is very low
        steps_in_restart = 0
        max_steps = 5000 # Cap per restart to ensure diversity
        
        while steps_in_restart < max_steps:
            if time.time() > deadline:
                break
            
            steps_in_restart += 1
            # Linear cooling
            progress = steps_in_restart / max_steps
            T = T_start * (T_end / T_start) ** progress
            
            # Pick random edge to flip
            idx = rng.integers(num_edges)
            old_sign = signs[idx]
            signs[idx] = -old_sign
            
            if is_valid(signs):
                new_radius = spectral_radius(signs)
                delta = new_radius - cur_radius
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    cur_radius = new_radius
                    if cur_radius < best_radius:
                        best_radius = cur_radius
                        best_signs = signs.copy()
                else:
                    signs[idx] = old_sign
            else:
                signs[idx] = old_sign

    # If no valid solution found (shouldn't happen), return all +1
    if best_signs is None:
        signs = np.ones(num_edges, dtype=int)
        if not is_valid(signs):
            signs = -np.ones(num_edges, dtype=int)
            if not is_valid(signs):
                for _ in range(1000):
                    signs = rng.choice([-1, 1], size=num_edges)
                    if is_valid(signs):
                        break
        best_signs = signs
    
    return best_signs
# EVOLVE-BLOCK-END
