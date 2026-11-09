"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def sign_edges(graph, budget):
    from autoresearch_suite.science_tcs_expansion import check_lift
    
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    m = graph['n'] // 2
    edges = graph['edges']
    n_edges = len(edges)
    n = graph['n']
    
    u_arr = np.array([e[0] for e in edges], dtype=np.int32)
    v_arr = np.array([e[1] for e in edges], dtype=np.int32)
    
    def build_matrix(signs):
        A = np.zeros((n, n), dtype=np.float64)
        for i in range(n_edges):
            s = float(signs[i])
            A[u_arr[i], v_arr[i]] = s
            A[v_arr[i], u_arr[i]] = s
        return A
    
    def spectral_radius(signs):
        A = build_matrix(signs)
        eigvals = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(eigvals)))
    
    def is_valid_lift(signs):
        """Check if the lift is connected and 3-regular."""
        try:
            radius, details = check_lift(graph, signs)
            return True
        except Exception:
            return False
    
    # Phase 1: Generate initial candidates using phase synchronization approach
    # Parameterize each vertex with a complex unit vector z_v
    # Edge sign = sign(Re(z_u * conj(z_v)))
    
    def phase_to_signs(phases):
        """Convert continuous phases to discrete signs."""
        signs = np.ones(n_edges, dtype=np.int8)
        for i in range(n_edges):
            u, v = u_arr[i], v_arr[i]
            cos_val = np.cos(phases[u] - phases[v])
            signs[i] = 1 if cos_val >= 0 else -1
        return signs
    
    def spectral_radius_from_phases(phases):
        signs = phase_to_signs(phases)
        return spectral_radius(signs)
    
    # Gradient descent on phases to minimize spectral radius
    def optimize_phases(init_phases, max_iters=5000, lr=0.05):
        phases = init_phases.copy()
        best_radius = spectral_radius_from_phases(phases)
        best_phases = phases.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute gradient via finite differences
            grad = np.zeros(n)
            eps = 0.01
            for v in range(n):
                phases_plus = phases.copy()
                phases_plus[v] += eps
                phases_minus = phases.copy()
                phases_minus[v] -= eps
                f_plus = spectral_radius_from_phases(phases_plus)
                f_minus = spectral_radius_from_phases(phases_minus)
                grad[v] = (f_plus - f_minus) / (2 * eps)
            
            # Gradient step
            phases = phases - lr * grad
            # Wrap phases to [-pi, pi]
            phases = (phases + np.pi) % (2 * np.pi) - np.pi
            
            current_radius = spectral_radius_from_phases(phases)
            if current_radius < best_radius - 1e-10:
                best_radius = current_radius
                best_phases = phases.copy()
        
        return best_phases, best_radius
    
    # Generate multiple initial phase configurations
    rng = np.random.default_rng(3)
    
    candidates = []
    
    # Random initial phases
    for trial in range(5):
        if time.time() > deadline:
            break
        init_phases = rng.uniform(-np.pi, np.pi, size=n)
        best_phases, best_radius = optimize_phases(init_phases, max_iters=2000, lr=0.03)
        signs = phase_to_signs(best_phases)
        candidates.append((signs, best_radius))
    
    # Also try some structured initializations
    # All zeros
    init_phases = np.zeros(n)
    best_phases, best_radius = optimize_phases(init_phases, max_iters=2000, lr=0.03)
    signs = phase_to_signs(best_phases)
    candidates.append((signs, best_radius))
    
    # Alternating phases
    init_phases = np.array([0 if i < m else np.pi for i in range(n)])
    best_phases, best_radius = optimize_phases(init_phases, max_iters=2000, lr=0.03)
    signs = phase_to_signs(best_phases)
    candidates.append((signs, best_radius))
    
    # Phase 2: Local search (single + pairwise flips) on the best candidates
    def local_search(signs, max_iters=100000):
        signs = signs.copy()
        current_radius = spectral_radius(signs)
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            
            # Single flips
            for i in range(n_edges):
                if time.time() > deadline:
                    return signs, current_radius
                signs[i] *= -1
                new_radius = spectral_radius(signs)
                if new_radius < current_radius - 1e-10:
                    current_radius = new_radius
                    improved = True
                else:
                    signs[i] *= -1
            
            # Pairwise flips
            for i in range(n_edges):
                if time.time() > deadline:
                    return signs, current_radius
                for j in range(i + 1, n_edges):
                    if time.time() > deadline:
                        return signs, current_radius
                    signs[i] *= -1
                    signs[j] *= -1
                    new_radius = spectral_radius(signs)
                    if new_radius < current_radius - 1e-10:
                        current_radius = new_radius
                        improved = True
                    else:
                        signs[i] *= -1
                        signs[j] *= -1
        
        return signs, current_radius
    
    # Sort candidates by radius and do local search on top ones
    candidates.sort(key=lambda x: x[1])
    
    best_signs = None
    best_radius = float('inf')
    
    for signs, radius in candidates[:5]:
        if time.time() > deadline:
            break
        ls_signs, ls_radius = local_search(signs, max_iters=50000)
        if ls_radius < best_radius:
            best_radius = ls_radius
            best_signs = ls_signs
    
    # Phase 3: Validate and return
    if best_signs is None:
        # Fallback: use the first candidate
        best_signs = candidates[0][0]
    
    # Verify validity
    if not is_valid_lift(best_signs):
        # Try other candidates
        for signs, radius in candidates:
            if signs is not best_signs and is_valid_lift(signs):
                best_signs = signs
                break
        else:
            # All invalid, generate random valid signs
            for attempt in range(100):
                if time.time() > deadline:
                    break
                signs = rng.choice([-1, 1], size=n_edges).astype(np.int8)
                if is_valid_lift(signs):
                    best_signs = signs
                    break
    
    return best_signs
# EVOLVE-BLOCK-END
