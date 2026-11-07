"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_signed_adjacency(graph, signs):
    """Build the signed adjacency matrix from graph and signs."""
    n = graph['n']
    A = np.zeros((n, n))
    for edge, s in zip(graph['edges'], signs):
        u, v = edge
        A[u, v] = s
        A[v, u] = s
    return A

def _spectral_radius(A):
    """Compute spectral radius of symmetric matrix."""
    eigvals = np.linalg.eigvalsh(A)
    return np.max(np.abs(eigvals))

def _is_valid_lift(graph, signs):
    """Check if the lift is connected and 3-regular using check_lift."""
    try:
        from autoresearch_suite.science_tcs_expansion import check_lift
        radius, details = check_lift(graph, signs)
        return True, radius
    except Exception:
        return False, None

def _simulate_annealing(graph, initial_signs, time_limit, start_time):
    """
    Simulated Annealing to minimize spectral radius.
    """
    n_edges = len(graph['edges'])
    signs = initial_signs.copy()
    
    # Check initial validity
    valid, current_radius = _is_valid_lift(graph, signs)
    if not valid:
        current_radius = float('inf')
    
    best_signs = signs.copy()
    best_radius = current_radius
    T = 1.0
    T_end = 0.01
    alpha = 0.995
    rng = np.random.default_rng(123)
    
    while time.time() - start_time < time_limit:
        # Pick a random edge to flip
        i = rng.integers(0, n_edges)
        old_sign = signs[i]
        signs[i] = -old_sign
        
        # Evaluate new state
        valid, new_radius = _is_valid_lift(graph, signs)
        
        if not valid:
            new_radius = float('inf')
        
        # Calculate delta
        delta = new_radius - current_radius
        
        # Metropolis criterion
        accepted = False
        if delta < 0:
            accepted = True
        elif current_radius < float('inf'):
            # Avoid overflow in exp
            if delta / T < 50: 
                if rng.random() < np.exp(-delta / T):
                    accepted = True
        
        if accepted:
            current_radius = new_radius
            if valid and new_radius < best_radius:
                best_radius = new_radius
                best_signs = signs.copy()
        else:
            signs[i] = old_sign
            
        T = max(T * alpha, T_end)
        
    return best_signs, best_radius

def sign_edges(graph, budget):
    m = graph['n'] // 2
    n_edges = len(graph['edges'])
    
    # Time limit management
    start_time = time.time()
    time_limit = 55.0 # Leave 5s buffer for overhead
    
    best_signs_overall = None
    best_radius_overall = float('inf')
    
    # Generate diverse initial states
    init_strategies = []
    
    # Strategy 0: All +1
    init_strategies.append(np.ones(n_edges, dtype=int))
    
    # Strategy 1: Alternating on cycles, +1 on matching
    signs_alt_pos = np.zeros(n_edges, dtype=int)
    for i, (u, v) in enumerate(graph['edges']):
        if u < m and v < m:
            signs_alt_pos[i] = 1 if (u % 2 == 0) else -1
        elif u >= m and v >= m:
            signs_alt_pos[i] = 1 if ((u - m) % 2 == 0) else -1
        else:
            signs_alt_pos[i] = 1
    init_strategies.append(signs_alt_pos)
    
    # Strategy 2: Alternating on cycles, -1 on matching
    signs_alt_neg = signs_alt_pos.copy()
    for i, (u, v) in enumerate(graph['edges']):
        if not (u < m and v < m) and not (u >= m and v >= m):
            signs_alt_neg[i] = -1
    init_strategies.append(signs_alt_neg)
    
    # Add random initializations
    rng_init = np.random.default_rng(99)
    for _ in range(5):
        init_strategies.append(rng_init.choice([-1, 1], size=n_edges))
        
    for init_signs in init_strategies:
        if time.time() - start_time > time_limit:
            break
            
        # Run Simulated Annealing
        # We pass the remaining time limit to the SA function
        # The SA function will run until the global time limit is reached
        # To allow multiple restarts, we should allocate a portion of the time to each.
        # Let's allocate 1/8 of the time to each of the 8 strategies (approx).
        # Or just let the first few run longer.
        # Better: Run SA for a fixed short time, then move to next.
        # Or: Run SA until time limit, but that means only 1 restart.
        # Let's do multiple short runs.
        
        # Allocate 5 seconds per restart
        restart_time_limit = min(5.0, time_limit - (time.time() - start_time))
        if restart_time_limit <= 0:
            break
            
        # Adjust start_time for the SA call? No, SA checks global time.
        # We need to modify _simulate_annealing to take a specific end time or duration.
        
        # Let's refactor _simulate_annealing to take a duration.
        # Actually, the current implementation of _simulate_annealing takes time_limit and start_time.
        # It runs while time.time() - start_time < time_limit.
        # So if I pass time_limit = 5.0, it will run for 5 seconds from start_time.
        # This is correct.
        
        signs, radius = _simulate_annealing(graph, init_signs, 5.0, start_time)
        
        if radius < best_radius_overall:
            best_radius_overall = radius
            best_signs_overall = signs.copy()
            
    if best_signs_overall is None:
        # Fallback to seed behavior if something went wrong
        from autoresearch_suite.science_tcs_expansion import lift_baseline
        return lift_baseline(graph, budget)
        
    return best_signs_overall
# EVOLVE-BLOCK-END
