"""Circle packing in unit square via 2D lattice gas parallel tempering Monte Carlo."""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START
import numpy as np

def _feasible(x, y, r, tol=1e-10):
    """Check if circles are feasible (disjoint interiors, inside unit square)."""
    n = len(r)
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    # Vectorized pairwise check
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    sep = r[:, None] + r[None, :]
    # Check upper triangle only
    iu = np.triu_indices(n, 1)
    if np.any(dist[iu] < sep[iu] - tol):
        return False
    return True

def _energy(x, y, r, beta):
    """
    Lattice gas energy: sum of radii minus penalty for violations.
    Lower energy is better. We minimize energy.
    """
    # Boundary violations
    vl = np.maximum(0, r - x)
    vr = np.maximum(0, x + r - 1)
    vb = np.maximum(0, r - y)
    vt = np.maximum(0, y + r - 1)
    boundary_pen = np.sum(vl**2 + vr**2 + vb**2 + vt**2)
    
    # Pairwise overlap penalties
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    sep = r[:, None] + r[None, :]
    iu = np.triu_indices(N, 1)
    overlaps = np.maximum(0, sep[iu] - dist[iu])
    pairwise_pen = np.sum(overlaps**2)
    
    # Energy = -sum(r) + beta * penalties  (we want to maximize sum(r), minimize penalties)
    energy = -np.sum(r) + beta * (boundary_pen + pairwise_pen)
    return energy

def _random_state(rng):
    """Generate a random feasible-ish initial state."""
    # Start with a grid and add noise
    side = 6
    x = np.array([(i + 0.5) / side + rng.normal(0, 0.02) for i in range(N)])
    y = np.array([(j + 0.5) / side + rng.normal(0, 0.02) for j in range(N)])
    # Map to actual positions
    x = np.array([(i % side + 0.5) / side + rng.normal(0, 0.02) for i in range(N)])
    y = np.array([(i // side + 0.5) / side + rng.normal(0, 0.02) for i in range(N)])
    r = np.full(N, 0.075)
    # Clamp to bounds
    r = np.clip(r, 0.005, 0.4)
    x = np.clip(x, r, 1 - r)
    y = np.clip(y, r, 1 - r)
    return x, y, r

def _perturb(x, y, r, rng, pos_scale=0.015, rad_scale=0.004):
    """
    Perturb one random circle: move position and/or change radius.
    Returns new state.
    """
    x_new = x.copy()
    y_new = y.copy()
    r_new = r.copy()
    
    idx = rng.randint(N)
    
    # Decide move type: position (70%), radius (20%), both (10%)
    move_type = rng.random()
    if move_type < 0.7:
        # Position move
        x_new[idx] += rng.normal(0, pos_scale)
        y_new[idx] += rng.normal(0, pos_scale)
    elif move_type < 0.9:
        # Radius move
        r_new[idx] += rng.normal(0, rad_scale)
        r_new[idx] = np.clip(r_new[idx], 0.005, 0.4)
    else:
        # Both
        x_new[idx] += rng.normal(0, pos_scale * 0.7)
        y_new[idx] += rng.normal(0, pos_scale * 0.7)
        r_new[idx] += rng.normal(0, rad_scale * 0.7)
        r_new[idx] = np.clip(r_new[idx], 0.005, 0.4)
    
    # Clamp position to stay in bounds
    x_new[idx] = np.clip(x_new[idx], r_new[idx], 1 - r_new[idx])
    y_new[idx] = np.clip(y_new[idx], r_new[idx], 1 - r_new[idx])
    
    return x_new, y_new, r_new

def _parallel_tempering(rng, num_replicas=12, total_sweeps=8000,
                        T_min=0.01, T_max=1.0, beta=200.0,
                        pos_scale=0.015, rad_scale=0.004):
    """
    Parallel tempering Monte Carlo.
    Minimize energy = -sum(r) + beta * violations.
    """
    # Initialize replicas
    states = []
    energies = []
    for rep in range(num_replicas):
        x, y, r = _random_state(rng)
        states.append((x, y, r))
        energies.append(_energy(x, y, r, beta))
    
    # Temperatures: geometric from T_min to T_max
    temps = np.geomspace(T_min, T_max, num_replicas)
    
    best_state = None
    best_energy = np.inf
    
    # Track best feasible
    for rep in range(num_replicas):
        x, y, r = states[rep]
        if _feasible(x, y, r):
            e = _energy(x, y, r, beta)
            if e < best_energy:
                best_energy = e
                best_state = (x.copy(), y.copy(), r.copy())
    
    # Main loop
    for sweep in range(total_sweeps):
        # Perturb each replica
        for rep in range(num_replicas):
            x, y, r = states[rep]
            x_new, y_new, r_new = _perturb(x, y, r, rng, pos_scale, rad_scale)
            
            e_old = energies[rep]
            e_new = _energy(x_new, y_new, r_new, beta)
            
            # Metropolis: accept if energy decreases, or with probability exp(-delta/T)
            delta = e_new - e_old
            if delta < 0 or rng.random() < np.exp(-delta / temps[rep]):
                states[rep] = (x_new, y_new, r_new)
                energies[rep] = e_new
                
                # Track best feasible
                if _feasible(x_new, y_new, r_new):
                    e_feas = _energy(x_new, y_new, r_new, beta)
                    if e_feas < best_energy:
                        best_energy = e_feas
                        best_state = (x_new.copy(), y_new.copy(), r_new.copy())
        
        # Replica exchange: try swapping adjacent replicas
        for rep in range(num_replicas - 1):
            a, b = rep, rep + 1
            e_a, e_b = energies[a], energies[b]
            T_a, T_b = temps[a], temps[b]
            
            # Swap probability: exp(-(e_b - e_a) * (1/T_a - 1/T_b))
            delta_swap = (e_b - e_a) * (1.0 / T_a - 1.0 / T_b)
            if delta_swap < 0 or rng.random() < np.exp(-delta_swap):
                states[a], states[b] = states[b], states[a]
                energies[a], energies[b] = energies[b], energies[a]
    
    return best_state, best_energy

def _local_refine(x, y, r, max_rounds=30, step_sizes=(0.002, 0.001, 0.0005, 0.0002)):
    """
    Local refinement: try to increase radii while maintaining feasibility.
    """
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    for _ in range(max_rounds):
        improved = False
        for i in range(N):
            for dr in step_sizes:
                r_test = r.copy()
                r_test[i] += dr
                if _feasible(x, y, r_test):
                    r = r_test
                    improved = True
                    break
        if not improved:
            break
    
    return x, y, r

def construct():
    """Construct 26 disjoint circles in unit square maximizing sum of radii."""
    start_time = time.time()
    time_budget = 100.0  # Leave 20s margin
    
    rng = np.random.RandomState(42)
    
    # Run parallel tempering
    best_state, best_energy = _parallel_tempering(
        rng,
        num_replicas=12,
        total_sweeps=6000,
        T_min=0.01,
        T_max=1.0,
        beta=200.0,
        pos_scale=0.015,
        rad_scale=0.004
    )
    
    elapsed = time.time() - start_time
    
    # If we have time, run more sweeps
    if best_state is not None and elapsed < time_budget * 0.6:
        x, y, r = best_state
        # Continue from best state with more replicas
        rng2 = np.random.RandomState(123)
        best_state2, best_energy2 = _parallel_tempering(
            rng2,
            num_replicas=10,
            total_sweeps=4000,
            T_min=0.005,
            T_max=0.5,
            beta=300.0,
            pos_scale=0.01,
            rad_scale=0.003
        )
        if best_state2 is not None:
            e2 = _energy(best_state2[0], best_state2[1], best_state2[2], 300.0)
            e1 = _energy(x, y, r, 300.0)
            if e2 < e1:
                best_state = best_state2
    
    if best_state is None:
        # Fallback: grid
        side = 6
        cells = []
        for i in range(N):
            ix = i % side
            iy = i // side
            cells.append(((ix + 0.5) / side, (iy + 0.5) / side, 0.075))
        return np.array(cells)
    
    x, y, r = best_state
    
    # Local refinement
    x, y, r = _local_refine(x, y, r)
    
    # Final feasibility check
    if not _feasible(x, y, r):
        for _ in range(200):
            r = r * 0.998
            if _feasible(x, y, r):
                break
    
    result = np.column_stack([x, y, r])
    return result
# EVOLVE-BLOCK-END
