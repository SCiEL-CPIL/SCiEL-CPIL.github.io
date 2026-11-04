"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def _evaluate(xy, r):
    """
    Evaluate the packing quality.
    Returns (score, is_valid).
    score is sum(r) if valid, 0 otherwise.
    """
    n = len(xy)
    # Check walls
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    # Check pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True

def _energy_and_grad(xy, r, beta):
    """
    Compute a smooth energy function and its gradient.
    Energy penalizes overlaps and wall violations.
    We want to MINIMIZE energy.
    
    E = sum over pairs (r_i + r_j - dist_ij)^2 if dist < r_i + r_j
      + sum over circles (r_i - min(x, 1-x, y, 1-y))^2 if outside
      - lambda * sum(r_i)  # We want to maximize sum(r), so minimize -sum(r)
    
    However, a better formulation for "growing" circles is to fix the "hardness" 
    and let the optimizer find positions, or vice versa.
    
    Let's use a penalty method:
    Objective: Maximize Sum(r) subject to constraints.
    Equivalent to: Minimize -Sum(r) + Penalty(Constraints).
    
    Let's define:
    f(x, r) = -sum(r) + beta * sum( max(0, r_i + r_j - d_ij)^2 ) + beta * sum( max(0, r_i - wall_i)^2 )
    
    Gradient with respect to x_i:
    d/dx_i [ max(0, r_i + r_j - d_ij)^2 ] = 2 * max(0, r_i + r_j - d_ij) * (-(x_i - x_j)/d_ij)
    
    Gradient with respect to r_i:
    d/dr_i [ -sum(r) ] = -1
    d/dr_i [ max(0, r_i + r_j - d_ij)^2 ] = 2 * max(0, r_i + r_j - d_ij) * 1
    d/dr_i [ max(0, r_i - wall_i)^2 ] = 2 * max(0, r_i - wall_i) * 1
    """
    n = len(xy)
    grad_x = np.zeros_like(xy)
    grad_r = np.zeros_like(r)
    energy = -np.sum(r) # We want to maximize sum(r), so minimize -sum(r)
    
    # Pairwise interactions
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            d = np.sqrt(dx * dx + dy * dy)
            if d < 1e-12:
                d = 1e-12
                dx = 1e-12
                dy = 0.0
            
            overlap = r[i] + r[j] - d
            if overlap > 0:
                # Penalty term: overlap^2
                energy += beta * overlap * overlap
                # Gradient w.r.t x_i
                # d(overlap^2)/dx_i = 2 * overlap * d(overlap)/dx_i
                # d(overlap)/dx_i = -dx/d
                grad_factor = 2 * beta * overlap * (-dx / d)
                grad_x[i, 0] += grad_factor
                grad_x[i, 1] += 2 * beta * overlap * (-dy / d)
                
                # Gradient w.r.t x_j
                # d(overlap)/dx_j = dx/d
                grad_x[j, 0] -= grad_factor
                grad_x[j, 1] -= 2 * beta * overlap * (-dy / d)
                
                # Gradient w.r.t r_i
                grad_r[i] += 2 * beta * overlap
                # Gradient w.r.t r_j
                grad_r[j] += 2 * beta * overlap
    
    # Wall constraints
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        
        # Left wall: x - ri < 0
        viol = ri - x
        if viol > 0:
            energy += beta * viol * viol
            # d/dx_i (ri - x)^2 = 2(ri-x)(-1)
            grad_x[i, 0] += 2 * beta * viol * (-1)
            # d/dr_i (ri - x)^2 = 2(ri-x)(1)
            grad_r[i] += 2 * beta * viol
        
        # Right wall: 1 - x - ri < 0  => x + ri > 1
        viol = x + ri - 1.0
        if viol > 0:
            energy += beta * viol * viol
            # d/dx_i (x + ri - 1)^2 = 2(x+ri-1)(1)
            grad_x[i, 0] += 2 * beta * viol * (1)
            # d/dr_i (x + ri - 1)^2 = 2(x+ri-1)(1)
            grad_r[i] += 2 * beta * viol
        
        # Bottom wall: y - ri < 0
        viol = ri - y
        if viol > 0:
            energy += beta * viol * viol
            grad_x[i, 1] += 2 * beta * viol * (-1)
            grad_r[i] += 2 * beta * viol
        
        # Top wall: 1 - y - ri < 0 => y + ri > 1
        viol = y + ri - 1.0
        if viol > 0:
            energy += beta * viol * viol
            grad_x[i, 1] += 2 * beta * viol * (1)
            grad_r[i] += 2 * beta * viol
            
    return energy, grad_x, grad_r

def _langevin_step(xy, r, T, dt, beta, rng):
    """
    Perform one Langevin dynamics step.
    dx = -grad_x * dt + sqrt(2*T*dt) * noise
    dr = -grad_r * dt + sqrt(2*T*dt) * noise
    """
    n = len(xy)
    energy, grad_x, grad_r = _energy_and_grad(xy, r, beta)
    
    noise_x = rng.normal(0, np.sqrt(2 * T * dt), size=xy.shape)
    noise_r = rng.normal(0, np.sqrt(2 * T * dt), size=r.shape)
    
    new_xy = xy - grad_x * dt + noise_x
    new_r = r - grad_r * dt + noise_r
    
    # Clip radii to be positive and small enough to fit in square
    new_r = np.clip(new_r, 0.001, 0.49)
    
    # Clip positions to keep centers within [0,1] roughly (walls handled by energy)
    # But hard clipping helps stability
    new_xy[:, 0] = np.clip(new_xy[:, 0], 0.0, 1.0)
    new_xy[:, 1] = np.clip(new_xy[:, 1], 0.0, 1.0)
    
    return new_xy, new_r, energy

def _optimize_langevin(xy, r, deadline, seed=0):
    """
    Run Langevin dynamics with adaptive cooling.
    """
    n = len(xy)
    rng = np.random.RandomState(seed)
    
    current_xy = xy.copy()
    current_r = r.copy()
    
    best_xy = current_xy.copy()
    best_r = current_r.copy()
    best_score = 0.0
    
    # Initial evaluation
    score, valid = _evaluate(current_xy, current_r)
    if valid:
        best_score = score
    
    T = 0.1
    T_min = 0.001
    beta = 100.0 # Penalty strength
    dt = 0.01
    
    # Adaptive cooling
    cooling_rate = 0.99
    
    max_iter = 50000
    iter_count = 0
    
    while time.time() < deadline and T > T_min and iter_count < max_iter:
        # Step
        new_xy, new_r, energy = _langevin_step(current_xy, current_r, T, dt, beta, rng)
        
        # Evaluate new state
        new_score, new_valid = _evaluate(new_xy, new_r)
        
        # Acceptance criterion (Metropolis-like based on energy or score)
        # If valid, we prefer higher score.
        # If invalid, we accept with prob exp(-(energy_new - energy_old)/T)
        
        # For simplicity, let's use the energy function for acceptance when invalid,
        # and score for valid.
        
        if new_valid:
            # Always accept valid moves if they improve score? 
            # Or use energy? Energy includes -sum(r), so lower energy = higher sum(r) + less penalty.
            # If valid, penalty is 0, so energy = -sum(r).
            # So minimizing energy is maximizing sum(r).
            # Let's use energy difference for acceptance always.
            pass
            
        # Calculate energy difference
        # We need old energy. _langevin_step returned new energy.
        # We didn't store old energy efficiently. Let's recompute or store.
        # Optimization: store old energy from previous step.
        
        # Actually, let's just recompute old energy if needed, or store it.
        # Let's store old_energy.
        
        # Refinement: Store old_energy outside loop.
        # This requires restructuring. Let's do a simple check:
        # If new is valid and better, accept.
        # If new is invalid, accept with prob exp(-delta_E / T).
        
        # To get delta_E, we need old_E.
        # Let's just use a simpler heuristic: 
        # Accept if new_score > current_score (if current valid)
        # Or if new is valid and current is invalid.
        # Otherwise, accept with prob exp(- (new_penalty - old_penalty) / T)
        
        # Let's stick to standard Metropolis on the Energy function E(x,r).
        # E = -Sum(r) + Penalty.
        # We want to minimize E.
        
        # We need old_E.
        # Let's modify _langevin_step to also return old_E? No, it calculates new.
        # Let's calculate old_E before the step.
        
        # This is inefficient. Let's assume we track `current_energy` from the previous step.
        # Initialize current_energy.
        
        # Let's restart this function with proper energy tracking.
        pass
        
    return best_xy, best_r, best_score

def _optimize_langevin_v2(xy, r, deadline, seed=0):
    n = len(xy)
    rng = np.random.RandomState(seed)
    
    current_xy = xy.copy()
    current_r = r.copy()
    
    # Initial energy
    current_energy, _, _ = _energy_and_grad(current_xy, current_r, beta=100.0)
    
    best_xy = current_xy.copy()
    best_r = current_r.copy()
    best_score = 0.0
    
    score, valid = _evaluate(current_xy, current_r)
    if valid:
        best_score = score
    
    T = 0.05
    T_min = 0.0005
    beta = 500.0 # Strong penalty to enforce constraints
    dt = 0.005
    
    cooling_rate = 0.995
    
    max_iter = 100000
    
    while time.time() < deadline and T > T_min:
        # Langevin Step
        energy, grad_x, grad_r = _energy_and_grad(current_xy, current_r, beta)
        
        noise_x = rng.normal(0, np.sqrt(2 * T * dt), size=xy.shape)
        noise_r = rng.normal(0, np.sqrt(2 * T * dt), size=r.shape)
        
        new_xy = current_xy - grad_x * dt + noise_x
        new_r = current_r - grad_r * dt + noise_r
        
        # Clip
        new_r = np.clip(new_r, 0.001, 0.49)
        new_xy[:, 0] = np.clip(new_xy[:, 0], 0.0, 1.0)
        new_xy[:, 1] = np.clip(new_xy[:, 1], 0.0, 1.0)
        
        # Calculate new energy
        new_energy, _, _ = _energy_and_grad(new_xy, new_r, beta)
        
        delta_E = new_energy - current_energy
        
        # Metropolis Acceptance
        if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
            current_xy = new_xy
            current_r = new_r
            current_energy = new_energy
            
            # Check if this is a valid packing and if it's the best
            score, valid = _evaluate(current_xy, current_r)
            if valid and score > best_score:
                best_score = score
                best_xy = current_xy.copy()
                best_r = current_r.copy()
        
        T *= cooling_rate
        
    return best_xy, best_r, best_score

def _grid_init(side):
    """Create a grid of up to N circles."""
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.002
                positions.append([x, y, r])
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.01])
    return np.array(positions[:N])

def _random_init(n, seed=0):
    """Random initialization with small radii."""
    rng = np.random.RandomState(seed)
    xy = rng.uniform(0.1, 0.9, size=(n, 2))
    r = rng.uniform(0.02, 0.05, size=n)
    return xy, r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    # Phase 1: Grid initialization + Langevin
    for side in [5, 6, 7]:
        if time.time() > deadline:
            break
        pos = _grid_init(side)
        xy = pos[:, :2].copy()
        r = pos[:, 2].copy()
        
        out_xy, out_r, total = _optimize_langevin_v2(xy, r, deadline, seed=side)
        if total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # Phase 2: Random initializations + Langevin
    for seed in range(10):
        if time.time() > deadline:
            break
        xy, r = _random_init(N, seed=seed)
        out_xy, out_r, total = _optimize_langevin_v2(xy, r, deadline, seed=100 + seed)
        if total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # Fallback
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        pos = _grid_init(side)
        xy = pos[:, :2].copy()
        r = pos[:, 2].copy()
        total, valid = _evaluate(xy, r)
        if valid:
            best_total = total
            best_valid = True
            best_xy = xy
            best_r = r
        else:
            result = np.zeros((N, 3))
            for i in range(N):
                gx = i % side
                gy = i // side
                result[i, 0] = (gx + 0.5) / side
                result[i, 1] = (gy + 0.5) / side
                result[i, 2] = 0.4 / side
            return result

    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - y)

    # Final check
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _evaluate(final_xy, final_r)
    if not valid:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side

    return result
# EVOLVE-BLOCK-END
