"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 105.0  # seconds, safety margin from 120s limit


def hexagonal_init(n, rng):
    """Generate a hexagonal-lattice-inspired initial configuration for n circles."""
    side = int(np.ceil(np.sqrt(n)))
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) >= n:
                break
            x = (i + 0.5) / side + (j % 2) * 0.3 / side
            y = (j + 0.5) / side
            positions.append([x, y])
    # If we don't have enough, add more in a grid
    while len(positions) < n:
        i = len(positions) // side
        j = len(positions) % side
        x = (i + 0.5) / side
        y = (j + 0.5) / side
        positions.append([x, y])
    positions = np.array(positions[:n])
    # Add small random perturbation to break symmetry
    positions += rng.uniform(-0.02, 0.02, size=positions.shape)
    positions = np.clip(positions, 0.01, 0.99)
    # Initial radius estimate
    r0 = 0.5 / side - 0.01
    radii = np.full(n, r0)
    return positions, radii


def objective_and_constraints(x, n):
    """
    x: vector of length 3n: [x1, y1, r1, x2, y2, r2, ...]
    Returns (objective, constraints)
    """
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]
    
    # Objective: maximize sum of radii -> minimize -sum
    obj = -np.sum(rs)
    
    # Constraints (all >= 0):
    # 1. Circles inside unit square: r <= x, r <= 1-x, r <= y, r <= 1-y
    # 2. Disjoint interiors: (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
    # 3. Radii positive: r >= 0
    
    cons = []
    
    # Boundary constraints
    cons.append({'type': 'ineq', 'fun': lambda x, xs=xs, ys=ys, rs=rs: rs - xs})  # r <= x
    cons.append({'type': 'ineq', 'fun': lambda x, xs=xs, ys=ys, rs=rs: rs - (1 - xs)})  # r <= 1-x
    cons.append({'type': 'ineq', 'fun': lambda x, xs=xs, ys=ys, rs=rs: rs - ys})  # r <= y
    cons.append({'type': 'ineq', 'fun': lambda x, xs=xs, ys=ys, rs=rs: rs - (1 - ys)})  # r <= 1-y
    cons.append({'type': 'ineq', 'fun': lambda x, xs=xs, ys=ys, rs=rs: rs})  # r >= 0
    
    # Pairwise disjointness
    for i in range(n):
        for j in range(i + 1, n):
            def dist_con(x, i=i, j=j):
                xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
                xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
                return (xi - xj)**2 + (yi - yj)**2 - (ri + rj)**2
            cons.append({'type': 'ineq', 'fun': dist_con})
    
    return obj, cons


def objective_only(x, n):
    """Simpler objective for use with penalty method."""
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]
    
    # Penalty for constraint violations
    penalty = 0.0
    mu = 100.0  # penalty weight
    
    # Boundary violations
    penalty += mu * np.sum(np.maximum(0, rs - xs))
    penalty += mu * np.sum(np.maximum(0, rs - (1 - xs)))
    penalty += mu * np.sum(np.maximum(0, rs - ys))
    penalty += mu * np.sum(np.maximum(0, rs - (1 - ys)))
    penalty += mu * np.sum(np.maximum(0, -rs))
    
    # Pairwise violations
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = xs[i], ys[i], rs[i]
            xj, yj, rj = xs[j], ys[j], rs[j]
            dist2 = (xi - xj)**2 + (yi - yj)**2
            sep2 = (ri + rj)**2
            penalty += mu * max(0, sep2 - dist2)
    
    return -np.sum(rs) + penalty


def is_valid(x, n, tol=1e-6):
    """Check if a configuration is valid."""
    xs = x[0::3]
    ys = x[1::3]
    rs = x[2::3]
    
    # Check radii positive
    if np.any(rs < -tol):
        return False
    
    # Check boundaries
    if np.any(xs + rs > 1 + tol) or np.any(xs - rs < -tol):
        return False
    if np.any(ys + rs > 1 + tol) or np.any(ys - rs < -tol):
        return False
    
    # Check pairwise
    for i in range(n):
        for j in range(i + 1, n):
            dist2 = (xs[i] - xs[j])**2 + (ys[i] - ys[j])**2
            sep2 = (rs[i] + rs[j])**2
            if dist2 < sep2 - tol:
                return False
    
    return True


def project_to_valid(x, n, max_iter=1000):
    """Project a configuration to be valid by shrinking radii."""
    xs = x[0::3].copy()
    ys = x[1::3].copy()
    rs = x[2::3].copy()
    
    for _ in range(max_iter):
        valid = True
        
        # Ensure boundaries
        for i in range(n):
            rs[i] = min(rs[i], xs[i], 1 - xs[i], ys[i], 1 - ys[i])
            rs[i] = max(rs[i], 0)
        
        # Ensure pairwise
        for i in range(n):
            for j in range(i + 1, n):
                dist2 = (xs[i] - xs[j])**2 + (ys[i] - ys[j])**2
                dist = np.sqrt(dist2)
                required_sep = rs[i] + rs[j]
                if required_sep > dist and dist > 0:
                    # Shrink both radii proportionally
                    scale = dist / required_sep * 0.999
                    rs[i] *= scale
                    rs[j] *= scale
                    valid = False
        
        if valid:
            break
    
    result = np.empty(3 * n)
    result[0::3] = xs
    result[1::3] = ys
    result[2::3] = rs
    return result


def construct():
    start_time = time.time()
    best_x = None
    best_obj = -np.inf
    
    rng = np.random.default_rng(42)
    
    # Try multiple restarts with different initial configurations
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Generate initial configuration
        if restart == 0:
            positions, radii = hexagonal_init(N, rng)
        else:
            # Perturb the previous best or use different seed
            if best_x is not None and restart < 3:
                positions = best_x[0::3] + rng.normal(0, 0.05, size=N)
                radii = best_x[2::3] + rng.normal(0, 0.02, size=N)
                positions = np.clip(positions, 0.05, 0.95)
                radii = np.clip(radii, 0.01, 0.5)
            else:
                # Random initialization
                positions = rng.uniform(0.1, 0.9, size=(N, 2))
                radii = rng.uniform(0.05, 0.15, size=N)
        
        # Build initial x vector
        x0 = np.empty(3 * N)
        x0[0::3] = positions
        x0[1::3] = positions[:, 1]
        x0[2::3] = radii
        
        # Use penalty method optimization
        for trial in range(2):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            try:
                # First pass: looser penalty
                result = minimize(
                    objective_only,
                    x0,
                    args=(N,),
                    method='L-BFGS-B',
                    bounds=[(0, 1)] * 3 * N,
                    options={'maxiter': 2000, 'ftol': 1e-8}
                )
                
                if result.success and result.fun < best_obj:
                    # Project to valid
                    valid_x = project_to_valid(result.x, N)
                    if is_valid(valid_x, N):
                        obj = np.sum(valid_x[2::3])
                        if obj > best_obj:
                            best_obj = obj
                            best_x = valid_x.copy()
                
                # Second pass: tighter optimization from improved point
                if result.x is not None and time.time() - start_time < DEADLINE - 5:
                    result2 = minimize(
                        objective_only,
                        result.x,
                        args=(N,),
                        method='L-BFGS-B',
                        bounds=[(0, 1)] * 3 * N,
                        options={'maxiter': 3000, 'ftol': 1e-10}
                    )
                    
                    if result2.success:
                        valid_x = project_to_valid(result2.x, N)
                        if is_valid(valid_x, N):
                            obj = np.sum(valid_x[2::3])
                            if obj > best_obj:
                                best_obj = obj
                                best_x = valid_x.copy()
                    
            except Exception:
                continue
        
        # Update x0 for next restart
        x0 = best_x if best_x is not None else x0
    
    # Final check and return
    if best_x is None:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.005])
        return np.array(cells[:N])
    
    # Final polish: small local search
    if time.time() - start_time < DEADLINE - 3:
        try:
            # Try to increase radii slightly
            xs = best_x[0::3].copy()
            ys = best_x[1::3].copy()
            rs = best_x[2::3].copy()
            
            # Try increasing all radii uniformly
            for delta in [0.001, 0.0005, 0.0001]:
                new_rs = rs + delta
                # Check if still valid
                valid = True
                for i in range(N):
                    if new_rs[i] > xs[i] or new_rs[i] > 1 - xs[i] or new_rs[i] > ys[i] or new_rs[i] > 1 - ys[i]:
                        valid = False
                        break
                if valid:
                    for i in range(N):
                        for j in range(i + 1, N):
                            dist2 = (xs[i] - xs[j])**2 + (ys[i] - ys[j])**2
                            sep2 = (new_rs[i] + new_rs[j])**2
                            if dist2 < sep2:
                                valid = False
                                break
                        if not valid:
                            break
                if valid:
                    best_x[2::3] = new_rs
                    best_obj = np.sum(new_rs)
        except Exception:
            pass
    
    # Format output: (x, y, radius)
    result = np.empty((N, 3))
    result[:, 0] = best_x[0::3]
    result[:, 1] = best_x[1::3]
    result[:, 2] = best_x[2::3]
    
    return result

# EVOLVE-BLOCK-END
