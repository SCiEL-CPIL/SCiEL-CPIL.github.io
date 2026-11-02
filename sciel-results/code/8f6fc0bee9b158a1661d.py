"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

N = 32

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
import time


def _initial_positions():
    """Create a reasonable initial configuration of 32 circles."""
    # Use a hexagonal packing pattern as starting point
    # 6x6 grid with some removals to get 32
    positions = []
    # Start with a 6x6 grid
    for i in range(6):
        for j in range(6):
            x = (i + 0.5) / 6.0
            y = (j + 0.5) / 6.0
            # Offset odd rows for hexagonal packing
            if j % 2 == 1:
                x += 0.25 / 6.0
            positions.append([x, y])
    
    # Remove 4 corners to get 32
    # Keep positions 0-31
    positions = positions[:32]
    return np.array(positions)


def _compute_radii_from_positions(pos):
    """Compute maximum possible radii given positions, respecting constraints."""
    n = len(pos)
    radii = np.zeros(n)
    
    for i in range(n):
        # Distance to walls
        r_wall = min(pos[i][0], 1.0 - pos[i][0], pos[i][1], 1.0 - pos[i][1])
        
        # Distance to other circles
        r_circles = r_wall
        for j in range(n):
            if i != j:
                dist = np.linalg.norm(pos[i] - pos[j])
                r_circles = min(r_circles, dist / 2.0)
        
        radii[i] = max(0, r_circles)
    
    return radii


def _objective(params):
    """Objective: negative sum of radii (we minimize negative)."""
    n = N
    pos = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    # Check constraints and penalize violations
    penalty = 0.0
    
    # Wall constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
    for i in range(n):
        x, y = pos[i]
        r = radii[i]
        if x < r or x > 1 - r or y < r or y > 1 - r:
            penalty += 1e6 * (max(0, r - x) + max(0, x - (1-r)) + 
                            max(0, r - y) + max(0, y - (1-r)))
    
    # Circle-circle constraints: dist >= r_i + r_j
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(pos[i] - pos[j])
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                penalty += 1e6 * (min_dist - dist)
    
    return -np.sum(radii) + penalty


def _constraints(params):
    """Return list of constraint dictionaries."""
    n = N
    pos = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    cons = []
    
    # Wall constraints
    for i in range(n):
        x, y = pos[i]
        r = radii[i]
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*n+i]})  # x >= r
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: (1 - x[2*i]) - x[2*n+i]})  # 1-x >= r
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*n+i]})  # y >= r
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: (1 - x[2*i+1]) - x[2*n+i]})  # 1-y >= r
    
    # Circle-circle constraints
    for i in range(n):
        for j in range(i+1, n):
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: np.sqrt((x[2*i]-x[2*j])**2 + (x[2*i+1]-x[2*j+1])**2) - x[2*n+i] - x[2*n+j]
            })
    
    return cons


def construct():
    """Pack 32 circles in unit square maximizing sum of radii."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Initial positions
    pos_init = _initial_positions()
    
    # Initial radii: small values
    radii_init = np.full(N, 0.02)
    
    # Initial parameters
    x0 = np.concatenate([pos_init.flatten(), radii_init])
    
    # Bounds: positions in [0,1], radii in [0, 0.5]
    bounds = []
    for i in range(N):
        bounds.append((0, 1))  # x
        bounds.append((0, 1))  # y
    for i in range(N):
        bounds.append((0, 0.5))  # radius
    
    best_params = x0.copy()
    best_obj = _objective(x0)
    
    # Try multiple optimization attempts
    max_attempts = 5
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Add some randomness for different starting points
        if attempt > 0:
            noise_scale = 0.1 / (attempt + 1)
            x_attempt = x0 + noise_scale * np.random.randn(len(x0))
            # Ensure bounds
            x_attempt[:2*N] = np.clip(x_attempt[:2*N], 0.01, 0.99)
            x_attempt[2*N:] = np.clip(x_attempt[2*N:], 0.01, 0.4)
        else:
            x_attempt = x0.copy()
        
        try:
            result = minimize(
                _objective,
                x_attempt,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if result.success and result.fun < best_obj:
                best_params = result.x.copy()
                best_obj = result.fun
                
        except Exception as e:
            pass
        
        # Check time
        if time.time() > deadline - 20:
            break
    
    # Extract results
    pos = best_params[:2*N].reshape(N, 2)
    radii = best_params[2*N:]
    
    # Ensure validity: adjust radii to satisfy constraints
    for i in range(N):
        # Check wall constraints
        max_r_wall = min(pos[i][0], 1 - pos[i][0], pos[i][1], 1 - pos[i][1])
        radii[i] = min(radii[i], max_r_wall)
    
    # Second pass: ensure circle-circle constraints
    for iteration in range(10):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(pos[i] - pos[j])
                if radii[i] + radii[j] > dist:
                    # Scale down both radii
                    excess = (radii[i] + radii[j] - dist) / 2.0
                    radii[i] = max(0, radii[i] - excess)
                    radii[j] = max(0, radii[j] - excess)
                    changed = True
        if not changed:
            break
    
    # Final validity check
    for i in range(N):
        max_r = min(pos[i][0], 1 - pos[i][0], pos[i][1], 1 - pos[i][1])
        radii[i] = min(radii[i], max_r)
    
    result = np.zeros((N, 3))
    result[:, 0] = pos[:, 0]
    result[:, 1] = pos[:, 1]
    result[:, 2] = radii
    
    return result
# EVOLVE-BLOCK-END
