"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import sys

def _constraints_for_circle(i, x, y, r):
    """Return list of constraint dicts for circle i."""
    cons = []
    # Circle i must be inside [0,1]^2
    cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[i*3] - p[i*3+2]})  # x - r >= 0
    cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[i*3+2] - (1 - p[i*3])})  # r - (1-x) >= 0
    cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[i*3+1] - p[i*3+2]})  # y - r >= 0
    cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[i*3+2] - (1 - p[i*3+1])})  # r - (1-y) >= 0
    # Non-overlap with other circles
    for j in range(N):
        if j == i:
            continue
        cons.append({
            'type': 'ineq',
            'fun': lambda p, i=i, j=j: (p[i*3]-p[j*3])**2 + (p[i*3+1]-p[j*3+1])**2 - (p[i*3+2]+p[j*3+2])**2
        })
    return cons

def _objective(p):
    """Negative sum of radii (to minimize)."""
    return -np.sum(p[2::3])

def _build_constraints(p):
    """Build all constraints for all circles."""
    cons = []
    for i in range(N):
        # Inside unit square
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[i*3] - p[i*3+2]})
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[i*3+2] - (1 - p[i*3])})
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[i*3+1] - p[i*3+2]})
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[i*3+2] - (1 - p[i*3+1])})
    # Non-overlap
    for i in range(N):
        for j in range(i+1, N):
            cons.append({
                'type': 'ineq',
                'fun': lambda p, i=i, j=j: (p[i*3]-p[j*3])**2 + (p[i*3+1]-p[j*3+1])**2 - (p[i*3+2]+p[j*3+2])**2
            })
    return cons

def _is_valid(p, tol=1e-6):
    """Check if packing is valid."""
    for i in range(N):
        x, y, r = p[i*3], p[i*3+1], p[i*3+2]
        if r < -tol:
            return False
        if x - r < -tol or 1 - x < r + tol or y - r < -tol or 1 - y < r + tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            x1, y1, r1 = p[i*3], p[i*3+1], max(0, p[i*3+2])
            x2, y2, r2 = p[j*3], p[j*3+1], max(0, p[j*3+2])
            dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            if dist < r1 + r2 - tol:
                return False
    return True

def _grid_init(n=N):
    """Create a grid-based initial packing."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= n:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            cells.append((x, y, r))
    return np.array(cells[:n])

def _optimize_slsqp(p0, maxiter=200, tol=1e-8):
    """Run SLSQP optimization from p0."""
    bounds = [(0, 1), (0, 1), (0, 0.5)] * N
    cons = _build_constraints(p0)
    
    def obj(p):
        return -np.sum(p[2::3])
    
    result = minimize(
        obj, p0, method='SLSQP', bounds=bounds, constraints=cons,
        options={'maxiter': maxiter, 'ftol': tol}
    )
    return result

def _radius_growth_pass(p, maxiter=100, tol=1e-10):
    """Fix positions, maximize radii via SLSQP."""
    # Extract positions
    positions = p[:2*N].reshape(N, 2)
    radii = p[2::3].copy()
    
    # For each circle, try to grow its radius
    improved = True
    while improved:
        improved = False
        for i in range(N):
            # Compute max possible radius for circle i given other circles and boundaries
            max_r = min(positions[i, 0], 1 - positions[i, 0], positions[i, 1], 1 - positions[i, 1])
            for j in range(N):
                if j == i:
                    continue
                dist = np.sqrt((positions[i, 0] - positions[j, 0])**2 + (positions[i, 1] - positions[j, 1])**2)
                max_r = min(max_r, dist - radii[j])
            
            if max_r > radii[i] + 1e-10:
                radii[i] = max_r
                improved = True
    
    # Build new parameter vector
    p_new = np.zeros(3 * N)
    for i in range(N):
        p_new[i*3] = positions[i, 0]
        p_new[i*3+1] = positions[i, 1]
        p_new[i*3+2] = radii[i]
    
    return p_new

def _polish(p, maxiter=300, tol=1e-12):
    """High-precision polish pass."""
    bounds = [(0, 1), (0, 1), (0, 0.5)] * N
    cons = _build_constraints(p)
    
    def obj(p):
        return -np.sum(p[2::3])
    
    result = minimize(
        obj, p, method='SLSQP', bounds=bounds, constraints=cons,
        options={'maxiter': maxiter, 'ftol': tol}
    )
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s
    
    # Start with grid initialization
    p = _grid_init(N).flatten()
    
    # Main SLSQP optimization
    result = _optimize_slsqp(p, maxiter=150, tol=1e-9)
    if result.success or result.fun < _objective(p):
        p = result.x
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        # Not enough time for more passes
        return p.reshape(N, 3)
    
    # Radius growth pass
    p = _radius_growth_pass(p)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 20:
        return p.reshape(N, 3)
    
    # Another SLSQP optimization
    result = _optimize_slsqp(p, maxiter=150, tol=1e-10)
    if result.success or result.fun < _objective(p):
        p = result.x
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return p.reshape(N, 3)
    
    # Final high-precision polish
    result = _polish(p, maxiter=300, tol=1e-12)
    if result.success or result.fun < _objective(p):
        p = result.x
    
    # Final radius growth
    p = _radius_growth_pass(p)
    
    # Ensure validity
    if not _is_valid(p, tol=1e-5):
        # Fall back to grid
        p = _grid_init(N).flatten()
    
    return p.reshape(N, 3)
# EVOLVE-BLOCK-END
