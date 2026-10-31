"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Initial guess: grid layout
    side = int(np.ceil(np.sqrt(N)))
    # Create a grid of N positions
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                positions.append([
                    (i + 0.5) / side,
                    (j + 0.5) / side
                ])
    
    # Add jitter to break symmetry
    np.random.seed(42)
    positions = np.array(positions)
    positions += np.random.uniform(-0.05, 0.05, size=positions.shape)
    
    # Initial radii: small
    radii = np.full(N, 0.02)
    
    # Variables: x_0, y_0, r_0, x_1, y_1, r_1, ...
    x0 = np.concatenate([positions, radii])
    
    def objective(v):
        """Negative sum of radii (we minimize negative sum to maximize sum)"""
        if time.time() > deadline:
            return 1e10  # large penalty
        pos = v[:2*N].reshape(N, 2)
        r = v[2*N:]
        # Check bounds
        if np.any(r < 0) or np.any(r > 0.5):
            return 1e10
        if np.any(pos[:, 0] - r < -1e-9) or np.any(pos[:, 0] + r > 1 + 1e-9):
            return 1e10
        if np.any(pos[:, 1] - r < -1e-9) or np.any(pos[:, 1] + r > 1 + 1e-9):
            return 1e10
        # Check non-overlap
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt(np.sum((pos[i] - pos[j])**2))
                if dist < r[i] + r[j] - 1e-9:
                    return 1e10
        return -np.sum(r)
    
    def gradient(v):
        """Gradient of objective (negative sum of radii)"""
        pos = v[:2*N].reshape(N, 2)
        r = v[2*N:]
        grad = np.zeros_like(v)
        # Gradient of -sum(r) with respect to r is -1
        grad[2*N:] = -1.0
        return grad
    
    # Bounds: x in [0,1], y in [0,1], r in [0, 0.5]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((0.0, 0.5))  # r
    
    # Constraints: 
    # 1. Circle i stays in unit square: x_i - r_i >= 0, x_i + r_i <= 1, etc.
    # 2. Non-overlap: dist(i,j) >= r_i + r_j
    
    def make_boundary_constraint(i, coord):
        """coord 0 for x, 1 for y"""
        def constraint(v):
            pos = v[:2*N].reshape(N, 2)
            r = v[2*N:]
            if coord == 0:
                return pos[i, 0] - r[i]
            else:
                return pos[i, 1] - r[i]
        def constraint2(v):
            pos = v[:2*N].reshape(N, 2)
            r = v[2*N:]
            if coord == 0:
                return 1.0 - pos[i, 0] - r[i]
            else:
                return 1.0 - pos[i, 1] - r[i]
        return constraint, constraint2
    
    def make_overlap_constraint(i, j):
        def constraint(v):
            pos = v[:2*N].reshape(N, 2)
            r = v[2*N:]
            dist = np.sqrt(np.sum((pos[i] - pos[j])**2))
            return dist - r[i] - r[j]
        return constraint
    
    constraints = []
    for i in range(N):
        c1, c2 = make_boundary_constraint(i, 0)
        c3, c4 = make_boundary_constraint(i, 1)
        constraints.append({'type': 'ineq', 'fun': c1})
        constraints.append({'type': 'ineq', 'fun': c2})
        constraints.append({'type': 'ineq', 'fun': c3})
        constraints.append({'type': 'ineq', 'fun': c4})
    
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({'type': 'ineq', 'fun': make_overlap_constraint(i, j)})
    
    # Run SLSQP
    best_result = None
    best_val = -np.sum(radii)
    
    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        if time.time() < deadline:
            if res.success or res.fun < best_val:
                if res.fun < 1e9:  # valid
                    pos = res.x[:2*N].reshape(N, 2)
                    r = res.x[2*N:]
                    # Validate
                    valid = True
                    if np.any(r < -1e-9) or np.any(pos[:, 0] - r < -1e-9) or np.any(pos[:, 0] + r > 1 + 1e-9) or \
                       np.any(pos[:, 1] - r < -1e-9) or np.any(pos[:, 1] + r > 1 + 1e-9):
                        valid = False
                    for i in range(N):
                        for j in range(i+1, N):
                            dist = np.sqrt(np.sum((pos[i] - pos[j])**2))
                            if dist < r[i] + r[j] - 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        best_result = np.column_stack([pos, r])
                        best_val = res.fun
    except Exception:
        pass
    
    if time.time() < deadline - 5.0 and best_result is not None:
        # Try to polish with more iterations
        try:
            res2 = minimize(
                objective,
                best_result.ravel(),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-14}
            )
            if time.time() < deadline:
                if res2.fun < best_val and res2.fun < 1e9:
                    pos = res2.x[:2*N].reshape(N, 2)
                    r = res2.x[2*N:]
                    valid = True
                    if np.any(r < -1e-9) or np.any(pos[:, 0] - r < -1e-9) or np.any(pos[:, 0] + r > 1 + 1e-9) or \
                       np.any(pos[:, 1] - r < -1e-9) or np.any(pos[:, 1] + r > 1 + 1e-9):
                        valid = False
                    for i in range(N):
                        for j in range(i+1, N):
                            dist = np.sqrt(np.sum((pos[i] - pos[j])**2))
                            if dist < r[i] + r[j] - 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        best_result = np.column_stack([pos, r])
                        best_val = res2.fun
        except Exception:
            pass
    
    if best_result is not None:
        # Small shrink to ensure validity
        r = best_result[:, 2]
        r = np.maximum(r - 1e-10, 0)
        best_result[:, 2] = r
        return best_result
    else:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
# EVOLVE-BLOCK-END
