"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

DEADLINE = 145.0  # safety margin before 150s kill

def construct():
    start_time = time.time()
    
    # Initial grid placement
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side) for i in range(side) for j in range(side)]
    centers = np.array(cells[:N])
    
    # Compute initial radii based on grid spacing
    r0 = 0.5 / side - 1e-3
    radii = np.full(N, r0)
    x0 = np.concatenate([centers, radii])
    
    def objective(x):
        centers = x[:2*N].reshape(N, 2)
        radii = x[2*N:]
        return -np.sum(radii)
    
    def objective_grad(x):
        centers = x[:2*N].reshape(N, 2)
        radii = x[2*N:]
        grad = np.zeros_like(x)
        grad[2*N:] = -1.0
        return grad
    
    # Constraints: circles must stay inside unit square and not overlap
    def make_constraints():
        cons = []
        # Boundary constraints: x_i >= r_i, x_i + r_i <= 1, y_i >= r_i, y_i + r_i <= 1
        # For each circle i:
        #   x_i - r_i >= 0
        #   1 - x_i - r_i >= 0
        #   y_i - r_i >= 0
        #   1 - y_i - r_i >= 0
        for i in range(N):
            xi, yi, ri = 2*i, 2*i+1, 2*N+i
            cons.append({'type': 'ineq', 'fun': lambda x, xi=xi, ri=ri: x[xi] - x[ri],
                         'jac': lambda x, xi=xi, ri=ri: np.concatenate([np.eye(1)[0,xi], np.eye(1)[0,ri]*(-1)])})
            cons.append({'type': 'ineq', 'fun': lambda x, xi=xi, ri=ri: 1 - x[xi] - x[ri],
                         'jac': lambda x, xi=xi, ri=ri: np.concatenate([np.eye(1)[0,xi]*(-1), np.eye(1)[0,ri]*(-1)])})
            cons.append({'type': 'ineq', 'fun': lambda x, yi=yi, ri=ri: x[yi] - x[ri],
                         'jac': lambda x, yi=yi, ri=ri: np.concatenate([np.eye(1)[0,yi], np.eye(1)[0,ri]*(-1)])})
            cons.append({'type': 'ineq', 'fun': lambda x, yi=yi, ri=ri: 1 - x[yi] - x[ri],
                         'jac': lambda x, yi=yi, ri=ri: np.concatenate([np.eye(1)[0,yi]*(-1), np.eye(1)[0,ri]*(-1)])})
        # Overlap constraints: dist(i,j) >= r_i + r_j
        for i in range(N):
            for j in range(i+1, N):
                cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: np.sum((x[2*i:2*i+2] - x[2*j:2*j+2])**2) - (x[2*N+i] + x[2*N+j])**2,
                             'jac': lambda x, i=i, j=j: _overlap_jac(x, i, j)})
        return cons
    
    def _overlap_jac(x, i, j):
        grad = np.zeros_like(x)
        ci = x[2*i:2*i+2]
        cj = x[2*j:2*j+2]
        ri = x[2*N+i]
        rj = x[2*N+j]
        d = ci - cj
        dist_sq = np.sum(d**2)
        if dist_sq < 1e-12:
            dist_sq = 1e-12
        dist = np.sqrt(dist_sq)
        # d/dx of (|ci-cj|^2 - (ri+rj)^2)
        grad[2*i] = 2 * d[0]
        grad[2*i+1] = 2 * d[1]
        grad[2*j] = -2 * d[0]
        grad[2*j+1] = -2 * d[1]
        grad[2*N+i] = -2 * (ri + rj)
        grad[2*N+j] = -2 * (ri + rj)
        return grad
    
    # Bounds: centers in [0,1], radii >= 0
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # x_i
        bounds.append((0.0, 1.0))  # y_i
    for i in range(N):
        bounds.append((1e-6, 0.5))  # r_i
    
    constraints = make_constraints()
    
    # Phase 1: L-BFGS-B with soft penalty (no hard constraints) for fast exploration
    def soft_penalty(x, beta=50.0):
        centers = x[:2*N].reshape(N, 2)
        radii = x[2*N:]
        pen = 0.0
        # Boundary violations
        for i in range(N):
            xi, yi, ri = centers[i, 0], centers[i, 1], radii[i]
            pen += max(0, ri - xi) ** 2
            pen += max(0, ri - yi) ** 2
            pen += max(0, xi + ri - 1) ** 2
            pen += max(0, yi + ri - 1) ** 2
        # Overlap violations
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                overlap = radii[i] + radii[j] - d
                if overlap > 0:
                    pen += overlap ** 2
        return beta * pen
    
    def soft_objective(x):
        return objective(x) + soft_penalty(x)
    
    # Run L-BFGS-B for phase 1
    time1 = time.time()
    if time1 - start_time < DEADLINE - 30:
        res1 = minimize(soft_objective, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6})
        if res1.success or res1.fun < objective(x0):
            x_best = res1.x
        else:
            x_best = x0
    else:
        x_best = x0
    
    # Phase 2: SLSQP with hard constraints for precise optimization
    time2 = time.time()
    if time2 - start_time < DEADLINE - 10:
        res2 = minimize(objective, x_best, method='SLSQP', bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': 300, 'ftol': 1e-10, 'disp': False})
        if res2.success or res2.fun < objective(x_best):
            x_best = res2.x
    
    # Phase 3: High-precision polish pass
    time3 = time.time()
    if time3 - start_time < DEADLINE - 2:
        # Tighter bounds for polish
        polish_bounds = []
        for i in range(N):
            xi = x_best[2*i]
            yi = x_best[2*i+1]
            ri = x_best[2*N+i]
            polish_bounds.append((max(0, ri - 1e-9), min(1, 1 - ri + 1e-9)))
            polish_bounds.append((max(0, ri - 1e-9), min(1, 1 - ri + 1e-9)))
        for i in range(N):
            ri = x_best[2*N+i]
            polish_bounds.append((max(1e-6, ri * 0.999), min(0.5, ri * 1.001)))
        
        res3 = minimize(objective, x_best, method='SLSQP', bounds=polish_bounds,
                       constraints=constraints,
                       options={'maxiter': 200, 'ftol': 1e-12, 'disp': False})
        if res3.success or res3.fun < objective(x_best):
            x_best = res3.x
    
    # Extract final result
    centers = x_best[:2*N].reshape(N, 2)
    radii = x_best[2*N:]
    
    # Clamp to valid range
    for i in range(N):
        radii[i] = max(1e-6, min(radii[i], 0.5))
        centers[i, 0] = max(radii[i], min(centers[i, 0], 1 - radii[i]))
        centers[i, 1] = max(radii[i], min(centers[i, 1], 1 - radii[i]))
    
    result = np.zeros((N, 3))
    result[:, 0] = centers[:, 0]
    result[:, 1] = centers[:, 1]
    result[:, 2] = radii
    
    # Final validation check
    for i in range(N):
        x, y, r = result[i]
        if x < r - 1e-9 or x + r > 1 + 1e-9 or y < r - 1e-9 or y + r > 1 + 1e-9:
            # Fix boundary violations
            r = min(r, x, 1 - x, y, 1 - y)
            result[i] = [x, y, max(1e-6, r)]
    
    return result
# EVOLVE-BLOCK-END
