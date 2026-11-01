"""Circle packing in perimeter-4 rectangle: maximize sum of radii for 21 circles."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
DEADLINE = 120.0  # seconds wall-clock limit


def _check_validity(arr, tol=1e-9):
    """Check if circles are valid: disjoint interiors, inside bounding box, perimeter <= 4."""
    n = arr.shape[0]
    xs, ys, rs = arr[:, 0], arr[:, 1], arr[:, 2]
    
    if np.any(rs <= 0):
        return False
    
    x_ext_min = np.min(xs - rs)
    x_ext_max = np.max(xs + rs)
    y_ext_min = np.min(ys - rs)
    y_ext_max = np.max(ys + rs)
    
    perim = 2.0 * ((x_ext_max - x_ext_min) + (y_ext_max - y_ext_min))
    if perim > 4.0 + tol:
        return False
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - tol:
                return False
    
    return True


def _vectorized_objective(z, w, h):
    """
    Vectorized objective: maximize sum of radii subject to constraints.
    z = [x0, y0, r0, x1, y1, r1, ..., x20, y20, r20]
    Returns negative sum of radii (to minimize), with large penalty for violations.
    """
    xs = z[0::3]
    ys = z[1::3]
    rs = z[2::3]
    
    # Sum of radii (we want to maximize)
    sum_r = np.sum(rs)
    
    # Penalty for violations
    penalty = 0.0
    
    # Boundary constraints: x >= r, x <= w-r, y >= r, y <= h-r
    bound_viol = np.maximum(0, rs - xs) + np.maximum(0, rs - (w - xs))
    bound_viol += np.maximum(0, rs - ys) + np.maximum(0, rs - (h - ys))
    penalty += 100.0 * np.sum(bound_viol)
    
    # Pairwise constraints: dist >= ri + rj
    for i in range(N):
        dx = xs[i] - xs[i+1:]
        dy = ys[i] - ys[i+1:]
        dists = np.sqrt(dx*dx + dy*dy)
        violations = np.maximum(0, rs[i] + rs[i+1:] - dists)
        penalty += 100.0 * np.sum(violations)
    
    # Negative radius penalty
    penalty += 1000.0 * np.sum(np.maximum(0, -rs))
    
    return -(sum_r - penalty)


def _differential_evolution(w, h, pop_size=40, max_iter=200, F=0.8, CR=0.9, time_limit=30.0):
    """
    Differential Evolution for circle packing.
    Variables: [x0, y0, r0, x1, y1, r1, ...]
    """
    start = time.time()
    dim = 3 * N
    
    # Bounds
    lb = np.zeros(dim)
    ub = np.zeros(dim)
    for i in range(N):
        lb[3*i] = 0.0
        ub[3*i] = w
        lb[3*i+1] = 0.0
        ub[3*i+1] = h
        lb[3*i+2] = 0.01
        ub[3*i+2] = min(w, h) / 2.0
    
    # Initialize population
    np.random.seed(42)
    pop = np.random.uniform(lb, ub, size=(pop_size, dim))
    
    # Evaluate initial population
    fitness = np.array([_vectorized_objective(pop[i], w, h) for i in range(pop_size)])
    
    best_idx = np.argmin(fitness)
    best_z = pop[best_idx].copy()
    best_fit = fitness[best_idx]
    
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        
        new_pop = pop.copy()
        new_fitness = fitness.copy()
        
        for i in range(pop_size):
            # Select three distinct random individuals
            a, b, c = np.random.choice(pop_size, size=3, replace=False)
            while a == b or b == c or a == c:
                a, b, c = np.random.choice(pop_size, size=3, replace=False)
            
            # Mutation
            mutant = pop[a] + F * (pop[b] - pop[c])
            
            # Crossover
            trial = np.zeros(dim)
            j_rand = np.random.randint(0, dim)
            for j in range(dim):
                if np.random.rand() < CR or j == j_rand:
                    trial[j] = mutant[j]
                else:
                    trial[j] = pop[i][j]
            
            # Clamp to bounds
            trial = np.clip(trial, lb, ub)
            
            # Evaluate
            trial_fit = _vectorized_objective(trial, w, h)
            
            # Selection
            if trial_fit <= fitness[i]:
                new_pop[i] = trial
                new_fitness[i] = trial_fit
                if trial_fit < best_fit:
                    best_fit = trial_fit
                    best_z = trial.copy()
        
        pop = new_pop
        fitness = new_fitness
    
    return best_z, -best_fit


def _refine_with_scipy(z, w, h, time_limit=10.0):
    """Refine a solution using SLSQP."""
    start = time.time()
    dim = 3 * N
    
    lb = np.zeros(dim)
    ub = np.zeros(dim)
    for i in range(N):
        lb[3*i] = 0.0
        ub[3*i] = w
        lb[3*i+1] = 0.0
        ub[3*i+1] = h
        lb[3*i+2] = 0.01
        ub[3*i+2] = min(w, h) / 2.0
    
    # Constraints
    constraints = []
    
    # Boundary constraints
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda z, i=i: z[3*i] - z[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda z, i=i: w - z[3*i] - z[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda z, i=i: z[3*i+1] - z[3*i+2]})
        constraints.append({'type': 'ineq', 'fun': lambda z, i=i: h - z[3*i+1] - z[3*i+2]})
    
    # Pairwise constraints
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({
                'type': 'ineq',
                'fun': lambda z, i=i, j=j: np.sqrt((z[3*i]-z[3*j])**2 + (z[3*i+1]-z[3*j+1])**2) - z[3*i+2] - z[3*j+2]
            })
    
    try:
        result = minimize(
            lambda z: _vectorized_objective(z, w, h),
            z,
            method='SLSQP',
            bounds=list(zip(lb, ub)),
            constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-12}
        )
        
        if time.time() - start < time_limit and result.success:
            return result.x, -_vectorized_objective(result.x, w, h)
    except Exception:
        pass
    
    return z, -_vectorized_objective(z, w, h)


def _extract_circles(z, w, h):
    """Extract circle positions and radii from parameter vector."""
    xs = z[0::3]
    ys = z[1::3]
    rs = z[2::3]
    
    arr = np.zeros((N, 3))
    arr[:, 0] = xs
    arr[:, 1] = ys
    arr[:, 2] = rs
    
    # Clip to valid range
    for i in range(N):
        arr[i, 0] = np.clip(arr[i, 0], arr[i, 2], w - arr[i, 2])
        arr[i, 1] = np.clip(arr[i, 1], arr[i, 2], h - arr[i, 2])
    
    return arr


def _is_valid(arr, w, h, tol=1e-9):
    """Fast validity check for circles in a fixed rectangle [0,w] x [0,h]."""
    xs, ys, rs = arr[:, 0], arr[:, 1], arr[:, 2]
    
    if np.any(rs <= 0):
        return False
    
    if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
        return False
    if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
        return False
    
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - tol:
                return False
    
    return True


def construct():
    start_time = time.time()
    
    best_arr = None
    best_sum = 0.0
    
    # Try different aspect ratios
    aspects = [1.0, 1.2, 0.8, 1.5, 0.67, 2.0, 0.5, 1.33, 0.75, 1.1, 0.9]
    
    for aspect in aspects:
        if time.time() - start_time > DEADLINE - 15.0:
            break
        
        h = 2.0 / (aspect + 1.0)
        w = aspect * h
        
        # Run differential evolution
        de_time = 20.0
        z, obj_val = _differential_evolution(w, h, pop_size=30, max_iter=150, time_limit=de_time)
        
        # Extract circles
        arr = _extract_circles(z, w, h)
        
        # Check validity
        if not _is_valid(arr, w, h):
            # Try to fix by shrinking radii
            for scale in [0.95, 0.9, 0.85, 0.8]:
                arr_fixed = arr.copy()
                arr_fixed[:, 2] *= scale
                if _is_valid(arr_fixed, w, h):
                    arr = arr_fixed
                    break
            else:
                continue
        
        # Refine with SLSQP
        if time.time() - start_time < DEADLINE - 10.0:
            z_refined, obj_refined = _refine_with_scipy(z, w, h, time_limit=8.0)
            arr_refined = _extract_circles(z_refined, w, h)
            if _is_valid(arr_refined, w, h):
                s_refined = np.sum(arr_refined[:, 2])
                s_current = np.sum(arr[:, 2])
                if s_refined > s_current:
                    arr = arr_refined
        
        s = np.sum(arr[:, 2])
        if s > best_sum:
            best_sum = s
            best_arr = arr.copy()
    
    # Final fallback
    if best_arr is None or not _check_validity(best_arr):
        w = h = 1.0
        side = 5
        best_arr = np.zeros((N, 3))
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                best_arr[count, 0] = (i + 0.5) / side
                best_arr[count, 1] = (j + 0.5) / side
                best_arr[count, 2] = 0.5 / side - 0.01
                count += 1
    
    return best_arr
