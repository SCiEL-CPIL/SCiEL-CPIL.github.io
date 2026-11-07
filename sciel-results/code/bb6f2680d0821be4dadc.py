"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_value(h):
    h = np.asarray(h, dtype=np.float64)
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 1.0 / (m + 1)
    # l2 = sum over i in 0..m of (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    l2 = (w / 3.0) * np.sum(y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 * norm_inf == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    # Seed
    seed_h = [1.0] * N
    seed_val = _compute_value(seed_h)
    best_h = list(seed_h)
    best_val = seed_val
    
    # Try some analytical shapes
    shapes = []
    
    # Triangular
    tri = [1.0 - abs(i - 24.5) / 25.0 for i in range(N)]
    shapes.append(tri)
    
    # Parabolic bump
    par = [1.0 - ((i - 24.5) / 25.0)**2 for i in range(N)]
    par = [max(0.0, p) for p in par]
    shapes.append(par)
    
    # Gaussian-like
    gauss = [np.exp(-((i - 24.5) / 10.0)**2) for i in range(N)]
    shapes.append(gauss)
    
    # Wider Gaussian
    gauss2 = [np.exp(-((i - 24.5) / 15.0)**2) for i in range(N)]
    shapes.append(gauss2)
    
    # Narrow Gaussian
    gauss3 = [np.exp(-((i - 24.5) / 7.0)**2) for i in range(N)]
    shapes.append(gauss3)
    
    # Uniform (seed)
    shapes.append([1.0] * N)
    
    # Ramp up then down (trapezoid)
    trap = []
    for i in range(N):
        if i < 10:
            trap.append((i+1)/10.0)
        elif i < 40:
            trap.append(1.0)
        else:
            trap.append((49-i)/10.0)
    shapes.append(trap)
    
    for s in shapes:
        v = _compute_value(s)
        if v > best_val:
            best_val = v
            best_h = list(s)
        if time.time() > deadline:
            break
    
    # Local optimization using Nelder-Mead style or simple coordinate descent
    if time.time() < deadline - 10:
        h = np.array(best_h, dtype=np.float64)
        
        # Simple coordinate descent with random perturbations
        rng = np.random.default_rng(42)
        step_size = 0.1
        for _ in range(5000):
            if time.time() > deadline:
                break
            idx = rng.integers(0, N)
            old_val = best_val
            h[idx] += rng.uniform(-step_size, step_size)
            h[idx] = max(0.0, h[idx])
            v = _compute_value(h)
            if v > best_val:
                best_val = v
                best_h = list(h)
            else:
                h[idx] = best_h[idx]
            # Adapt step size
            if v > old_val:
                step_size = min(step_size * 1.2, 1.0)
            else:
                step_size = max(step_size * 0.9, 0.01)
        
        # Multi-coordinate perturbation
        if time.time() < deadline - 5:
            h = np.array(best_h, dtype=np.float64)
            for _ in range(3000):
                if time.time() > deadline:
                    break
                # Perturb a few random coordinates
                n_perturb = rng.integers(1, 4)
                idxs = rng.choice(N, size=n_perturb, replace=False)
                old_vals = h[idxs].copy()
                h[idxs] += rng.uniform(-step_size, step_size, size=n_perturb)
                h[idxs] = np.maximum(0.0, h[idxs])
                v = _compute_value(h)
                if v > best_val:
                    best_val = v
                    best_h = list(h)
                else:
                    h[idxs] = old_vals
    
    return [max(0.0, x) for x in best_h]
# EVOLVE-BLOCK-END
