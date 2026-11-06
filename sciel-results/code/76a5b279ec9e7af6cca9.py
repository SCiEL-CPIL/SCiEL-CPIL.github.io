"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(c):
    """Compute the radius for a set of centers."""
    # Distance to walls
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_wall, min_pair)

def _neg_score(c):
    return -_score(c)

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    best_c = None
    best_score = -np.inf
    
    def try_candidate(c):
        nonlocal best_c, best_score
        s = _score(c)
        if s > best_score:
            best_score = s
            best_c = c.copy()
    
    # Initial grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    c0 = grid[:N].astype(float)
    try_candidate(c0)
    
    # Multiple random restarts with local optimization
    np.random.seed(42)
    
    # Try different initial configurations
    configs = []
    
    # 1. Grid
    configs.append(c0.copy())
    
    # 2. Random points
    for _ in range(5):
        configs.append(np.random.uniform(-0.45, 0.45, size=(N, 2)))
    
    # 3. Hexagonal-like pattern
    for dx in [0.1, 0.15, 0.2]:
        pts = []
        row = 0
        while len(pts) < N:
            offset = (row % 2) * dx * 0.5
            x = -0.5 + dx/2 + offset
            while x < 0.5 and len(pts) < N:
                y = -0.5 + dx/2 + row * dx * np.sqrt(3)/2
                if -0.49 <= x <= 0.49 and -0.49 <= y <= 0.49:
                    pts.append([x, y])
                x += dx
            row += 1
        if len(pts) >= N:
            configs.append(np.array(pts[:N]))
    
    # 4. Concentric rings
    for nr in range(3, 8):
        pts = [[0.0, 0.0]]
        for ring in range(1, nr):
            n_in_ring = 6 * ring if ring > 1 else 6
            r = 0.1 * ring
            for i in range(n_in_ring):
                ang = 2 * np.pi * i / n_in_ring
                pts.append([r * np.cos(ang), r * np.sin(ang)])
        if len(pts) >= N:
            configs.append(np.array(pts[:N]))
    
    # 5. Jittered grid
    for jitter in [0.01, 0.03, 0.05]:
        c_jit = c0 + np.random.uniform(-jitter, jitter, size=c0.shape)
        configs.append(c_jit)
    
    for c_init in configs:
        if time.time() > deadline - 10:
            break
        
        # Local optimization using L-BFGS-B
        try:
            res = minimize(
                _neg_score,
                c_init,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.success or res.fun < 0:
                try_candidate(res.x)
        except Exception:
            pass
        
        # Also try with different tolerances
        if time.time() > deadline - 5:
            break
        try:
            res2 = minimize(
                _neg_score,
                c_init,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            try_candidate(res2.x)
        except Exception:
            pass
    
    # Final polish on best candidate
    if best_c is not None and time.time() < deadline - 2:
        try:
            res_final = minimize(
                _neg_score,
                best_c,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            try_candidate(res_final.x)
        except Exception:
            pass
    
    # Ensure validity
    if best_c is None:
        best_c = c0
    
    # Clamp to valid range
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
