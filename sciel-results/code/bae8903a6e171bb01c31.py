import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin for 150s limit
    
    # Strategy: 
    # 1. Define a "core" set of variables that determine the packing topology.
    #    We assume a hexagonal-like structure. 
    #    Let's define the first few circles explicitly and derive others via contacts.
    #    However, a fully rigid graph might be too brittle. 
    #    Instead, we use a "soft" contact graph where we optimize:
    #    - Radii of all 26 circles (26 vars)
    #    - Positions of a subset of "anchor" circles (e.g., 4-6 circles) 
    #    - The rest are placed by solving a small local optimization to satisfy 
    #      contact constraints with anchors and neighbors, or we just optimize 
    #      all positions but with a much better initialization and a 
    #      "rigid body" penalty that encourages specific distances.
    
    # Let's try a different structural reduction:
    # Assume the packing is composed of a central cluster and boundary circles.
    # Actually, the most robust "different" approach is to use 
    # Differential Evolution (DE) on the full 78D space but with a 
    # highly structured initial population and a tighter penalty.
    # But the directive asks for a "different design or modelling family".
    
    # New Family: "Contact Graph" Optimization
    # We define a graph G=(V,E) of contacts. 
    # We optimize the radii r_i and the positions of a subset of nodes (anchors).
    # The positions of non-anchor nodes are determined by intersecting circles 
    # centered at their anchor neighbors with radii r_i + r_j.
    # This is complex to implement robustly for arbitrary graphs.
    
    # Alternative Structural Change: 
    # Use a "Voronoi-like" or "Apollonian" packing generation? No.
    
    # Let's implement a "Rigid Cluster" approach.
    # We partition the 26 circles into 3-4 rigid clusters.
    # Each cluster has internal fixed relative positions (scaled by a global or local scale factor).
    # We optimize:
    # - Global scale factors for each cluster
    # - Translation and rotation of each cluster
    # This reduces 78 vars to ~ 3 clusters * (1 scale + 2 trans + 1 rot) + 26 radii? 
    # If internal geometry is fixed, radii are fixed relative to scale.
    # So, for a cluster of k circles with fixed shape, we have 1 scale, 2 trans, 1 rot.
    # If we assume all circles in a cluster have proportional radii, we reduce variables significantly.
    
    # Let's try: 26 circles. 
    # Assume a hexagonal lattice is the best topology.
    # We can define the positions as a linear transformation of a standard hex lattice.
    # x = A * x_hex + t
    # r = s * r_hex
    # Variables: A (2x2 matrix, 4 vars), t (2 vars), s (1 var). Total 7 vars.
    # But this assumes a perfect hex lattice, which might not be optimal for 26 in a square.
    # We can allow "defects" by adding small perturbation variables for each circle.
    # Variables: 7 global + 26*2 perturbations = 59 vars. Still less than 78.
    
    # Even better: 
    # Optimize 26 radii + 26 x + 26 y using Differential Evolution.
    # DE is global. The current program uses L-BFGS-B (local).
    # The "different strategy" is switching from Local Gradient Descent to Global Stochastic Search (DE).
    
    def objective(v):
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        
        # Penalty for out of bounds
        p_bound = 0.0
        p_bound += np.sum(np.maximum(r - x, 0)**2)
        p_bound += np.sum(np.maximum(r - (1.0 - x), 0)**2)
        p_bound += np.sum(np.maximum(r - y, 0)**2)
        p_bound += np.sum(np.maximum(r - (1.0 - y), 0)**2)
        
        # Penalty for overlap
        p_overlap = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                viol = r[i] + r[j] - d
                if viol > 0:
                    p_overlap += viol**2
        
        # Maximize sum of radii -> Minimize -sum(r) + penalty
        return -np.sum(r) + 100.0 * (p_bound + p_overlap)
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0)) # x
    for i in range(N):
        bounds.append((0.0, 1.0)) # y
    for i in range(N):
        bounds.append((0.0, 0.5)) # r
        
    # Initial population strategy for DE:
    # DE requires bounds. It generates random points.
    # To help it, we can use a seed or just let it run.
    # However, DE is slow in high dimensions. 78D is borderline.
    # Let's try to reduce dimensionality further.
    
    # Refined Strategy: 
    # Assume all circles have the same radius? No, that's suboptimal.
    # Assume a hexagonal lattice with variable row/col spacing?
    
    # Let's stick to the "Global Optimizer on Full Space" but with a trick:
    # Use `differential_evolution` with `popsize=15` (default) and `maxiter=100`.
    # This might be too slow.
    
    # Alternative: Use a "Multi-start" L-BFGS-B but with a *different* initialization 
    # that is structurally diverse, e.g., using a "Jittered Grid" and "Random Hex" and "Ring".
    # The current program uses Hex. 
    # Let's use a "Random" initialization with high temperature annealing?
    
    # Actually, the most promising "different" approach that is fast:
    # **Simulated Annealing** or **Particle Swarm Optimization** on the 78D space.
    # But implementing SA/PSO from scratch is risky.
    
    # Let's go with **Differential Evolution** but optimized for speed.
    # We will run DE for a short time, then take the best result and polish with L-BFGS-B.
    
    best_result = None
    best_score = np.inf
    
    # Phase 1: Global Search with Differential Evolution
    # DE is robust to local minima.
    # We limit the iterations to keep it fast.
    
    # To speed up DE, we can reduce the population size or iterations.
    # popsize=10, maxiter=50.
    
    try:
        de_result = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=10, 
            tol=1e-4,
            update_interval=1,
            polish=False # We will polish manually
        )
        if de_result.fun < best_score:
            best_score = de_result.fun
            best_result = de_result.x
    except Exception:
        pass
    
    # Phase 2: Local Polish with L-BFGS-B on the best result from DE
    if best_result is not None:
        v_start = best_result
    else:
        # Fallback to hex init if DE fails
        r0 = 0.08
        dx = 2 * r0
        dy = r0 * np.sqrt(3)
        positions = []
        for j in range(5):
            for i in range(6):
                x = (i + 0.5 * (j % 2)) * dx
                y = j * dy
                positions.append([x, y])
        pos = np.array(positions[:N])
        pos[:, 0] -= pos[:, 0].mean()
        pos[:, 1] -= pos[:, 1].mean()
        x_max = pos[:, 0].max(); x_min = pos[:, 0].min()
        y_max = pos[:, 1].max(); y_min = pos[:, 1].min()
        scale = min(0.8 / (x_max - x_min), 0.8 / (y_max - y_min))
        pos *= scale
        pos[:, 0] += 0.5
        pos[:, 1] += 0.5
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist: min_dist = d
        r_init = min_dist * 0.45
        radii = np.full(N, r_init)
        v_start = np.concatenate([pos[:, 0], pos[:, 1], radii])
        
    # Run L-BFGS-B to polish
    def softplus(x, beta=20):
        z = beta * x
        return (np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z)))) / beta

    def objective_polish(v):
        x = v[:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        
        p_bound = 0.0
        p_bound += np.sum(softplus(r - x))
        p_bound += np.sum(softplus(r - (1.0 - x)))
        p_bound += np.sum(softplus(r - y))
        p_bound += np.sum(softplus(r - (1.0 - y)))
        
        p_overlap = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                viol = r[i] + r[j] - d
                p_overlap += softplus(viol)**2
        
        return -np.sum(r) + 50.0 * (p_bound + p_overlap)

    try:
        res = minimize(objective_polish, v_start, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8})
        if res.fun < best_score:
            best_score = res.fun
            best_result = res.x
    except Exception:
        pass
    
    # Phase 3: Multi-start Polish from diverse initializations
    # Use the remaining time to try 2-3 more local starts
    starts = [
        "hex", 
        "grid", 
        "random"
    ]
    
    for s_name in starts:
        if time.time() > deadline - 10:
            break
            
        if s_name == "hex":
            # Re-use hex init logic
            r0 = 0.08
            dx = 2 * r0
            dy = r0 * np.sqrt(3)
            positions = []
            for j in range(5):
                for i in range(6):
                    x = (i + 0.5 * (j % 2)) * dx
                    y = j * dy
                    positions.append([x, y])
            pos = np.array(positions[:N])
            pos[:, 0] -= pos[:, 0].mean()
            pos[:, 1] -= pos[:, 1].mean()
            x_max = pos[:, 0].max(); x_min = pos[:, 0].min()
            y_max = pos[:, 1].max(); y_min = pos[:, 1].min()
            scale = min(0.8 / (x_max - x_min), 0.8 / (y_max - y_min))
            pos *= scale
            pos[:, 0] += 0.5
            pos[:, 1] += 0.5
            min_dist = np.inf
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(pos[i] - pos[j])
                    if d < min_dist: min_dist = d
            r_init = min_dist * 0.45
            radii = np.full(N, r_init)
            v_start = np.concatenate([pos[:, 0], pos[:, 1], radii])
        elif s_name == "grid":
            side = int(np.ceil(np.sqrt(N)))
            positions = []
            for i in range(side):
                for j in range(side):
                    positions.append([(i+0.5)/side, (j+0.5)/side])
            pos = np.array(positions[:N])
            min_dist = np.inf
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(pos[i] - pos[j])
                    if d < min_dist: min_dist = d
            r_init = min_dist * 0.45
            radii = np.full(N, r_init)
            v_start = np.concatenate([pos[:, 0], pos[:, 1], radii])
        else: # random
            np.random.seed(123)
            x = np.random.rand(N) * 0.6 + 0.2
            y = np.random.rand(N) * 0.6 + 0.2
            r = np.full(N, 0.05)
            v_start = np.concatenate([x, y, r])
            
        try:
            res = minimize(objective_polish, v_start, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
        except Exception:
            pass
            
    # Final Validation and Repair
    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    x = best_result[:N]
    y = best_result[N:2*N]
    r = best_result[2*N:3*N]
    
    def is_valid(x, y, r):
        for i in range(N):
            if r[i] > x[i] + 1e-9 or r[i] > 1 - x[i] + 1e-9 or r[i] > y[i] + 1e-9 or r[i] > 1 - y[i] + 1e-9:
                return False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if r[i] + r[j] > d + 1e-9:
                    return False
        return True
        
    if not is_valid(x, y, r):
        lo, hi = 0.0, 1.0
        for _ in range(50):
            mid = (lo + hi) / 2
            r_test = r * mid
            if is_valid(x, y, r_test):
                lo = mid
            else:
                hi = mid
        r = r * lo
        r = np.maximum(r, 0)
        
    if not is_valid(x, y, r):
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return np.column_stack((x, y, r))
