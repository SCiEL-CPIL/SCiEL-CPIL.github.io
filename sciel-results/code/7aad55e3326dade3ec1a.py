"""Circle packing in unit square: PBD physics-based approach for 26 circles."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
EPS = 1e-12

def _valid_check(pos, r, tol=1e-7):
    n = len(r)
    for i in range(n):
        if r[i] < -tol:
            return False
        if pos[i, 0] < r[i] - tol or pos[i, 0] > 1.0 - r[i] + tol:
            return False
        if pos[i, 1] < r[i] - tol or pos[i, 1] > 1.0 - r[i] + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
            if d < r[i] + r[j] - tol:
                return False
    return True

def _delaunay_edges(pos):
    n = len(pos)
    if n < 3:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    try:
        tri = Delaunay(pos)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    edges.add((min(a, b), max(a, b)))
        return list(edges)
    except Exception:
        return [(i, j) for i in range(n) for j in range(i + 1, n)]

def _pbd_simulate(pos_init, r_init, t_start, deadline, n_iters=8000):
    """
    PBD simulation: circles as particles, outward pressure grows radii,
    Gauss-Seidel resolves distance and boundary constraints.
    """
    n = len(r_init)
    pos = pos_init.copy().astype(np.float64)
    r = r_init.copy().astype(np.float64)
    # Mass proportional to r^2 (with floor to avoid zero mass)
    mass = np.maximum(r, 0.01) ** 2
    
    # Pressure: constant force per circle pushing radius outward
    pressure = 0.002  # per-iteration radius growth target
    
    best_score = np.sum(r)
    best_pos = pos.copy()
    best_r = r.copy()
    
    stiffness = 0.5  # constraint stiffness (0..1)
    
    for it in range(n_iters):
        if time.time() > t_start + deadline:
            break
        
        # Phase 1: Apply pressure - grow radii
        r += pressure
        # Cap radii at 0.5 (can't exceed half the square)
        r = np.minimum(r, 0.5)
        # Update mass
        mass = np.maximum(r, 0.01) ** 2
        
        # Phase 2: Gauss-Seidel constraint relaxation
        n_gauss = 3  # iterations of constraint solving per pressure step
        
        for gs in range(n_gauss):
            # Boundary constraints
            for i in range(n):
                # Left wall: x >= r
                if pos[i, 0] < r[i]:
                    pos[i, 0] = r[i]
                # Right wall: x <= 1 - r
                if pos[i, 0] > 1.0 - r[i]:
                    pos[i, 0] = 1.0 - r[i]
                # Bottom wall: y >= r
                if pos[i, 1] < r[i]:
                    pos[i, 1] = r[i]
                # Top wall: y <= 1 - r
                if pos[i, 1] > 1.0 - r[i]:
                    pos[i, 1] = 1.0 - r[i]
            
            # Pairwise distance constraints (all pairs for correctness)
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pos[j, 0] - pos[i, 0]
                    dy = pos[j, 1] - pos[i, 1]
                    d = np.hypot(dx, dy)
                    min_d = r[i] + r[j]
                    if d < min_d:
                        if d < EPS:
                            # Overlapping at same point - push apart randomly
                            angle = 2.0 * np.pi * (i * 0.31 + j * 0.17)
                            dx = np.cos(angle)
                            dy = np.sin(angle)
                            d = 1.0
                        # Correct proportional to inverse mass
                        w_i = 1.0 / mass[i]
                        w_j = 1.0 / mass[j]
                        w_sum = w_i + w_j
                        correction = (min_d - d) / d * stiffness
                        pos[i, 0] -= correction * (w_i / w_sum) * dx
                        pos[i, 1] -= correction * (w_i / w_sum) * dy
                        pos[j, 0] += correction * (w_j / w_sum) * dx
                        pos[j, 1] += correction * (w_j / w_sum) * dy
            
            # Re-clamp to boundary after pairwise corrections
            for i in range(n):
                pos[i, 0] = np.clip(pos[i, 0], r[i], 1.0 - r[i])
                pos[i, 1] = np.clip(pos[i, 1], r[i], 1.0 - r[i])
        
        # Track best
        score = np.sum(r)
        if score > best_score:
            best_score = score
            best_pos = pos.copy()
            best_r = r.copy()
        
        # Adaptive: if no improvement in a while, reduce pressure
        if it % 500 == 499 and score - best_score < 1e-9:
            pressure *= 0.95
    
    return best_pos, best_r

def _slsqp_polish(pos, r, t_start, deadline):
    """SLSQP polish on the PBD result."""
    n = len(r)
    x0 = np.concatenate([pos.flatten(), r])
    
    def objective(p):
        return -np.sum(p[2*n:])
    
    edges = _delaunay_edges(pos)
    
    cons = []
    # Wall constraints
    for i in range(n):
        xi, yi, ri = 2*i, 2*i+1, 2*n+i
        def c1(p, xi=xi, ri=ri): return p[xi] - p[ri]
        def j1(p, xi=xi, ri=ri):
            v = np.zeros(3*n); v[xi]=1.0; v[ri]=-1.0; return v
        cons.append({'type': 'ineq', 'fun': c1, 'jac': j1})
        
        def c2(p, xi=xi, ri=ri): return 1.0 - p[xi] - p[ri]
        def j2(p, xi=xi, ri=ri):
            v = np.zeros(3*n); v[xi]=-1.0; v[ri]=-1.0; return v
        cons.append({'type': 'ineq', 'fun': c2, 'jac': j2})
        
        def c3(p, yi=yi, ri=ri): return p[yi] - p[ri]
        def j3(p, yi=yi, ri=ri):
            v = np.zeros(3*n); v[yi]=1.0; v[ri]=-1.0; return v
        cons.append({'type': 'ineq', 'fun': c3, 'jac': j3})
        
        def c4(p, yi=yi, ri=ri): return 1.0 - p[yi] - p[ri]
        def j4(p, yi=yi, ri=ri):
            v = np.zeros(3*n); v[yi]=-1.0; v[ri]=-1.0; return v
        cons.append({'type': 'ineq', 'fun': c4, 'jac': j4})
        
        def c5(p, ri=ri): return p[ri]
        def j5(p, ri=ri):
            v = np.zeros(3*n); v[ri]=1.0; return v
        cons.append({'type': 'ineq', 'fun': c5, 'jac': j5})
    
    # Overlap constraints for Delaunay edges
    for (i, j) in edges:
        xi, yi = 2*i, 2*i+1
        xj, yj = 2*j, 2*j+1
        ri, rj = 2*n+i, 2*n+j
        
        def c_ov(p, xi=xi, yi=yi, xj=xj, yj=yj, ri=ri, rj=rj):
            dx = p[xi] - p[xj]
            dy = p[yi] - p[yj]
            return np.hypot(dx, dy) - p[ri] - p[rj]
        
        def j_ov(p, xi=xi, yi=yi, xj=xj, yj=yj, ri=ri, rj=rj):
            dx = p[xi] - p[xj]
            dy = p[yi] - p[yj]
            d = max(np.hypot(dx, dy), EPS)
            v = np.zeros(3*n)
            v[xi] = dx/d; v[yi] = dy/d
            v[xj] = -dx/d; v[yj] = -dy/d
            v[ri] = -1.0; v[rj] = -1.0
            return v
        
        cons.append({'type': 'ineq', 'fun': c_ov, 'jac': j_ov})
    
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    
    best_pos = pos.copy()
    best_r = r.copy()
    best_score = np.sum(r)
    
    for _ in range(3):
        if time.time() > t_start + deadline:
            break
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 80, 'ftol': 1e-10})
            if res.success or res.fun < objective(x0):
                new_pos = res.x[:2*n].reshape(n, 2)
                new_r = res.x[2*n:]
                if _valid_check(new_pos, new_r):
                    if np.sum(new_r) > best_score:
                        best_score = np.sum(new_r)
                        best_pos = new_pos
                        best_r = new_r
                    x0 = np.concatenate([new_pos.flatten(), new_r])
        except Exception:
            break
    
    return best_pos, best_r

def _hex_init(rng, n):
    """Hexagonal close-packed initial arrangement."""
    pos = np.zeros((n, 2))
    spacing = 1.0 / (int(np.ceil(np.sqrt(n * 2 / np.sqrt(3)))) + 1)
    row_h = spacing * np.sqrt(3) / 2
    idx = 0
    row = 0
    while idx < n:
        col = 0
        while idx < n:
            x = (col + 0.5) * spacing + (row % 2) * spacing / 2
            y = (row + 0.5) * row_h
            if x < 1.0 and y < 1.0:
                pos[idx, 0] = x + rng.normal(0, 0.01)
                pos[idx, 1] = y + rng.normal(0, 0.01)
                pos[idx] = np.clip(pos[idx], 0.05, 0.95)
                idx += 1
            col += 1
        row += 1
    # If we didn't fill all, add remaining randomly
    if idx < n:
        for i in range(idx, n):
            pos[i] = rng.random(2) * 0.8 + 0.1
    return pos

def construct():
    t_start = time.time()
    DEADLINE = 140.0
    
    best_score = -1.0
    best_result = None
    
    num_restarts = 6
    
    for restart in range(num_restarts):
        if time.time() > t_start + DEADLINE - 15.0:
            break
        
        rng = np.random.default_rng(42 + restart * 137)
        
        if restart == 0:
            # Hexagonal init
            pos = _hex_init(rng, N)
            r = np.full(N, 0.03)
        elif restart == 1:
            # Grid init
            side = int(np.ceil(np.sqrt(N)))
            pos = np.zeros((N, 2))
            for i in range(N):
                row = i // side
                col = i % side
                pos[i, 0] = (col + 0.5) / side
                pos[i, 1] = (row + 0.5) / side
            r = np.full(N, 0.3 / side)
        elif restart == 2:
            # Hex with larger spacing
            pos = _hex_init(rng, N)
            pos += rng.normal(0, 0.03, pos.shape)
            pos = np.clip(pos, 0.05, 0.95)
            r = np.full(N, 0.02)
        else:
            # Random
            pos = rng.random((N, 2)) * 0.8 + 0.1
            r = np.full(N, 0.015)
        
        # Ensure valid start
        if not _valid_check(pos, r):
            scale = 0.5
            while not _valid_check(pos, r * scale) and scale > 0.005:
                scale *= 0.7
            r = r * scale
        
        # PBD simulation
        remaining = DEADLINE - (time.time() - t_start) - 10.0
        pos, r = _pbd_simulate(pos, r, t_start, remaining)
        
        # SLSQP polish
        remaining2 = DEADLINE - (time.time() - t_start) - 3.0
        pos, r = _slsqp_polish(pos, r, t_start, remaining2)
        
        score = np.sum(r)
        if score > best_score:
            best_score = score
            best_result = np.column_stack([pos, r])
        
        if best_score > 2.8:
            break
    
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        pos = np.zeros((N, 2))
        r = np.zeros(N)
        for i in range(N):
            row = i // side
            col = i % side
            pos[i, 0] = (col + 0.5) / side
            pos[i, 1] = (row + 0.5) / side
            r[i] = 0.4 / side
        best_result = np.column_stack([pos, r])
    
    return best_result

if __name__ == '__main__':
    out = construct()
    print("sum of radii:", out[:, 2].sum())
    print(out)
