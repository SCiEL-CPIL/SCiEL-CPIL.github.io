import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    def softplus(z, eps):
        return np.logaddexp(0, z / eps) * eps

    def get_initial_layout_hex():
        # Generate a hex grid of points
        r0 = 0.12
        centers = []
        for row in range(10):
            for col in range(10):
                x = col * 2 * r0
                y = row * np.sqrt(3) * r0
                if row % 2 == 1:
                    x += r0
                centers.append((x, y))
        
        # Sort by distance from center (0.5, 0.5) to get a compact cluster
        centers.sort(key=lambda p: (p[0]-0.5)**2 + (p[1]-0.5)**2)
        centers = centers[:N]
        
        # Calculate bounding box to scale and center
        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max_x - min_x
        height = max_y - min_y
        
        # Scale to fit in [0.1, 0.9] to allow room for optimization
        scale = min(0.8 / width, 0.8 / height)
        
        scaled_centers = []
        for x, y in centers:
            sx = (x - min_x) * scale + 0.1
            sy = (y - min_y) * scale + 0.1
            scaled_centers.append((sx, sy))
            
        # Initial radii
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((scaled_centers[i][0]-scaled_centers[j][0])**2 + (scaled_centers[i][1]-scaled_centers[j][1])**2)
                if d < min_dist:
                    min_dist = d
        min_boundary = np.inf
        for x, y in scaled_centers:
            min_boundary = min(min_boundary, x, 1-x, y, 1-y)
            
        r_init = min(min_dist / 2, min_boundary) * 0.95
        
        v = []
        for x, y in scaled_centers:
            v.extend([x, y, r_init])
        return np.array(v)

    def get_initial_layout_corner():
        # Corner-weighted initial layout
        positions = []
        corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92],
                   [0.5, 0.08], [0.5, 0.92], [0.08, 0.5], [0.92, 0.5]]
        for cx, cy in corners:
            positions.append([cx, cy, 0.08])
        rng = np.random.default_rng(42)
        for _ in range(N - len(positions)):
            x = rng.uniform(0.05, 0.95)
            y = rng.uniform(0.05, 0.95)
            positions.append([x, y, rng.uniform(0.06, 0.10)])
        
        v = []
        for x, y, r in positions[:N]:
            v.extend([x, y, r])
        return np.array(v)

    def get_initial_layout_grid():
        side = int(np.ceil(np.sqrt(N)))
        positions = []
        r = 0.5/side * 0.9
        for i in range(side):
            for j in range(side):
                x = (i+0.5)/side
                y = (j+0.5)/side
                positions.append([x, y, r])
                if len(positions) >= N:
                    break
            if len(positions) >= N:
                break
        while len(positions) < N:
            positions.append([0.5, 0.5, r*0.5])
        
        v = []
        for x, y, r in positions[:N]:
            v.extend([x, y, r])
        return np.array(v)

    def get_contact_edges(v, tol=1e-4):
        """Get contact edges using Delaunay triangulation."""
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        points = np.column_stack([x, y])
        
        try:
            tri = Delaunay(points)
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i+1, 3):
                        a, b = simplex[i], simplex[j]
                        edges.add((min(a,b), max(a,b)))
            
            # Filter edges to only those where circles are close (potential contacts)
            contact_edges = []
            for i, j in edges:
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] + tol:
                    contact_edges.append((i, j))
            
            return contact_edges
        except Exception:
            return []

    def objective(v, penalty=1000.0, eps=1e-4, contact_edges=None):
        x = v[0::3]
        y = v[1::3]
        r = np.maximum(v[2::3], 1e-6)
        
        score = -np.sum(r)
        
        # Boundary violations
        lx = r - x
        ux = x + r - 1.0
        ly = r - y
        uy = y + r - 1.0
        wall_pen = np.sum(softplus(lx, eps)**2) + np.sum(softplus(ux, eps)**2) + \
                   np.sum(softplus(ly, eps)**2) + np.sum(softplus(uy, eps)**2)
        
        # Overlap violations - use contact edges if available for speed
        if contact_edges is not None and len(contact_edges) > 0:
            iu0 = np.array([e[0] for e in contact_edges])
            iu1 = np.array([e[1] for e in contact_edges])
        else:
            iu0, iu1 = np.triu_indices(N, k=1)
        
        dx = x[iu0] - x[iu1]
        dy = y[iu0] - y[iu1]
        dist = np.sqrt(dx*dx + dy*dy + 1e-12)
        overlap = (r[iu0] + r[iu1]) - dist
        pair_pen = np.sum(softplus(overlap, eps)**2)
        
        return score + penalty * (wall_pen + pair_pen)

    def gradient(v, penalty=1000.0, eps=1e-4, contact_edges=None):
        x = v[0::3]
        y = v[1::3]
        r = np.maximum(v[2::3], 1e-6)
        
        grad = np.zeros_like(v)
        
        # Gradient of -sum(r)
        grad[2::3] -= 1.0
        
        def d_softplus_sq(z, eps):
            sp = softplus(z, eps)
            sig = 1.0 / (1.0 + np.exp(-z / eps))
            return 2.0 * sp * sig

        # Boundary gradients
        for i in range(N):
            z = r[i] - x[i]
            dz = d_softplus_sq(z, eps)
            grad[i*3] += -penalty * dz
            grad[i*3+2] += penalty * dz
            
            z = x[i] + r[i] - 1.0
            dz = d_softplus_sq(z, eps)
            grad[i*3] += penalty * dz
            grad[i*3+2] += penalty * dz
            
            z = r[i] - y[i]
            dz = d_softplus_sq(z, eps)
            grad[i*3+1] += -penalty * dz
            grad[i*3+2] += penalty * dz
            
            z = y[i] + r[i] - 1.0
            dz = d_softplus_sq(z, eps)
            grad[i*3+1] += penalty * dz
            grad[i*3+2] += penalty * dz

        # Pairwise gradients
        if contact_edges is not None and len(contact_edges) > 0:
            iu0 = np.array([e[0] for e in contact_edges])
            iu1 = np.array([e[1] for e in contact_edges])
        else:
            iu0, iu1 = np.triu_indices(N, k=1)
        
        dx = x[iu0] - x[iu1]
        dy = y[iu0] - y[iu1]
        dist = np.sqrt(dx*dx + dy*dy + 1e-12)
        overlap = (r[iu0] + r[iu1]) - dist
        
        dz = d_softplus_sq(overlap, eps)
        ux = dx / dist
        uy = dy / dist
        
        np.add.at(grad[0::3], iu0, -penalty * dz * ux)
        np.add.at(grad[0::3], iu1, penalty * dz * ux)
        np.add.at(grad[1::3], iu0, -penalty * dz * uy)
        np.add.at(grad[1::3], iu1, penalty * dz * uy)
        np.add.at(grad[2::3], iu0, penalty * dz)
        np.add.at(grad[2::3], iu1, penalty * dz)
        
        return grad

    def repair(v):
        """Enforce validity by shrinking radii if constraints are violated."""
        p = v.reshape(N, 3).copy()
        p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
        p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
        p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)
        
        # Enforce wall constraints
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))
        
        # Enforce pairwise constraints iteratively
        for _ in range(50):
            iu = np.triu_indices(N, k=1)
            dx = p[iu[0], 0] - p[iu[1], 0]
            dy = p[iu[0], 1] - p[iu[1], 1]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = (p[iu[0], 2] + p[iu[1], 2]) - dist
            active = overlap > 0
            if not np.any(active):
                break
            for k in range(len(active)):
                if not active[k]:
                    continue
                i = iu[0][k]
                j = iu[1][k]
                ov = overlap[k]
                if p[i, 2] >= p[j, 2]:
                    shrink = min(ov, p[i, 2] - 1e-6)
                    p[i, 2] -= shrink
                    rem = ov - shrink
                    if rem > 0:
                        shrink2 = min(rem, p[j, 2] - 1e-6)
                        p[j, 2] -= shrink2
                else:
                    shrink = min(ov, p[j, 2] - 1e-6)
                    p[j, 2] -= shrink
                    rem = ov - shrink
                    if rem > 0:
                        shrink2 = min(rem, p[i, 2] - 1e-6)
                        p[i, 2] -= shrink2
            wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                              np.minimum(p[:, 1], 1.0 - p[:, 1]))
            p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))
            
        return p.flatten()

    def is_valid(v, tol=1e-7):
        p = v.reshape(N, 3)
        x, y, r = p[:, 0], p[:, 1], p[:, 2]
        if np.any(r < 0) or np.any(x < -tol) or np.any(x > 1+tol) or np.any(y < -tol) or np.any(y > 1+tol):
            return False
        if np.any(r > x + tol) or np.any(r > 1-x + tol) or np.any(r > y + tol) or np.any(r > 1-y + tol):
            return False
        iu = np.triu_indices(N, k=1)
        dx = x[iu[0]] - x[iu[1]]
        dy = y[iu[0]] - y[iu[1]]
        dist = np.sqrt(dx*dx + dy*dy)
        if np.any(r[iu[0]] + r[iu[1]] > dist + tol):
            return False
        return True

    bounds = [(0, 1), (0, 1), (1e-6, 0.5)] * N
    
    best_score = -1
    best_v = None
    
    # Multi-start with different initial layouts
    initial_generators = [get_initial_layout_hex, get_initial_layout_corner, get_initial_layout_grid]
    gen_idx = 0
    
    while time.time() < deadline:
        # Choose initial layout
        if gen_idx == 0:
            v_init = get_initial_layout_hex()
        elif gen_idx == 1:
            v_init = get_initial_layout_corner()
        else:
            v_init = get_initial_layout_grid()
        
        # Add some randomness
        v_init = v_init + np.random.normal(0, 0.03, size=3*N)
        v_init[0::3] = np.clip(v_init[0::3], 0, 1)
        v_init[1::3] = np.clip(v_init[1::3], 0, 1)
        v_init[2::3] = np.clip(v_init[2::3], 1e-6, 0.5)
        
        # Phase 1: Optimization with full pairwise constraints
        try:
            res = minimize(objective, v_init, jac=gradient, method='L-BFGS-B', bounds=bounds, 
                           args=(1000.0, 1e-4, None),
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            v_res = res.x
        except Exception:
            v_res = v_init
        
        # Repair
        v_repaired = repair(v_res)
        
        if is_valid(v_repaired):
            score = np.sum(v_repaired[2::3])
            if score > best_score:
                best_score = score
                best_v = v_repaired.copy()
        
        # Phase 2: Polish with contact edges
        if best_v is not None and time.time() < deadline - 2:
            # Get contact edges from current best
            contact_edges = get_contact_edges(best_v)
            
            # Polish around best solution
            v_polish = best_v + np.random.normal(0, 0.01, size=3*N)
            v_polish[0::3] = np.clip(v_polish[0::3], 0, 1)
            v_polish[1::3] = np.clip(v_polish[1::3], 0, 1)
            v_polish[2::3] = np.clip(v_polish[2::3], 1e-6, 0.5)
            
            try:
                res_polish = minimize(objective, v_polish, jac=gradient, method='L-BFGS-B', bounds=bounds,
                                      args=(5000.0, 1e-5, contact_edges),
                                      options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10})
                v_polish_res = res_polish.x
            except Exception:
                v_polish_res = v_polish
            
            v_polish_repaired = repair(v_polish_res)
            
            if is_valid(v_polish_repaired):
                score_polish = np.sum(v_polish_repaired[2::3])
                if score_polish > best_score:
                    best_score = score_polish
                    best_v = v_polish_repaired.copy()
        
        # Next iteration
        gen_idx = (gen_idx + 1) % len(initial_generators)
        
        # If we have a good solution, perturb around it sometimes
        if best_v is not None and np.random.rand() < 0.5:
            v_init = best_v + np.random.normal(0, 0.02, size=3*N)
            v_init[0::3] = np.clip(v_init[0::3], 0, 1)
            v_init[1::3] = np.clip(v_init[1::3], 0, 1)
            v_init[2::3] = np.clip(v_init[2::3], 1e-6, 0.5)
    
    if best_v is None:
        # Fallback
        v_fallback = get_initial_layout_hex()
        v_fallback = repair(v_fallback)
        return v_fallback.reshape(N, 3)
    
    return best_v.reshape(N, 3)
