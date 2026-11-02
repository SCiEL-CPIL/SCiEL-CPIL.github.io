"""Combined program: symmetric global search (DE) + asymmetric local refinement (Nelder-Mead)."""
import numpy as np
import time
import random
import scipy.optimize
from scipy.optimize import differential_evolution, minimize

N = 12
SQRT3 = np.sqrt(3.0)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        """Vertices of regular hexagon centered at (cx,cy), side length, rotation in degrees."""
        angle = np.radians(angle_deg)
        thetas = angle + np.arange(6) * np.pi / 3.0
        return np.column_stack([
            cx + side * np.cos(thetas),
            cy + side * np.sin(thetas)
        ])

    def get_all_vertices(inner):
        """Stack all vertices of all inner hexagons into (N*6, 2) array."""
        verts = []
        for i in range(N):
            verts.append(hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]))
        return np.vstack(verts)

    def compute_min_side(inner, center, angle):
        """Compute minimum outer hexagon side length to contain all inner hexagons."""
        all_verts = get_all_vertices(inner)
        offsets = all_verts - np.array(center)
        
        # Outer hexagon edge normals (6 directions)
        base_angle = np.radians(angle) + np.radians(30)
        max_extent = 0.0
        for k in range(6):
            na = base_angle + k * np.pi / 3.0
            normal = np.array([np.cos(na), np.sin(na)])
            extents = offsets @ normal
            m = np.max(extents)
            if m > max_extent:
                max_extent = m
        return max_extent * 2.0 / SQRT3

    def sat_disjoint(v1, v2):
        """SAT check between two hexagon vertex arrays (6,2) each. Returns True if disjoint."""
        edges = []
        for i in range(6):
            edges.append(v1[i] - v1[(i+1) % 6])
            edges.append(v2[i] - v2[(i+1) % 6])
        for e in edges:
            n = np.array([-e[1], e[0]])
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                continue
            n /= nn
            p1 = v1 @ n
            p2 = v2 @ n
            if np.max(p1) < np.min(p2) - 1e-10 or np.max(p2) < np.min(p1) - 1e-10:
                return True
        return False

    def hexagons_disjoint(h1, h2):
        """Check if two hexagons (cx,cy,angle) are disjoint."""
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        if dist_sq > 4.5:
            return True
        v1 = hexagon_vertices(h1[0], h1[1], 1.0, h1[2])
        v2 = hexagon_vertices(h2[0], h2[1], 1.0, h2[2])
        return sat_disjoint(v1, v2)

    def smooth_overlap_penalty(inner):
        """
        Smooth penalty for overlapping pairs. Uses distance-based soft penalty.
        """
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = inner[i][0] - inner[j][0]
                dy = inner[i][1] - inner[j][1]
                d = np.sqrt(dx*dx + dy*dy)
                if d < 2.2:
                    penalty += (2.2 - d) ** 2
        return penalty

    def objective(params):
        """
        Objective: side length + penalty for overlaps.
        params: [x1,y1,a1, x2,y2,a2, ..., x12,y12,a12, cx, cy, ca]
        """
        inner = [(params[3*i], params[3*i+1], params[3*i+2]) for i in range(N)]
        center = (params[3*N], params[3*N+1])
        angle = params[3*N+2]
        
        side = compute_min_side(inner, center, angle)
        pen = smooth_overlap_penalty(inner)
        return side + 50.0 * pen

    def objective_grad(params):
        """Same as objective but used for gradient computation via finite differences."""
        return objective(params)

    def initial_guess():
        """
        Honeycomb-like initial layout:
        1 center hexagon, 6 in first ring, 5 in second ring.
        """
        params = np.zeros(3*N + 3)
        positions = []
        # Center
        positions.append((0.0, 0.0, 0.0))
        # First ring: 6 hexagons at distance 2
        for i in range(6):
            theta = i * np.pi / 3.0
            positions.append((2.0*np.cos(theta), 2.0*np.sin(theta), 0.0))
        # Second ring: 5 hexagons at distance ~3.8, spread out
        for i in range(5):
            theta = i * 2.0*np.pi / 5.0 + np.pi / 5.0
            r = 3.8
            positions.append((r*np.cos(theta), r*np.sin(theta), 0.0))
        
        for i in range(N):
            params[3*i] = positions[i][0]
            params[3*i+1] = positions[i][1]
            params[3*i+2] = positions[i][2]
        
        params[3*N] = 0.0
        params[3*N+1] = 0.0
        params[3*N+2] = 0.0
        return params

    def random_guess():
        """Random initial guess in a reasonable region."""
        params = np.zeros(3*N + 3)
        for i in range(N):
            r = random.uniform(0, 4.5)
            theta = random.uniform(0, 2*np.pi)
            params[3*i] = r * np.cos(theta)
            params[3*i+1] = r * np.sin(theta)
            params[3*i+2] = random.uniform(0, 60)
        params[3*N] = 0.0
        params[3*N+1] = 0.0
        params[3*N+2] = 0.0
        return params

    # Try multiple initializations and pick the best
    best_params = initial_guess()
    best_score = objective(best_params)
    
    for trial in range(5):
        if time.time() > deadline - 30:
            break
        p = random_guess()
        s = objective(p)
        if s < best_score:
            best_score = s
            best_params = p

    # Phase 1: Gradient descent with adaptive step size
    def finite_diff_gradient(params, h=1e-5):
        grad = np.zeros_like(params)
        for i in range(len(params)):
            orig = params[i]
            params[i] = orig + h
            f_plus = objective(params)
            params[i] = orig - h
            f_minus = objective(params)
            params[i] = orig
            grad[i] = (f_plus - f_minus) / (2*h)
        return grad

    current_params = best_params.copy()
    current_score = best_score
    lr = 0.05
    min_lr = 1e-6
    max_gd_steps = 300
    
    step_count = 0
    while time.time() < deadline - 40.0 and step_count < max_gd_steps:
        step_count += 1
        grad = finite_diff_gradient(current_params)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-8:
            break
        
        direction = -grad / grad_norm
        alpha = lr
        improved = False
        for _ in range(10):
            new_params = current_params + alpha * direction
            new_score = objective(new_params)
            if new_score < current_score:
                current_params = new_params
                current_score = new_score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_params = current_params.copy()
                break
            alpha *= 0.5
        
        if not improved:
            lr *= 0.5
            if lr < min_lr:
                break
        else:
            lr = min(lr * 1.1, 0.1)

    # Phase 2: Nelder-Mead polish
    if time.time() < deadline - 10.0:
        try:
            result = scipy.optimize.minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if result.fun < best_score:
                p = result.x
                inner = [(p[3*i], p[3*i+1], p[3*i+2]) for i in range(N)]
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if not hexagons_disjoint(inner[i], inner[j]):
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    best_score = compute_min_side(inner, (p[3*N], p[3*N+1]), p[3*N+2])
                    best_params = p
        except Exception:
            pass

    # Phase 3: Final high-precision polish with tighter tolerances
    if time.time() < deadline - 5.0:
        try:
            result = scipy.optimize.minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if result.fun < best_score:
                p = result.x
                inner = [(p[3*i], p[3*i+1], p[3*i+2]) for i in range(N)]
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if not hexagons_disjoint(inner[i], inner[j]):
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    best_score = compute_min_side(inner, (p[3*N], p[3*N+1]), p[3*N+2])
                    best_params = p
        except Exception:
            pass

    # Extract best
    inner = []
    for i in range(N):
        inner.append([float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])])
    
    center = (float(best_params[3*N]), float(best_params[3*N+1]))
    angle = float(best_params[3*N+2])
    side = compute_min_side(inner, center, angle)
    
    # Final validity check
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if not hexagons_disjoint(inner[i], inner[j]):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: use the initial guess which should be valid
        inner = [[0.0, 0.0, 0.0]] * N
        center = (0.0, 0.0)
        side = 10.0
        angle = 0.0

    return inner, center, side, angle
# EVOLVE-BLOCK-END
