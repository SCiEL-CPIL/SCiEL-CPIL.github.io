import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    # --- Helper: Softplus ---
    def softplus(z, eps):
        return np.logaddexp(0, z / eps) * eps

    # --- Helper: Validate and Repair ---
    def validate_and_repair(v):
        p = v.reshape(N, 3).copy()
        # Clip positions
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
                # Shrink the larger radius or split if equal
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
            # Re-enforce wall constraints after shrinking
            wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                              np.minimum(p[:, 1], 1.0 - p[:, 1]))
            p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))
            
        return p

    # --- Objective and Gradient for Optimization ---
    def objective(v, penalty=1000.0, eps=1e-4):
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
        
        # Overlap violations
        iu = np.triu_indices(N, k=1)
        dx = x[iu[0]] - x[iu[1]]
        dy = y[iu[0]] - y[iu[1]]
        dist = np.sqrt(dx*dx + dy*dy + 1e-12)
        overlap = (r[iu[0]] + r[iu[1]]) - dist
        pair_pen = np.sum(softplus(overlap, eps)**2)
        
        return score + penalty * (wall_pen + pair_pen)

    def gradient(v, penalty=1000.0, eps=1e-4):
        x = v[0::3]
        y = v[1::3]
        r = np.maximum(v[2::3], 1e-6)
        
        grad = np.zeros_like(v)
        
        # Gradient of -sum(r)
        grad[2::3] -= 1.0
        
        # Helper for derivative of softplus(z, eps)^2
        def d_softplus_sq(z, eps):
            sp = softplus(z, eps)
            sig = 1.0 / (1.0 + np.exp(-z / eps))
            return 2.0 * sp * sig

        # Boundary gradients
        for i in range(N):
            # Left wall: z = r - x
            z = r[i] - x[i]
            dz = d_softplus_sq(z, eps)
            grad[i*3] += -penalty * dz
            grad[i*3+2] += penalty * dz
            
            # Right wall: z = x + r - 1
            z = x[i] + r[i] - 1.0
            dz = d_softplus_sq(z, eps)
            grad[i*3] += penalty * dz
            grad[i*3+2] += penalty * dz
            
            # Bottom wall: z = r - y
            z = r[i] - y[i]
            dz = d_softplus_sq(z, eps)
            grad[i*3+1] += -penalty * dz
            grad[i*3+1+2] += penalty * dz
            
            # Top wall: z = y + r - 1
            z = y[i] + r[i] - 1.0
            dz = d_softplus_sq(z, eps)
            grad[i*3+1] += penalty * dz
            grad[i*3+1+2] += penalty * dz

        # Pairwise gradients
        iu = np.triu_indices(N, k=1)
        dx = x[iu[0]] - x[iu[1]]
        dy = y[iu[0]] - y[iu[1]]
        dist = np.sqrt(dx*dx + dy*dy + 1e-12)
        overlap = (r[iu[0]] + r[iu[1]]) - dist
        
        dz = d_softplus_sq(overlap, eps)
        ux = dx / dist
        uy = dy / dist
        
        # Accumulate into grad
        np.add.at(grad[0::3], iu[0], -penalty * dz * ux)
        np.add.at(grad[0::3], iu[1], penalty * dz * ux)
        np.add.at(grad[1::3], iu[0], -penalty * dz * uy)
        np.add.at(grad[1::3], iu[1], penalty * dz * uy)
        np.add.at(grad[2::3], iu[0], penalty * dz)
        np.add.at(grad[2::3], iu[1], penalty * dz)
        
        return grad

    # --- Initial Layout Generation (Hexagonal) ---
    def get_initial_layout():
        r0 = 0.12
        centers = []
        for row in range(10):
            for col in range(10):
                x = col * 2 * r0
                y = row * np.sqrt(3) * r0
                if row % 2 == 1:
                    x += r0
                centers.append((x, y))
        
        centers.sort(key=lambda p: (p[0]-0.5)**2 + (p[1]-0.5)**2)
        centers = centers[:N]
        
        xs = [c[0] for c in centers]
        ys = [c[1] for c in centers]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max_x - min_x
        height = max_y - min_y
        
        scale = min(0.8 / width, 0.8 / height)
        
        scaled_centers = []
        for x, y in centers:
            sx = (x - min_x) * scale + 0.1
            sy = (y - min_y) * scale + 0.1
            scaled_centers.append((sx, sy))
            
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

    # --- Main Optimization Loop ---
    best_score = -1
    best_v = None
    
    # Initial solution from hex grid
    v0 = get_initial_layout()
    p0 = validate_and_repair(v0)
    best_v = p0
    best_score = np.sum(p0[:, 2])
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Multi-start: perturb the best solution or use random hex
        if rng.random() < 0.5:
            v_init = best_v.copy()
            v_init += rng.normal(0, 0.02, v_init.shape)
        else:
            v_init = get_initial_layout()
            v_init += rng.normal(0, 0.05, v_init.shape)
        
        v_init = validate_and_repair(v_init)
        
        try:
            res = minimize(
                objective, 
                v_init, 
                jac=gradient, 
                method='L-BFGS-B', 
                bounds=[(0, 1), (0, 1), (1e-6, 0.5)] * N,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            v_opt = res.x
            p_opt = validate_and_repair(v_opt)
            score = np.sum(p_opt[:, 2])
            
            if score > best_score:
                best_score = score
                best_v = p_opt
                
        except Exception as e:
            pass
            
    return best_v
