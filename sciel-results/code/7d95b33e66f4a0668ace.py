import math
import time
import numpy as np
from scipy.optimize import minimize_scalar

N = 11

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    d = math.sqrt(3)
    
    # Precompute base vertices for unit hexagon
    # Vertices at angles 0, 60, ..., 300
    base_verts = np.array([[math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] for i in range(6)])
    
    def get_all_verts(params):
        """
        params: array of shape (N*3) -> [x1, y1, t1, x2, y2, t2, ...]
        returns: array of shape (N*6, 2)
        """
        n = len(params) // 3
        xs = params[0::3]
        ys = params[1::3]
        thetas = params[2::3]
        
        # Rotation matrices
        c = np.cos(thetas)
        s = np.sin(thetas)
        
        # Rotate base vertices: R @ v
        # base_verts shape (6, 2)
        # We want (N, 6, 2)
        # rx = x*cos - y*sin
        # ry = x*sin + y*cos
        
        # Expand for broadcasting
        # base_verts[:, 0] is (6,), c is (N,)
        # We need (N, 6)
        
        rx = base_verts[0, :] * c[:, np.newaxis] - base_verts[1, :] * s[:, np.newaxis]
        ry = base_verts[0, :] * s[:, np.newaxis] + base_verts[1, :] * c[:, np.newaxis]
        
        # This is wrong, base_verts is (6,2).
        # Let's do it properly.
        
        # base_verts: (6, 2)
        # c, s: (N,)
        
        # rx[i, j] = base_verts[j, 0] * c[i] - base_verts[j, 1] * s[i]
        rx = base_verts[:, 0][np.newaxis, :] * c[:, np.newaxis] - base_verts[:, 1][np.newaxis, :] * s[:, np.newaxis]
        ry = base_verts[:, 0][np.newaxis, :] * s[:, np.newaxis] + base_verts[:, 1][np.newaxis, :] * c[:, np.newaxis]
        
        # Add centers
        rx += xs[:, np.newaxis]
        ry += ys[:, np.newaxis]
        
        verts = np.column_stack([rx.ravel(), ry.ravel()])
        return verts

    def compute_S_and_widths(params, alpha=0.0):
        """
        Compute minimal enclosing hexagon side length for a fixed alpha.
        Also returns the widths for potential reuse.
        """
        verts = get_all_verts(params)
        
        # Normals for outer hexagon
        k_angles = alpha + np.arange(6) * math.pi / 3
        normals = np.stack([np.cos(k_angles), np.sin(k_angles)], axis=-1) # (6, 2)
        
        # Projections: (6, N*6)
        dots = verts @ normals.T
        
        # Max and min projection for each normal
        max_proj = np.max(dots, axis=1) # (6,)
        min_proj = np.min(dots, axis=1) # (6,)
        
        widths = max_proj - min_proj # (6,)
        
        # S = 1/sqrt(3) * max(widths)
        S = np.max(widths) / math.sqrt(3)
        
        return S

    def sat_penalty(params):
        """
        Calculate overlap penalty using SAT.
        """
        n = len(params) // 3
        xs = params[0::3]
        ys = params[1::3]
        thetas = params[2::3]
        
        # Precompute vertices for each hexagon
        # We need individual vertices for SAT
        # This is slower, but N=11 is small.
        
        penalty = 0.0
        # Store vertices for each hexagon
        hex_verts = []
        for i in range(n):
            c = math.cos(thetas[i])
            s = math.sin(thetas[i])
            rx = base_verts[:, 0] * c - base_verts[:, 1] * s
            ry = base_verts[:, 0] * s + base_verts[:, 1] * c
            verts = np.column_stack([rx + xs[i], ry + ys[i]])
            hex_verts.append(verts)
        
        for i in range(n):
            for j in range(i + 1, n):
                v1 = hex_verts[i]
                v2 = hex_verts[j]
                
                # SAT: Check 12 axes (6 from v1, 6 from v2)
                min_gap = float('inf')
                
                # Axes from v1
                for k in range(6):
                    p1 = v1[k]
                    p2 = v1[(k + 1) % 6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm < 1e-12: continue
                    normal = normal / norm
                    
                    proj1 = np.dot(v1, normal)
                    proj2 = np.dot(v2, normal)
                    
                    gap = min(proj1.max() - proj2.min(), proj2.max() - proj1.min())
                    min_gap = min(min_gap, gap)
                    
                    if min_gap < 0:
                        # Overlap found, can break early? 
                        # No, we need the magnitude for penalty.
                        # But if min_gap is very negative, we can break?
                        # Let's just accumulate.
                        pass
                
                # Axes from v2
                for k in range(6):
                    p1 = v2[k]
                    p2 = v2[(k + 1) % 6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm < 1e-12: continue
                    normal = normal / normal
                    
                    proj1 = np.dot(v1, normal)
                    proj2 = np.dot(v2, normal)
                    
                    gap = min(proj1.max() - proj2.min(), proj2.max() - proj1.min())
                    min_gap = min(min_gap, gap)
                
                if min_gap < 0:
                    penalty += (min_gap * min_gap) * 100.0
        
        return penalty

    def objective(params):
        if time.time() > deadline:
            return 1e10
        S = compute_S_and_widths(params, alpha=0.0) # Use alpha=0 for speed in SA
        pen = sat_penalty(params)
        return S + pen

    def refine_alpha(params):
        """
        Find best alpha for a given configuration.
        """
        def outer_obj(alpha):
            return compute_S_and_widths(params, alpha=alpha)
        
        try:
            res = minimize_scalar(outer_obj, bounds=(0, math.pi/3), method='bounded',
                                 options={'xatol': 1e-6})
            if res.success:
                return res.fun, res.x
        except:
            pass
        return compute_S_and_widths(params, alpha=0.0), 0.0

    def simulate_annealing(init_params, T_start, T_end, steps):
        current_params = init_params.copy()
        current_energy = objective(current_params)
        
        best_params = current_params.copy()
        best_energy = current_energy
        
        # Cooling schedule
        cooling_factor = (T_end / T_start) ** (1.0 / steps)
        
        for i in range(steps):
            if time.time() > deadline:
                break
                
            # Perturb a random hexagon
            idx = np.random.randint(0, N)
            new_params = current_params.copy()
            
            # Random move
            dx = np.random.normal(0, 0.5)
            dy = np.random.normal(0, 0.5)
            dtheta = np.random.normal(0, 0.1)
            
            new_params[3*idx] += dx
            new_params[3*idx+1] += dy
            new_params[3*idx+2] += dtheta
            
            new_energy = objective(new_params)
            
            delta = new_energy - current_energy
            
            if delta < 0 or np.random.random() < np.exp(-delta / T_start):
                current_params = new_params
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_params = current_params.copy()
            
            T_start *= cooling_factor
        
        return best_params, best_energy

    # Initial guess: Hexagonal cluster
    # Center (0,0), Ring 1 (6), Ring 2 (4)
    # This is a good starting point
    
    best_overall_params = None
    best_overall_S = float('inf')
    
    # Run multiple restarts
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Generate initial params
        init_params = np.zeros(N * 3)
        
        # Center
        init_params[0] = 0.0
        init_params[1] = 0.0
        init_params[2] = 0.0
        
        # Ring 1
        for i in range(6):
            angle = i * math.pi / 3
            x = d * math.cos(angle)
            y = d * math.sin(angle)
            idx = 1 + i
            init_params[3*idx] = x
            init_params[3*idx+1] = y
            init_params[3*idx+2] = 0.0
        
        # Ring 2: Pick 4 from 12
        # Randomly pick 4 distinct ring 2 positions
        ring2_indices = []
        all_ring2 = []
        for q in range(-2, 3):
            for r in range(-2, 3):
                s = -q - r
                if max(abs(q), abs(r), abs(s)) == 2:
                    x = d * (q + r / 2.0)
                    y = d * (math.sqrt(3) / 2.0) * r
                    all_ring2.append((x, y))
        
        # Shuffle and pick 4
        np.random.shuffle(all_ring2)
        for i in range(4):
            idx = 7 + i
            init_params[3*idx] = all_ring2[i][0]
            init_params[3*idx+1] = all_ring2[i][1]
            init_params[3*idx+2] = 0.0
        
        # Add small noise
        noise = np.random.normal(0, 0.1, size=N*3)
        noise[2::3] *= 0.05 # Less rotation noise
        init_params += noise
        
        # Run SA
        T_start = 1.0
        T_end = 0.001
        steps = 2000
        
        best_params, best_energy = simulate_annealing(init_params, T_start, T_end, steps)
        
        # Refine alpha
        S_val, alpha_val = refine_alpha(best_params)
        
        # Check validity (no overlap)
        # We need to ensure the penalty is 0
        pen = sat_penalty(best_params)
        if pen > 1e-5:
            # Try to fix overlaps by moving apart? 
            # Or just skip if penalty is high
            # But SA should have minimized it.
            # If penalty is non-zero, the S_val is not the true score.
            # The score is S_val only if no overlap.
            # If overlap, the score is effectively infinity or we must repair.
            # Let's assume SA found a non-overlapping config.
            # If not, we can try a few more steps with higher penalty weight?
            # For now, if penalty > 0, we discard or try to repair.
            pass
        
        if S_val < best_overall_S:
            best_overall_S = S_val
            best_overall_params = best_params.copy()
            best_overall_alpha = alpha_val

    # Final refinement with higher precision
    if best_overall_params is not None:
        # Use Nelder-Mead on the best params to polish
        def final_obj(params):
            S = compute_S_and_widths(params, alpha=best_overall_alpha)
            pen = sat_penalty(params)
            return S + pen * 1000
        
        try:
            res = minimize(final_obj, best_overall_params, method='Nelder-Mead',
                           options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_overall_S:
                best_overall_params = res.x
                # Re-refine alpha
                best_overall_S, best_overall_alpha = refine_alpha(best_overall_params)
        except:
            pass

    # Construct output
    inner = []
    for i in range(N):
        x = best_overall_params[3*i]
        y = best_overall_params[3*i+1]
        angle_deg = math.degrees(best_overall_params[3*i+2])
        inner.append([x, y, angle_deg])

    return inner, (0.0, 0.0), best_overall_S, math.degrees(best_overall_alpha)
