import numpy as np
from scipy.optimize import minimize
import itertools

N = 21

def construct():
    best_value = -np.inf
    best_solution = None
    
    def evaluate_solution(x, y, r, w, h):
        if np.any(r <= 0):
            return -np.inf, False
        for i in range(N):
            if x[i] < r[i] - 1e-6 or x[i] > w - r[i] + 1e-6 or y[i] < r[i] - 1e-6 or y[i] > h - r[i] + 1e-6:
                return -np.inf, False
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if d < r[i] + r[j] - 1e-6:
                    return -np.inf, False
        return np.sum(r), True

    def cost_and_grad(z):
        w = z[0]
        h = 2.0 - w
        if w <= 0.1 or h <= 0.1:
            return 1e10, np.zeros_like(z)
        
        x = z[1::3]
        y = z[2::3]
        r = z[3::3]
        
        cost = -np.sum(r)
        grad = np.zeros_like(z)
        Lambda = 10.0
        
        for i in range(N):
            xi, yi, ri = x[i], y[i], r[i]
            
            viol_l = ri - xi
            if viol_l > 0:
                cost += Lambda * viol_l**2
                grad[1 + 3*i] += -2 * Lambda * viol_l
                grad[3 + 3*i] += 2 * Lambda * viol_l
                
            viol_r = xi + ri - w
            if viol_r > 0:
                cost += Lambda * viol_r**2
                grad[1 + 3*i] += 2 * Lambda * viol_r
                grad[3 + 3*i] += 2 * Lambda * viol_r
                grad[0] += -2 * Lambda * viol_r
                
            viol_b = ri - yi
            if viol_b > 0:
                cost += Lambda * viol_b**2
                grad[2 + 3*i] += -2 * Lambda * viol_b
                grad[3 + 3*i] += 2 * Lambda * viol_b
                
            viol_t = yi + ri - h
            if viol_t > 0:
                cost += Lambda * viol_t**2
                grad[2 + 3*i] += 2 * Lambda * viol_t
                grad[3 + 3*i] += 2 * Lambda * viol_t
                grad[0] += 2 * Lambda * viol_t
                
            grad[3 + 3*i] += -1.0

        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq + 1e-12)
                r_sum = r[i] + r[j]
                
                viol = r_sum - dist
                if viol > 0:
                    cost += Lambda * viol**2
                    grad[1 + 3*i] += 2 * Lambda * viol * ( -dx / dist )
                    grad[2 + 3*i] += 2 * Lambda * viol * ( -dy / dist )
                    grad[1 + 3*j] += 2 * Lambda * viol * ( dx / dist )
                    grad[2 + 3*j] += 2 * Lambda * viol * ( dy / dist )
                    grad[3 + 3*i] += 2 * Lambda * viol
                    grad[3 + 3*j] += 2 * Lambda * viol

        return cost, grad

    def generate_hex_candidates(w, h, spacing):
        points = []
        dy = spacing * np.sqrt(3) / 2
        row = 0
        y = dy / 2
        while y < h - dy/2:
            offset = 0 if row % 2 == 0 else spacing / 2
            x = offset
            while x < w - spacing/2:
                if x >= 0 and x <= w and y >= 0 and y <= h:
                    points.append((x, y))
                x += spacing
            row += 1
            y += dy
        return points

    def generate_grid_candidates(w, h, nx, ny):
        points = []
        for i in range(nx):
            for j in range(ny):
                x = (i + 0.5) * w / nx
                y = (j + 0.5) * h / ny
                points.append((x, y))
        return points

    def greedy_select(candidates, w, h, n_select):
        scored_candidates = []
        for (px, py) in candidates:
            r_max = min(px, py, w - px, h - py)
            if r_max > 1e-4:
                scored_candidates.append((px, py, r_max))
        
        scored_candidates.sort(key=lambda x: x[2], reverse=True)
        
        selected = []
        for px, py, r_max in scored_candidates:
            if len(selected) == n_select:
                break
            
            is_valid = True
            for (sx, sy, sr) in selected:
                dist = np.sqrt((px - sx)**2 + (py - sy)**2)
                if dist < (r_max + sr) * 0.9:
                    is_valid = False
                    break
            
            if is_valid:
                selected.append((px, py, r_max))
                
        if len(selected) < n_select:
            for px, py, r_max in scored_candidates:
                if len(selected) == n_select:
                    break
                already = False
                for (sx, sy, _) in selected:
                    if np.allclose([px, py], [sx, sy]):
                        already = True
                        break
                if not already:
                    selected.append((px, py, r_max))
                    
        return selected[:n_select]

    aspect_ratios = [1.0, 0.8, 1.2, 0.9, 1.1]
    
    for w in aspect_ratios:
        h = 2.0 - w
        if w <= 0.1 or h <= 0.1:
            continue
            
        candidates_grid = generate_grid_candidates(w, h, 6, 6)
        candidates_hex = generate_hex_candidates(w, h, 0.2)
        all_candidates = candidates_grid + candidates_hex
        
        selected = greedy_select(all_candidates, w, h, N)
        
        if len(selected) < N:
            continue
            
        z0 = np.zeros(1 + 3*N)
        z0[0] = w
        for i in range(N):
            px, py, r_max = selected[i]
            z0[1 + 3*i] = px
            z0[2 + 3*i] = py
            z0[3 + 3*i] = r_max * 0.5 
            
        try:
            res = minimize(cost_and_grad, z0, method='L-BFGS-B', 
                           jac=True, options={'maxiter': 500, 'ftol': 1e-10})
            
            z_opt = res.x
            w_opt = z_opt[0]
            h_opt = 2.0 - w_opt
            
            x = z_opt[1::3]
            y = z_opt[2::3]
            r = z_opt[3::3]
            
            val, valid = evaluate_solution(x, y, r, w_opt, h_opt)
            
            if valid and val > best_value:
                best_value = val
                best_solution = np.column_stack((x, y, r))
                
        except Exception:
            pass

    if best_solution is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_solution
