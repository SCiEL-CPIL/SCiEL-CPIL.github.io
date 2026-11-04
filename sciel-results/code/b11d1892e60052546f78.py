"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, linprog
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    best_value = 0.0
    best_solution = None

    def validate_and_value(w, h, x, y, r):
        """Check validity and compute value."""
        if np.any(r < 1e-6) or w <= 0 or h <= 0:
            return 0.0, False
        if np.any(x < r - 1e-5) or np.any(x > w - r + 1e-5) or \
           np.any(y < r - 1e-5) or np.any(y > h - r + 1e-5):
            return 0.0, False
        diff_x = x[:, np.newaxis] - x[np.newaxis, :]
        diff_y = y[:, np.newaxis] - y[np.newaxis, :]
        dist = np.sqrt(diff_x**2 + diff_y**2)
        np.fill_diagonal(dist, np.inf)
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        if np.any(r_sum > dist + 1e-5):
            return 0.0, False
        return np.sum(r), True

    def build_hex_contact_graph():
        """Build contact graph for a dense hexagonal cluster of 21 circles."""
        # Hexagonal arrangement: rows of 4,5,4,5,3
        rows = [4, 5, 4, 5, 3]
        positions = []
        idx = 0
        for row_i, count in enumerate(rows):
            offset = (max(rows) - count) * 0.5
            for col in range(count):
                x = col + offset
                y = row_i * np.sqrt(3) / 2
                positions.append((idx, x, y))
                idx += 1
        
        # Find contacts: two circles are in contact if their centers are ~2*r apart
        # In a unit hexagonal lattice, nearest neighbors are distance 1.0
        # We'll use distance threshold of 1.1
        contacts = set()
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                xi, yi = positions[i][1], positions[i][2]
                xj, yj = positions[j][1], positions[j][2]
                d = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                if d < 1.1:
                    contacts.add((i, j))
        
        return contacts, positions

    def solve_positions_for_radii(w, h, r, contacts, positions_template):
        """
        Given fixed radii and aspect ratio, solve for positions using the contact graph.
        We set up equations: for each contact (i,j), |pi - pj| = ri + rj
        This is nonlinear, so we use penalty-based optimization with the contact structure.
        """
        n = len(r)
        
        # Initial positions from template, scaled to fit
        pos_template = np.array([(p[1], p[2]) for p in positions_template])
        # Scale template to fit in rectangle
        template_w = pos_template[:, 0].max() - pos_template[:, 0].min()
        template_h = pos_template[:, 1].max() - pos_template[:, 1].min()
        scale = min(w / (template_w + 2*r.max()), h / (template_h + 2*r.max())) * 0.9
        pos_init = pos_template * scale
        pos_init[:, 0] += (w - pos_init[:, 0].max() + pos_init[:, 0].min()) / 2
        pos_init[:, 1] += (h - pos_init[:, 1].max() + pos_init[:, 1].min()) / 2
        
        def objective(pos_flat):
            pos = pos_flat.reshape(n, 2)
            x = pos[:, 0]
            y = pos[:, 1]
            
            # Boundary penalties
            pen_x_low = np.maximum(0, r - x)
            pen_x_high = np.maximum(0, x + r - w)
            pen_y_low = np.maximum(0, r - y)
            pen_y_high = np.maximum(0, y + r - h)
            bound_pen = np.sum(pen_x_low**2) + np.sum(pen_x_high**2) + \
                       np.sum(pen_y_low**2) + np.sum(pen_y_high**2)
            
            # Contact penalties: for each contact, enforce distance = ri + rj
            contact_pen = 0.0
            for (i, j) in contacts:
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx**2 + dy**2)
                target = r[i] + r[j]
                contact_pen += (d - target)**2
            
            # Non-contact separation penalty
            non_contact_pen = 0.0
            for i in range(n):
                for j in range(i+1, n):
                    if (i, j) not in contacts and (j, i) not in contacts:
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        d = np.sqrt(dx**2 + dy**2)
                        target = r[i] + r[j]
                        if d < target:
                            non_contact_pen += (target - d)**2
            
            total_pen = bound_pen + 10.0 * contact_pen + 100.0 * non_contact_pen
            return total_pen
        
        pos_flat_init = pos_init.flatten()
        bounds_list = [(r[i], w - r[i]) for i in range(n)] + [(r[i], h - r[i]) for i in range(n)]
        
        try:
            res = minimize(objective, pos_flat_init, method='L-BFGS-B', 
                          bounds=bounds_list, options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            return res.x.reshape(n, 2)
        except:
            return pos_init

    def try_aspect_ratio(w, h, contacts, positions_template):
        """For a given aspect ratio, optimize radii using the contact graph structure."""
        n = len(contacts)
        
        # Initial radii: uniform
        r_init = np.ones(n) * 0.1
        
        def objective_r(r_flat):
            r = np.maximum(r_flat, 1e-6)
            pos = solve_positions_for_radii(w, h, r, contacts, positions_template)
            x = pos[:, 0]
            y = pos[:, 1]
            
            # Check if solution is valid
            valid, val = validate_and_value(w, h, x, y, r)
            if not valid:
                # Penalty for invalidity
                pen = 0.0
                pen += np.sum(np.maximum(0, r - x)**2)
                pen += np.sum(np.maximum(0, x + r - w)**2)
                pen += np.sum(np.maximum(0, r - y)**2)
                pen += np.sum(np.maximum(0, y + r - h)**2)
                diff_x = x[:, np.newaxis] - x[np.newaxis, :]
                diff_y = y[:, np.newaxis] - y[np.newaxis, :]
                dist = np.sqrt(diff_x**2 + diff_y**2)
                np.fill_diagonal(dist, np.inf)
                r_sum = r[:, np.newaxis] + r[np.newaxis, :]
                pen += np.sum(np.maximum(0, r_sum - dist)**2)
                return pen - np.sum(r)
            return -val
        
        # Optimize radii
        bounds_r = [(1e-4, min(w, h)/2)] * n
        try:
            res = minimize(objective_r, r_init, method='Nelder-Mead', 
                          options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
            r_opt = np.maximum(res.x, 1e-6)
            pos_opt = solve_positions_for_radii(w, h, r_opt, contacts, positions_template)
            x_opt = pos_opt[:, 0]
            y_opt = pos_opt[:, 1]
            
            valid, val = validate_and_value(w, h, x_opt, y_opt, r_opt)
            if valid and val > best_value:
                return val, np.column_stack((x_opt, y_opt, r_opt))
        except:
            pass
        return 0.0, None

    # Build contact graph
    contacts, positions_template = build_hex_contact_graph()
    
    # Try multiple aspect ratios
    aspect_ratios = np.linspace(0.5, 1.5, 20)  # w/h ratio
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        w = 2.0 * ar / (1 + ar)
        h = 2.0 / (1 + ar)
        
        val, sol = try_aspect_ratio(w, h, contacts, positions_template)
        if sol is not None and val > best_value:
            best_value = val
            best_solution = sol
    
    # If contact-graph approach didn't find anything good, fall back to general optimization
    if best_value < 0.5:
        def init_hex_general(w, h):
            rows = [4, 5, 4, 5, 3]
            points = []
            r_init = 0.1
            max_row = max(rows)
            grid_w = max_row * 2.0 * r_init
            grid_h = len(rows) * np.sqrt(3) * r_init
            r_init = min(w / grid_w, h / grid_h) * 0.8 * r_init
            if r_init < 0.01:
                r_init = 0.01
            y_offset = (h - grid_h) / 2.0
            y_curr = y_offset
            for count in rows:
                x_offset = (w - count * 2.0 * r_init) / 2.0
                for i in range(count):
                    x = x_offset + i * 2.0 * r_init
                    y = y_curr
                    points.append([x, y, r_init])
                y_curr += np.sqrt(3) * r_init
            while len(points) < N:
                points.append([w/2, h/2, r_init])
            points = points[:N]
            x = np.array([p[0] for p in points])
            y = np.array([p[1] for p in points])
            r = np.array([p[2] for p in points])
            return np.concatenate(([w], x, y, r))
        
        def objective_general(v):
            w = v[0]
            h = 2.0 - w
            x = v[1:N+1]
            y = v[N+1:2*N+1]
            r = v[2*N+1:3*N+1]
            pen_x_low = np.maximum(0, r - x)
            pen_x_high = np.maximum(0, x + r - w)
            pen_y_low = np.maximum(0, r - y)
            pen_y_high = np.maximum(0, y + r - h)
            bound_pen = np.sum(pen_x_low**2) + np.sum(pen_x_high**2) + \
                       np.sum(pen_y_low**2) + np.sum(pen_y_high**2)
            diff_x = x[:, np.newaxis] - x[np.newaxis, :]
            diff_y = y[:, np.newaxis] - y[np.newaxis, :]
            dist = np.sqrt(diff_x**2 + diff_y**2)
            np.fill_diagonal(dist, np.inf)
            r_sum = r[:, np.newaxis] + r[np.newaxis, :]
            violation = np.maximum(0, r_sum - dist)
            tri_idx = np.triu_indices(N, k=1)
            overlap_pen = np.sum(violation[tri_idx]**2)
            total_pen = bound_pen + overlap_pen
            sum_r = np.sum(r)
            weight = 100.0
            return -sum_r + weight * total_pen
        
        bounds_list = [(0.1, 1.9)]
        for i in range(N):
            bounds_list.append((None, None))
        for i in range(N):
            bounds_list.append((None, None))
        for i in range(N):
            bounds_list.append((1e-6, None))
        
        for w_start in [0.8, 1.0, 1.2]:
            if time.time() > deadline:
                break
            x0 = init_hex_general(w_start, 2.0 - w_start)
            try:
                res = minimize(objective_general, x0, method='L-BFGS-B', bounds=bounds_list, 
                               options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
                w = res.x[0]
                h = 2.0 - w
                x = res.x[1:N+1]
                y = res.x[N+1:2*N+1]
                r = res.x[2*N+1:3*N+1]
                val, valid = validate_and_value(w, h, x, y, r)
                if valid and val > best_value:
                    best_value = val
                    best_solution = np.column_stack((x, y, r))
            except:
                pass

    if best_solution is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_solution

# EVOLVE-BLOCK-END
