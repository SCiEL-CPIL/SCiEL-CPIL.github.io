import numpy as np
import scipy.optimize as opt
import time

N = 26

def construct():
    start_time = time.time()
    deadline = 130  # Safety margin

    def unpack(z):
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        return x, y, r

    def objective(z):
        _, _, r = unpack(z)
        return -np.sum(r)

    def c_boundary_x_min(z):
        x, _, r = unpack(z)
        return x - r

    def c_boundary_x_max(z):
        x, _, r = unpack(z)
        return 1 - x - r

    def c_boundary_y_min(z):
        _, y, r = unpack(z)
        return y - r

    def c_boundary_y_max(z):
        _, y, r = unpack(z)
        return 1 - y - r

    def c_non_overlap(z):
        x, y, r = unpack(z)
        dx = np.subtract.outer(x, x)
        dy = np.subtract.outer(y, y)
        dr = np.add.outer(r, r)
        dist_sq = dx**2 + dy**2
        radius_sum_sq = dr**2
        mask = np.triu_indices(N, k=1)
        return dist_sq[mask] - radius_sum_sq[mask]

    def c_radius_pos(z):
        _, _, r = unpack(z)
        return r - 1e-6

    bounds = [(0, 1), (0, 1), (1e-6, 0.5)] * N

    cons_list = [
        {'type': 'ineq', 'fun': c_boundary_x_min},
        {'type': 'ineq', 'fun': c_boundary_x_max},
        {'type': 'ineq', 'fun': c_boundary_y_min},
        {'type': 'ineq', 'fun': c_boundary_y_max},
        {'type': 'ineq', 'fun': c_non_overlap},
        {'type': 'ineq', 'fun': c_radius_pos}
    ]

    def solve_single(x0):
        # Phase 1: SLSQP
        try:
            res = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, 
                               constraints=cons_list, options={'maxiter': 1000, 'ftol': 1e-8})
            if not res.success:
                return x0
            z_best = res.x
        except Exception:
            return x0

        # Phase 2: LP
        x, y, r = unpack(z_best)
        c = -np.ones(N)
        A_ub = []
        b_ub = []
        
        I = np.eye(N)
        A_ub.append(I); b_ub.append(x)
        A_ub.append(I); b_ub.append(1 - x)
        A_ub.append(I); b_ub.append(y)
        A_ub.append(I); b_ub.append(1 - y)
        
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                row = np.zeros(N)
                row[i] = 1
                row[j] = 1
                A_ub.append(row)
                b_ub.append(d)
                
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        try:
            res_lp = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)]*N, method='highs')
            if res_lp.success:
                r_lp = res_lp.x
                z_best[2::3] = r_lp
        except Exception:
            pass

        # Phase 3: SLSQP Polish
        try:
            res2 = opt.minimize(objective, z_best, method='SLSQP', bounds=bounds, 
                                constraints=cons_list, options={'maxiter': 500, 'ftol': 1e-10})
            if res2.success:
                z_best = res2.x
        except Exception:
            pass
            
        return z_best

    # Config 1: 6x6 Grid
    side = 6
    xs = np.linspace(0.1, 0.9, side)
    ys = np.linspace(0.1, 0.9, side)
    XX, YY = np.meshgrid(xs, ys)
    points = np.vstack([XX.flatten(), YY.flatten()]).T
    points = points[:N]
    x0_1 = np.zeros(N * 3)
    x0_1[0::3] = points[:, 0]
    x0_1[1::3] = points[:, 1]
    x0_1[2::3] = 0.05

    best_z = x0_1
    best_val = objective(x0_1)

    if time.time() - start_time < deadline:
        z1 = solve_single(x0_1)
        if objective(z1) < best_val:
            best_val = objective(z1)
            best_z = z1

    # Config 2: 5x5 Grid + 1
    side = 5
    xs = np.linspace(0.1, 0.9, side)
    ys = np.linspace(0.1, 0.9, side)
    XX, YY = np.meshgrid(xs, ys)
    points = np.vstack([XX.flatten(), YY.flatten()]).T
    # Take 25 points
    points25 = points[:25]
    # Add 1 point at center (0.5, 0.5)
    # But (0.5, 0.5) is occupied in 5x5 grid?
    # 5x5 grid centers: 0.1, 0.3, 0.5, 0.7, 0.9. Yes, (0.5,0.5) is center.
    # So remove center and add 2 points? Or just perturb.
    # Let's just use a random perturbation of the 6x6 grid for the second run.
    x0_2 = x0_1 + np.random.normal(0, 0.01, N*3)
    # Ensure bounds
    x0_2[0::3] = np.clip(x0_2[0::3], 0.01, 0.99)
    x0_2[1::3] = np.clip(x0_2[1::3], 0.01, 0.99)
    x0_2[2::3] = 0.05

    if time.time() - start_time < deadline:
        z2 = solve_single(x0_2)
        if objective(z2) < best_val:
            best_val = objective(z2)
            best_z = z2

    # Final Validation
    x, y, r = unpack(best_z)
    
    # Check validity
    valid = True
    if np.any(x < r - 1e-9) or np.any(x > 1 - r + 1e-9) or np.any(y < r - 1e-9) or np.any(y > 1 - r + 1e-9):
        valid = False
        
    if valid:
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if dist < r[i] + r[j] - 1e-9:
                    valid = False
                    break
            if not valid: break

    if not valid:
        # Fallback to grid
        side = 6
        xs = np.linspace(0.1, 0.9, side)
        ys = np.linspace(0.1, 0.9, side)
        XX, YY = np.meshgrid(xs, ys)
        points = np.vstack([XX.flatten(), YY.flatten()]).T
        points = points[:N]
        r0 = 0.5/side - 1e-3
        result = np.column_stack((points[:, 0], points[:, 1], np.full(N, r0)))
        return result

    result = np.column_stack((x, y, r))
    return result
