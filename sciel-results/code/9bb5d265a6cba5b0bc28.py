import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import linprog
from scipy.spatial import distance

N = 26
DEADLINE = 145.0  # Safety margin before 150s limit

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def _is_valid(pos, radii, tol=1e-9):
        """Strict validity check."""
        if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
            return False
        if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
            return False
        dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
        dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
        dist_sq = dx * dx + dy * dy
        min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        if np.any(dist_sq[mask] < (min_dist[mask] - tol) ** 2):
            return False
        return True
    
    def _post_process(pos, radii):
        """Ensure strict validity by minimally adjusting radii/positions."""
        pos = pos.copy()
        radii = radii.copy()
        
        for _ in range(100):
            changed = False
            for i in range(N):
                max_r_bound = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
                if radii[i] > max_r_bound:
                    radii[i] = max_r_bound * (1.0 - 1e-10)
                    changed = True
                for j in range(i + 1, N):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < radii[i] + radii[j]:
                        total_r = radii[i] + radii[j]
                        if total_r > 0:
                            scale = dist / total_r * (1.0 - 1e-10)
                            radii[i] *= scale
                            radii[j] *= scale
                            changed = True
            if not changed:
                break
        
        if not _is_valid(pos, radii, tol=1e-9):
            for i in range(N):
                radii[i] *= (1.0 - 1e-8)
        
        return pos, radii
    
    def _lp_maximize_radii(pos, tol=1e-6):
        """
        Given fixed positions, solve LP to maximize sum of radii subject to:
        - r_i <= pos[i,0], r_i <= 1-pos[i,0], r_i <= pos[i,1], r_i <= 1-pos[i,1]
        - r_i + r_j <= dist(i,j) for all pairs
        - r_i >= 0
        """
        n = N
        c = -np.ones(n)
        
        A_rows = []
        b_rows = []
        
        for i in range(n):
            row = np.zeros(n)
            row[i] = 1.0
            A_rows.append(row)
            b_rows.append(pos[i, 0])
            
            row = np.zeros(n)
            row[i] = 1.0
            A_rows.append(row)
            b_rows.append(1.0 - pos[i, 0])
            
            row = np.zeros(n)
            row[i] = 1.0
            A_rows.append(row)
            b_rows.append(pos[i, 1])
            
            row = np.zeros(n)
            row[i] = 1.0
            A_rows.append(row)
            b_rows.append(1.0 - pos[i, 1])
        
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                d = np.sqrt(dx * dx + dy * dy)
                row = np.zeros(n)
                row[i] = 1.0
                row[j] = 1.0
                A_rows.append(row)
                b_rows.append(d)
        
        A_ub = np.array(A_rows)
        b_ub = np.array(b_rows)
        bounds = [(0, None)] * n
        
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if result.success:
                radii = np.maximum(result.x, 0.0)
                return radii
        except Exception:
            pass
        
        # Fallback: greedy radius assignment
        radii = np.zeros(n)
        for i in range(n):
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            for j in range(n):
                if j != i:
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    d = np.sqrt(dx * dx + dy * dy)
                    max_r = min(max_r, d - radii[j])
            radii[i] = max(0.0, max_r)
        return radii
    
    def _force_directed_relaxation(pos, radii, n_iter=2000, lr=0.01):
        """
        Force-directed relaxation: push overlapping circles apart,
        pull circles toward center if near boundary.
        """
        pos = pos.copy()
        radii = radii.copy()
        
        for it in range(n_iter):
            if time_left() < 5:
                break
            
            forces = np.zeros_like(pos)
            
            for i in range(N):
                for j in range(i + 1, N):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    d = np.sqrt(dx * dx + dy * dy)
                    min_d = radii[i] + radii[j]
                    if d < min_d and d > 1e-10:
                        overlap = min_d - d
                        fx = (dx / d) * overlap
                        fy = (dy / d) * overlap
                        forces[i, 0] += fx
                        forces[i, 1] += fy
                        forces[j, 0] -= fx
                        forces[j, 1] -= fy
                    elif d <= 1e-10:
                        angle = np.random.uniform(0, 2 * np.pi)
                        forces[i, 0] += 0.01 * np.cos(angle)
                        forces[i, 1] += 0.01 * np.sin(angle)
                        forces[j, 0] -= 0.01 * np.cos(angle)
                        forces[j, 1] -= 0.01 * np.sin(angle)
            
            for i in range(N):
                if pos[i, 0] < radii[i]:
                    forces[i, 0] += (radii[i] - pos[i, 0])
                if pos[i, 0] > 1.0 - radii[i]:
                    forces[i, 0] -= (pos[i, 0] - (1.0 - radii[i]))
                if pos[i, 1] < radii[i]:
                    forces[i, 1] += (radii[i] - pos[i, 1])
                if pos[i, 1] > 1.0 - radii[i]:
                    forces[i, 1] -= (pos[i, 1] - (1.0 - radii[i]))
            
            decay = 1.0 - 0.5 * (it / n_iter)
            step = lr * decay
            pos += step * forces
            
            pos[:, 0] = np.clip(pos[:, 0], 0.0, 1.0)
            pos[:, 1] = np.clip(pos[:, 1], 0.0, 1.0)
        
        return pos, radii
    
    def _best_fit_init(seed=0):
        """Greedy Best-Fit initialization."""
        rng = np.random.RandomState(seed)
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        
        pos[0] = [0.5, 0.5]
        radii[0] = 0.25

        for i in range(1, N):
            best_x, best_y, best_r = 0.5, 0.5, 0.0
            found = False
            
            for _ in range(10000):
                x = rng.uniform(0.05, 0.95)
                y = rng.uniform(0.05, 0.95)
                max_r_wall = min(x, 1.0 - x, y, 1.0 - y)
                max_r_circ = max_r_wall
                for j in range(i):
                    dx = x - pos[j, 0]
                    dy = y - pos[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    possible_r = dist - radii[j]
                    if possible_r < max_r_circ:
                        max_r_circ = possible_r
                if max_r_circ > best_r:
                    best_r = max_r_circ
                    best_x = x
                    best_y = y
                    found = True
            
            if found and best_r > 0.01:
                pos[i] = [best_x, best_y]
                radii[i] = best_r * 0.98 
            else:
                for _ in range(1000):
                    x = rng.uniform(0.05, 0.95)
                    y = rng.uniform(0.05, 0.95)
                    min_dist = min(x, 1-x, y, 1-y)
                    for j in range(i):
                        d = np.sqrt((x-pos[j,0])**2 + (y-pos[j,1])**2)
                        min_dist = min(min_dist, d - radii[j])
                    if min_dist > 0.005:
                        pos[i] = [x, y]
                        radii[i] = min_dist * 0.95
                        break
                else:
                    pos[i] = [0.5, 0.5]
                    radii[i] = 0.001

        return pos, radii
    
    def _hex_init():
        """Hexagonal lattice initialization."""
        pos = np.zeros((N, 2))
        radii = np.zeros(N)
        r = 0.11
        dx = r * np.sqrt(3)
        dy = r * 1.5
        
        k = 0
        for row in range(6):
            offset = dx / 2 if row % 2 == 1 else 0
            start_x = r + offset
            y = row * dy + r
            for col in range(5):
                if k >= N:
                    break
                x = start_x + col * dx
                if x + r <= 1.0 and x - r >= 0.0 and y + r <= 1.0 and y - r >= 0.0:
                    pos[k] = [x, y]
                    radii[k] = r
                    k += 1
            if k >= N:
                break

        rng = np.random.RandomState(123)
        while k < N:
            best_x, best_y, best_r = 0.5, 0.5, 0.0
            for _ in range(5000):
                x = rng.uniform(0.05, 0.95)
                y = rng.uniform(0.05, 0.95)
                max_r_wall = min(x, 1.0 - x, y, 1.0 - y)
                max_r_circ = max_r_wall
                for j in range(k):
                    dx = x - pos[j, 0]
                    dy = y - pos[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    possible_r = dist - radii[j]
                    if possible_r < max_r_circ:
                        max_r_circ = possible_r
                if max_r_circ > best_r:
                    best_r = max_r_circ
                    best_x = x
                    best_y = y
            if best_r > 0.01:
                pos[k] = [best_x, best_y]
                radii[k] = best_r * 0.98
            else:
                pos[k] = [0.5, 0.5]
                radii[k] = 0.01
            k += 1
        
        return pos, radii
    
    def _grid_init():
        """Grid-based initial configuration."""
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        pos = np.array([c[:2] for c in cells])
        radii = np.array([c[2] for c in cells])
        return pos, radii

    def _slsqp_polish(pos, radii, max_iter=300):
        """
        Final high-precision SLSQP polish with hard constraints.
        Optimizes both positions and radii simultaneously.
        """
        if time_left() < 10:
            return pos, radii

        x0 = np.concatenate([pos.flatten(), radii])

        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0))
            bounds.append((0.0, 1.0))
        for i in range(N):
            bounds.append((1e-8, 0.5))

        ii, jj = np.triu_indices(N, k=1)

        def constraint_boundary(x):
            p = x[:2 * N].reshape(N, 2)
            r = x[2 * N:]
            c1 = p[:, 0] - r
            c2 = 1.0 - p[:, 0] - r
            c3 = p[:, 1] - r
            c4 = 1.0 - p[:, 1] - r
            return np.concatenate([c1, c2, c3, c4])

        def constraint_overlap(x):
            p = x[:2 * N].reshape(N, 2)
            r = x[2 * N:]
            dx = p[ii, 0] - p[jj, 0]
            dy = p[ii, 1] - p[jj, 1]
            dist_sq = dx * dx + dy * dy
            min_dist = r[ii] + r[jj]
            return dist_sq - min_dist * min_dist

        def objective(x):
            return -np.sum(x[2*N:])

        constraints = [
            {'type': 'ineq', 'fun': constraint_boundary},
            {'type': 'ineq', 'fun': constraint_overlap}
        ]

        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=constraints,
                           options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-13, 'disp': False})
            if res.success or res.fun < objective(x0):
                pos_new = res.x[:2*N].reshape(N, 2)
                rad_new = np.maximum(res.x[2*N:], 1e-8)
                pos_pp, rad_pp = _post_process(pos_new, rad_new)
                if _is_valid(pos_pp, rad_pp, tol=1e-9):
                    return pos_pp, rad_pp
        except Exception:
            pass

        return pos, radii

    def _radius_growth_slsqp(pos, radii, max_iter=500):
        """
        Fix positions and maximize radii via SLSQP with hard constraints.
        """
        if time_left() < 10:
            return pos, radii

        pos_fixed = pos.copy()
        x0 = radii.copy()
        n = N

        ii, jj = np.triu_indices(n, k=1)
        dx = pos_fixed[ii, 0] - pos_fixed[jj, 0]
        dy = pos_fixed[ii, 1] - pos_fixed[jj, 1]
        dist_sq = dx * dx + dy * dy
        dist_vals = np.sqrt(dist_sq)

        def obj(x):
            return -np.sum(x)

        def constraint_boundary(x):
            c1 = pos_fixed[:, 0] - x
            c2 = 1.0 - pos_fixed[:, 0] - x
            c3 = pos_fixed[:, 1] - x
            c4 = 1.0 - pos_fixed[:, 1] - x
            return np.concatenate([c1, c2, c3, c4])

        def constraint_overlap(x):
            return dist_sq - (x[ii] + x[jj])**2

        constraints = [
            {'type': 'ineq', 'fun': constraint_boundary},
            {'type': 'ineq', 'fun': constraint_overlap}
        ]

        bounds = [(1e-8, 0.5)] * n

        try:
            res = minimize(obj, x0, method='SLSQP', bounds=bounds,
                           constraints=constraints,
                           options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'disp': False})
            if res.success or res.fun < obj(x0):
                new_radii = np.maximum(res.x, 1e-8)
                if _is_valid(pos_fixed, new_radii, tol=1e-9):
                    return pos_fixed, new_radii
        except Exception:
            pass

        return pos, radii

    # --- Main Optimization Loop ---
    
    # Generate diverse initializations
    initializations = []
    initializations.append(_best_fit_init(seed=42))
    initializations.append(_best_fit_init(seed=123))
    initializations.append(_hex_init())
    initializations.append(_grid_init())
    
    rng = np.random.RandomState(42)
    for _ in range(2):
        pos = rng.uniform(0.1, 0.9, size=(N, 2))
        radii = rng.uniform(0.03, 0.10, size=N)
        initializations.append((pos, radii))
    
    best_pos = None
    best_radii = None
    best_value = -np.inf
    
    # Phase 1: Quick relaxation and LP for multiple starts
    for init_idx, (pos, radii) in enumerate(initializations):
        if time_left() < 30:
            break
        
        # Force directed relaxation to resolve initial overlaps
        pos_rel, _ = _force_directed_relaxation(pos, radii, n_iter=1000, lr=0.005)
        
        # LP to maximize radii for these positions
        radii_lp = _lp_maximize_radii(pos_rel)
        
        if _is_valid(pos_rel, radii_lp, tol=1e-7):
            val = np.sum(radii_lp)
            if val > best_value:
                best_value = val
                best_pos = pos_rel.copy()
                best_radii = radii_lp.copy()
    
    if best_pos is None:
        best_pos, best_radii = _grid_init()
        best_value = np.sum(best_radii)
    
    # Phase 2: SLSQP Radius Growth (Fixed positions)
    if time_left() > 40:
        pos_g, rad_g = _radius_growth_slsqp(best_pos, best_radii)
        if np.sum(rad_g) > best_value:
            best_pos = pos_g
            best_radii = rad_g
            best_value = np.sum(rad_g)
            
    # Phase 3: Full SLSQP Polish (Positions + Radii)
    # This is the most powerful step, allowing positions to adjust to accommodate larger radii
    if time_left() > 30:
        pos_p, rad_p = _slsqp_polish(best_pos, best_radii, max_iter=500)
        if np.sum(rad_p) > best_value:
            best_pos = pos_p
            best_radii = rad_p
            best_value = np.sum(rad_p)
            
    # Phase 4: Final Polish if time remains
    if time_left() > 15:
        pos_f, rad_f = _slsqp_polish(best_pos, best_radii, max_iter=200)
        if np.sum(rad_f) > best_value:
            best_pos = pos_f
            best_radii = rad_f
            best_value = np.sum(rad_f)

    # Final post-process
    best_pos, best_radii = _post_process(best_pos, best_radii)
    
    result = np.column_stack([best_pos, best_radii])
    return result

if __name__ == "__main__":
    res = construct()
    print(res.shape)
