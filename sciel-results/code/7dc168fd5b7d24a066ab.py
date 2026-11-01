import numpy as np
from scipy.optimize import minimize
import time

N = 32
TIME_LIMIT = 135.0  # Safety margin before 150s kill

def construct():
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    def is_valid(x, y, r, tol=1e-9):
        for i in range(N):
            if x[i] < -tol or x[i] > 1 + tol or y[i] < -tol or y[i] > 1 + tol:
                return False
            if r[i] < -tol:
                return False
            if r[i] > x[i] + tol or r[i] > 1 - x[i] + tol:
                return False
            if r[i] > y[i] + tol or r[i] > 1 - y[i] + tol:
                return False
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if d < r[i] + r[j] - tol:
                    return False
        return True
    
    def sum_r(r):
        return float(np.sum(r))
    
    def make_valid(x, y, r):
        x = x.copy()
        y = y.copy()
        r = r.copy()
        for _ in range(200):
            changed = False
            for i in range(N):
                mb = min(x[i], 1-x[i], y[i], 1-y[i])
                if r[i] > mb:
                    r[i] = max(0.0, mb * 0.999)
                    changed = True
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    s = r[i] + r[j]
                    if d < s:
                        if d < 1e-12:
                            r[i] = r[j] = min(r[i], r[j]) * 0.5
                        else:
                            f = d / s * 0.999
                            r[i] *= f
                            r[j] *= f
                        changed = True
            if not changed:
                break
        return x, y, r
    
    def hex_init():
        """Hexagonal packing initialization"""
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        # Hexagonal lattice with spacing ~ 2*r
        # For N=32, use a 6x6 grid clipped to 32
        cols = 6
        rows = 6
        spacing = 1.0 / (cols + 0.5)
        r0 = spacing * 0.48
        count = 0
        for row in range(rows):
            for col in range(cols):
                if count >= N:
                    break
                cx = (col + 0.5) * spacing + (0.5 * spacing if row % 2 else 0)
                cy = (row + 0.5) * spacing
                cx = np.clip(cx, r0, 1 - r0)
                cy = np.clip(cy, r0, 1 - r0)
                x[count] = cx
                y[count] = cy
                r[count] = r0
                count += 1
        return x, y, r
    
    def rect_init():
        """Rectangular grid initialization"""
        side = int(np.ceil(np.sqrt(N)))
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        spacing = 1.0 / side
        r0 = spacing * 0.48
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                x[count] = (i + 0.5) * spacing
                y[count] = (j + 0.5) * spacing
                r[count] = r0
                count += 1
        return x, y, r
    
    def random_init(seed):
        rng = np.random.RandomState(seed)
        x = rng.uniform(0.1, 0.9, N)
        y = rng.uniform(0.1, 0.9, N)
        r = rng.uniform(0.03, 0.08, N)
        return x, y, r
    
    def push_apart(x, y, r, iterations=1000, strength=0.5):
        """Physics-based repulsion to resolve overlaps"""
        x = x.copy()
        y = y.copy()
        r = r.copy()
        for it in range(iterations):
            if elapsed() > TIME_LIMIT - 5:
                break
            forces_x = np.zeros(N)
            forces_y = np.zeros(N)
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    s = r[i] + r[j]
                    if d < s and d > 1e-12:
                        overlap = s - d
                        fx = (dx / d) * overlap * strength
                        fy = (dy / d) * overlap * strength
                        forces_x[i] += fx
                        forces_y[i] += fy
                        forces_x[j] -= fx
                        forces_y[j] -= fy
            x += forces_x
            y += forces_y
            # Clamp to boundaries
            for i in range(N):
                x[i] = np.clip(x[i], r[i], 1 - r[i])
                y[i] = np.clip(y[i], r[i], 1 - r[i])
        return x, y, r
    
    def physics_optimize(x0, y0, r0, max_iters=5000):
        """Physics-based optimization: repel circles, grow radii"""
        x = x0.copy()
        y = y0.copy()
        r = r0.copy()
        dt = 0.01
        for it in range(max_iters):
            if elapsed() > TIME_LIMIT - 10:
                break
            # Compute forces
            fx = np.zeros(N)
            fy = np.zeros(N)
            for i in range(N):
                # Boundary forces
                if x[i] < r[i]:
                    fx[i] += (r[i] - x[i]) * 10
                if x[i] > 1 - r[i]:
                    fx[i] -= (x[i] - (1 - r[i])) * 10
                if y[i] < r[i]:
                    fy[i] += (r[i] - y[i]) * 10
                if y[i] > 1 - r[i]:
                    fy[i] -= (y[i] - (1 - r[i])) * 10
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    s = r[i] + r[j]
                    if d < s and d > 1e-12:
                        overlap = s - d
                        force = overlap * 5.0
                        fx[i] += (dx / d) * force
                        fy[i] += (dy / d) * force
                        fx[j] -= (dx / d) * force
                        fy[j] -= (dy / d) * force
            # Update positions
            x += dt * fx
            y += dt * fy
            # Try to grow radii slightly
            for i in range(N):
                mb = min(x[i], 1-x[i], y[i], 1-y[i])
                if r[i] < mb * 0.99:
                    r[i] += 0.0001
            # Clamp
            for i in range(N):
                x[i] = np.clip(x[i], 0, 1)
                y[i] = np.clip(y[i], 0, 1)
                r[i] = max(0.001, r[i])
        return x, y, r
    
    def penalty_objective(params, mu, start_time):
        if time.time() - start_time > TIME_LIMIT - 5:
            return 1e10
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        obj = -np.sum(r)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx*dx + dy*dy)
        min_dist = r[:, None] + r[None, :]
        diff = min_dist - dist
        smooth_max = 0.5 * (diff + np.sqrt(diff*diff + 1e-6))
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        overlap_pen = np.sum(smooth_max[mask])
        pen = 0.0
        for i in range(N):
            for val in [r[i] - x[i], x[i] - (1 - r[i]), r[i] - y[i], y[i] - (1 - r[i])]:
                pen += 0.5 * (val + np.sqrt(val*val + 1e-6))
        return obj + mu * (overlap_pen + pen)
    
    def penalty_optimize(x0, y0, r0, mu=100.0, maxiter=1000):
        params0 = np.concatenate([x0, y0, r0])
        try:
            res = minimize(penalty_objective, params0, args=(mu, start_time),
                          method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': 1e-12})
            x = res.x[:N]
            y = res.x[N:2*N]
            r = res.x[2*N:3*N]
            x, y, r = make_valid(x, y, r)
            return x, y, r
        except:
            return x0, y0, r0
    
    # Generate multiple initializations
    inits = []
    x, y, r = hex_init()
    inits.append((x, y, r))
    x, y, r = rect_init()
    inits.append((x, y, r))
    for seed in [42, 123, 456, 789, 1000]:
        x, y, r = random_init(seed)
        inits.append((x, y, r))
    
    best_x, best_y, best_r = None, None, None
    best_score = 0.0
    
    # Phase 1: Physics-based optimization from each init
    for i, (x0, y0, r0) in enumerate(inits):
        if elapsed() > TIME_LIMIT - 20:
            break
        x, y, r = physics_optimize(x0, y0, r0, max_iters=3000)
        x, y, r = make_valid(x, y, r)
        score = sum_r(r)
        if score > best_score:
            best_score = score
            best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    # Phase 2: Penalty-based refinement from best result
    if best_x is not None and elapsed() < TIME_LIMIT - 15:
        for mu in [50.0, 200.0, 1000.0]:
            if elapsed() > TIME_LIMIT - 10:
                break
            x, y, r = penalty_optimize(best_x, best_y, best_r, mu=mu, maxiter=1500)
            score = sum_r(r)
            if score > best_score:
                best_score = score
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    # Phase 3: Final push-apart + polish
    if elapsed() < TIME_LIMIT - 5 and best_x is not None:
        x, y, r = push_apart(best_x, best_y, best_r, iterations=2000, strength=0.3)
        x, y, r = make_valid(x, y, r)
        score = sum_r(r)
        if score > best_score:
            best_score = score
            best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        # Final tight penalty polish
        if elapsed() < TIME_LIMIT - 3:
            x, y, r = penalty_optimize(best_x, best_y, best_r, mu=5000.0, maxiter=2000)
            score = sum_r(r)
            if score > best_score:
                best_score = score
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    if best_x is None:
        # Fallback to grid
        x, y, r = rect_init()
        x, y, r = make_valid(x, y, r)
        best_x, best_y, best_r = x, y, r
    
    result = np.column_stack((best_x, best_y, best_r))
    return result
