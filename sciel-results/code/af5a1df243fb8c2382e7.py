"""Circle packing in perimeter-4 rectangle: DE + Contact-Graph Local Search."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 21
_TIME_LIMIT = 148.0  # Safety margin before 150s kill


def _is_valid(x, y, r, w, h, tol=1e-9):
    """Check validity of circles in a w x h rectangle."""
    if w <= 0.05 or h <= 0.05:
        return False
    for i in range(len(r)):
        if r[i] <= 0:
            return False
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return False
    for i in range(len(r)):
        for j in range(i + 1, len(r)):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx * dx + dy * dy
            r_sum = r[i] + r[j]
            if d2 < r_sum * r_sum - tol:
                return False
    return True


def _grid_init(w, h, n_circles):
    """Generate grid-based initial positions and radii."""
    # Try several grid configurations
    best = None
    best_score = -1
    for cols in range(1, n_circles + 1):
        rows = int(np.ceil(n_circles / cols))
        if rows * cols < n_circles:
            continue
        cw = w / cols
        ch = h / rows
        r_cell = min(cw, ch) / 2.0 - 1e-6
        score = np.pi * r_cell * r_cell * n_circles / (w * h)
        if score > best_score:
            best_score = score
            best = (cols, rows, cw, ch, r_cell)
    if best is None:
        cols = int(np.ceil(np.sqrt(n_circles)))
        rows = int(np.ceil(n_circles / cols))
        cw = w / cols
        ch = h / rows
        r_cell = min(cw, ch) / 2.0 - 1e-6
        best = (cols, rows, cw, ch, r_cell)
    cols, rows, cw, ch, r_cell = best
    x = np.zeros(n_circles)
    y = np.zeros(n_circles)
    r = np.zeros(n_circles)
    for i in range(n_circles):
        row, col = divmod(i, cols)
        x[i] = (col + 0.5) * cw
        y[i] = (row + 0.5) * ch
        r[i] = r_cell
    return x, y, r


def _de_search(w, h, x0, y0, r0, max_time=20.0, start_time=None):
    """
    Differential Evolution search over positions and radii for fixed w,h.
    Objective: -sum(r) + large penalty for invalidity.
    """
    if start_time is None:
        start_time = time.time()
    
    n = N
    def objective(params):
        if time.time() - start_time > max_time:
            return 1e6
        x = params[0:n]
        y = params[n:2*n]
        r = params[2*n:3*n]
        
        # Hard validity check with tolerance
        valid = _is_valid(x, y, r, w, h, tol=1e-7)
        if not valid:
            # Compute violation magnitude for gradient-like feedback
            viol = 0.0
            for i in range(n):
                viol += max(0, r[i] - x[i]) + max(0, x[i] + r[i] - w)
                viol += max(0, r[i] - y[i]) + max(0, y[i] + r[i] - h)
            for i in range(n):
                for j in range(i+1, n):
                    dx = x[i]-x[j]; dy = y[i]-y[j]
                    d = np.sqrt(dx*dx+dy*dy)
                    viol += max(0, r[i]+r[j]-d)
            return -np.sum(r) + 1000.0 * viol + 10.0 * viol*viol
        
        return -np.sum(r)
    
    # Bounds: x in [0, w], y in [0, h], r in [0, min(w,h)/2]
    r_max = min(w, h) / 2.0 - 1e-6
    bounds = [(0, w)] * n + [(0, h)] * n + [(0, r_max)] * n
    
    try:
        result = differential_evolution(
            objective, bounds,
            seed=42, maxiter=200, popsize=15,
            tol=1e-10, polish=True,
            init='sobol', mutation=(0.5, 1.0), recombination=0.7
        )
        return result.x, result.fun
    except Exception:
        return None, 1e6


def _contact_local_search(x, y, r, w, h, max_time=15.0, start_time=None):
    """
    Local search on the contact graph: only move circles that are in contact.
    Try small perturbations to increase total radius while maintaining validity.
    """
    if start_time is None:
        start_time = time.time()
    
    n = N
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    def build_contact_graph(x, y, r, tol=1e-5):
        contacts = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i]-x[j]; dy = y[i]-y[j]
                d = np.sqrt(dx*dx+dy*dy)
                if d < r[i]+r[j] + tol:
                    contacts[i,j] = contacts[j,i] = True
        return contacts
    
    for iteration in range(50):
        if time.time() - start_time > max_time:
            break
        
        contacts = build_contact_graph(x, y, r)
        improved = False
        
        # Try to expand each circle slightly and adjust neighbors
        for i in range(n):
            if time.time() - start_time > max_time:
                break
            
            # Try expanding circle i
            dr = 0.001
            new_r = r.copy()
            new_r[i] += dr
            
            # Check if expansion is possible without moving
            if new_r[i] <= x[i] and new_r[i] <= w - x[i] and \
               new_r[i] <= y[i] and new_r[i] <= h - y[i]:
                # Check overlaps with non-contact circles
                ok = True
                for j in range(n):
                    if i == j:
                        continue
                    dx = x[i]-x[j]; dy = y[i]-y[j]
                    d = np.sqrt(dx*dx+dy*dy)
                    if d < new_r[i] + r[j] - 1e-8:
                        ok = False
                        break
                if ok:
                    r = new_r
                    improved = True
                    continue
            
            # Try moving circle i slightly in random directions
            for _ in range(10):
                angle = np.random.uniform(0, 2*np.pi)
                step = 0.002
                new_x = x.copy()
                new_y = y.copy()
                new_x[i] += step * np.cos(angle)
                new_y[i] += step * np.sin(angle)
                
                # Check validity
                if new_x[i] < r[i] or new_x[i] > w - r[i] or \
                   new_y[i] < r[i] or new_y[i] > h - r[i]:
                    continue
                
                valid = True
                for j in range(n):
                    if i == j:
                        continue
                    dx = new_x[i]-x[j]; dy = new_y[i]-y[j]
                    d = np.sqrt(dx*dx+dy*dy)
                    if d < r[i]+r[j] - 1e-8:
                        valid = False
                        break
                if valid:
                    x = new_x
                    y = new_y
                    improved = True
                    break
        
        if not improved:
            break
    
    return x, y, r


def _slsqp_polish(x, y, r, w, h, max_time=10.0, start_time=None):
    """
    Final SLSQP polish with penalty method for fine-tuning.
    """
    if start_time is None:
        start_time = time.time()
    
    n = N
    z0 = np.concatenate([x, y, r])
    
    def objective(z):
        x_ = z[0:n]
        y_ = z[n:2*n]
        r_ = z[2*n:3*n]
        
        obj = -np.sum(r_)
        
        # Penalty for violations
        viol = 0.0
        for i in range(n):
            viol += max(0, r_[i] - x_[i])**2 + max(0, x_[i] + r_[i] - w)**2
            viol += max(0, r_[i] - y_[i])**2 + max(0, y_[i] + r_[i] - h)**2
        for i in range(n):
            for j in range(i+1, n):
                dx = x_[i]-x_[j]; dy = y_[i]-y_[j]
                d = np.sqrt(dx*dx+dy*dy)
                viol += max(0, r_[i]+r_[j]-d)**2
        
        return obj + 5000.0 * viol
    
    bounds = [(0, w)] * n + [(0, h)] * n + [(0, min(w,h)/2)] * n
    
    try:
        res = minimize(objective, z0, method='SLSQP', bounds=bounds,
                       options={'maxiter': 300, 'ftol': 1e-14, 'maxls': 100})
        x_opt = res.x[0:n]
        y_opt = res.x[n:2*n]
        r_opt = res.x[2*n:3*n]
        return x_opt, y_opt, r_opt
    except Exception:
        return x, y, r


def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT
    
    best_value = 0.0
    best_result = None
    
    def check_and_update(x, y, r, w, h):
        nonlocal best_value, best_result
        if _is_valid(x, y, r, w, h):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w, h)
    
    # Scan aspect ratios
    # Perimeter = 2(w+h) = 4 => w+h = 2
    aspect_ratios = np.linspace(0.4, 1.6, 13)  # w from 0.4 to 1.6
    
    for w0 in aspect_ratios:
        if time.time() > deadline - 30:
            break
        
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue
        
        # Grid initialization
        x, y, r = _grid_init(w0, h0, N)
        check_and_update(x, y, r, w0, h0)
        
        # DE search
        if time.time() > deadline - 25:
            break
        
        x_de, fun_de = _de_search(w0, h0, x, y, r, max_time=15.0, start_time=start_time)
        if x_de is not None:
            x_de = x_de[0:N]
            y_de = x_de[N:2*N]
            r_de = x_de[2*N:3*N]
            check_and_update(x_de, y_de, r_de, w0, h0)
            
            # Contact local search
            if time.time() > deadline - 15:
                break
            x_ls, y_ls, r_ls = _contact_local_search(x_de, y_de, r_de, w0, h0,
                                                      max_time=8.0, start_time=start_time)
            check_and_update(x_ls, y_ls, r_ls, w0, h0)
            
            # SLSQP polish
            if time.time() > deadline - 8:
                break
            x_sp, y_sp, r_sp = _slsqp_polish(x_ls, y_ls, r_ls, w0, h0,
                                              max_time=5.0, start_time=start_time)
            check_and_update(x_sp, y_sp, r_sp, w0, h0)
    
    # If we have time, do a focused search on the best aspect ratio
    if best_result is not None and time.time() < deadline - 15:
        x_best, y_best, r_best, w_best, h_best = best_result
        
        # Try a few random restarts around the best configuration
        rng = np.random.default_rng(123)
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            
            # Perturb the best solution
            x_p = x_best + rng.normal(0, 0.01, N)
            y_p = y_best + rng.normal(0, 0.01, N)
            r_p = r_best * (1 + rng.normal(0, 0.02, N))
            r_p = np.clip(r_p, 0.01, min(w_best, h_best)/2 - 1e-6)
            x_p = np.clip(x_p, 0, w_best)
            y_p = np.clip(y_p, 0, h_best)
            
            check_and_update(x_p, y_p, r_p, w_best, h_best)
            
            # Quick DE polish
            x_q, fun_q = _de_search(w_best, h_best, x_p, y_p, r_p,
                                     max_time=5.0, start_time=start_time)
            if x_q is not None:
                x_q = x_q[0:N]
                y_q = x_q[N:2*N]
                r_q = x_q[2*N:3*N]
                check_and_update(x_q, y_q, r_q, w_best, h_best)
    
    if best_result is None:
        # Fallback: simple grid in unit square
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    x, y, r, w, h = best_result
    
    # Final safety: ensure strict validity with minimal shrink
    for _ in range(20):
        r_trial = r * (1 - 1e-8)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break
    
    return np.column_stack((x, y, r))
