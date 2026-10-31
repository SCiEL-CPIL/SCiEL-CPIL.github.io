"""Circle packing in perimeter-4 rectangle: Unified SLSQP optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 145.0  # Safety margin before 150s limit

def construct():
    start_time = time.time()
    
    def unpack_vars(v):
        """
        v layout: [r_0..r_{N-1}, x_0..x_{N-1}, y_0..y_{N-1}, w]
        Returns r, x, y, w, h
        """
        r = v[:N]
        x = v[N:2*N]
        y = v[2*N:3*N]
        w = v[3*N]
        h = 2.0 - w
        return r, x, y, w, h

    def objective(v):
        r, x, y, w, h = unpack_vars(v)
        return -np.sum(r)

    def constraints(v):
        r, x, y, w, h = unpack_vars(v)
        cons = []
        
        # 1. Radius positivity
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - 1e-5})
            
        # 2. Boundary constraints
        # x_i >= r_i
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[N+i] - v[i]})
        # w - x_i >= r_i  =>  w - x_i - r_i >= 0
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*N] - v[N+i] - v[i]})
        # y_i >= r_i
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*N+i] - v[i]})
        # h - y_i >= r_i  =>  (2-w) - y_i - r_i >= 0
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: (2.0 - v[3*N]) - v[2*N+i] - v[i]})
            
        # 3. Rectangle validity
        # w > 0, h > 0
        cons.append({'type': 'ineq', 'fun': lambda v: v[3*N] - 0.2})
        cons.append({'type': 'ineq', 'fun': lambda v: (2.0 - v[3*N]) - 0.2})

        # 4. Pairwise non-overlap
        # (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
        for i in range(N):
            for j in range(i + 1, N):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: (
                        (v[N+i] - v[N+j])**2 + 
                        (v[2*N+i] - v[2*N+j])**2 - 
                        (v[i] + v[j])**2
                    )
                })
                
        return cons

    def generate_seed(w_init):
        """
        Generate a valid initial guess for circles in a rectangle of width w_init.
        Uses a grid layout.
        """
        h_init = 2.0 - w_init
        
        # Choose grid dimensions
        # Try to find cols/rows that fit N
        best_seed = None
        best_score = -1e9
        
        # Iterate over possible grid structures
        for cols in range(1, N + 1):
            rows = (N + cols - 1) // cols
            if cols * rows < N:
                continue
            
            cw = w_init / cols
            ch = h_init / rows
            # Max possible radius is half the min cell dimension
            r_est = 0.5 * min(cw, ch) * 0.95 # 5% shrink for safety
            
            x = np.zeros(N)
            y = np.zeros(N)
            r = np.zeros(N)
            
            idx = 0
            valid = True
            for j in range(rows):
                for i in range(cols):
                    if idx >= N:
                        break
                    cx = (i + 0.5) * cw
                    cy = (j + 0.5) * ch
                    x[idx] = cx
                    y[idx] = cy
                    r[idx] = r_est
                    idx += 1
            
            # Pad remaining circles if N is not divisible (shouldn't happen with ceil logic, but just in case)
            while idx < N:
                 x[idx] = w_init / 2.0
                 y[idx] = h_init / 2.0
                 r[idx] = 1e-4
                 idx += 1

            # Quick feasibility check
            ok = True
            for i in range(N):
                if x[i] < r[i] - 1e-9 or x[i] > w_init - r[i] + 1e-9:
                    ok = False
                    break
                if y[i] < r[i] - 1e-9 or y[i] > h_init - r[i] + 1e-9:
                    ok = False
                    break
            
            if ok:
                for i in range(N):
                    for j in range(i + 1, N):
                        d2 = (x[i] - x[j])**2 + (y[i] - y[j])**2
                        if d2 < (r[i] + r[j])**2 - 1e-9:
                            ok = False
                            break
                    if not ok:
                        break
            
            if ok:
                score = np.sum(r)
                if score > best_score:
                    best_score = score
                    best_seed = np.concatenate([r, x, y, [w_init]])
                    
        if best_seed is None:
            # Fallback: tiny circles in center
            r = np.full(N, 1e-4)
            x = np.linspace(w_init*0.1, w_init*0.9, N)
            y = np.full(N, h_init*0.5)
            best_seed = np.concatenate([r, x, y, [w_init]])
            
        return best_seed

    def run_slsqp(seed_v, maxiter=100, ftol=1e-8):
        """
        Run SLSQP optimization.
        """
        bounds = []
        # Radii
        for i in range(N):
            bounds.append((1e-5, 1.0)) # Max radius 1.0 is safe for perimeter 4
        # X coords
        for i in range(N):
            bounds.append((0.0, 2.0))
        # Y coords
        for i in range(N):
            bounds.append((0.0, 2.0))
        # Width
        bounds.append((0.4, 1.6)) # Constrain aspect ratio to reasonable range
        
        try:
            res = minimize(
                objective,
                seed_v,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': maxiter, 'ftol': ftol}
            )
            return res
        except Exception as e:
            return None

    def validate_and_score(v):
        """
        Check if solution is valid and return score.
        """
        r, x, y, w, h = unpack_vars(v)
        if w <= 0 or h <= 0:
            return -1e9, None
        if np.any(r < 1e-6):
            return -1e9, None
        if np.any(x < r - 1e-7) or np.any(x > w - r + 1e-7):
            return -1e9, None
        if np.any(y < r - 1e-7) or np.any(y > h - r + 1e-7):
            return -1e9, None
        for i in range(N):
            for j in range(i + 1, N):
                d2 = (x[i] - x[j])**2 + (y[i] - y[j])**2
                if d2 < (r[i] + r[j])**2 - 1e-7:
                    return -1e9, None
        return np.sum(r), v

    best_v = None
    best_score = -1e9

    # Phase 1: Explore aspect ratios with fixed w optimization
    # We will try a few aspect ratios to get good starting points
    aspect_ratios = [1.0, 1.2, 0.85, 1.3, 0.75]
    
    for ar in aspect_ratios:
        if time.time() - start_time > DEADLINE - 60:
            break
            
        w_init = 2.0 / (1.0 + ar)
        seed_v = generate_seed(w_init)
        
        # Run SLSQP with w fixed? No, SLSQP with w free is the goal.
        # But starting with w fixed in the seed is good.
        # We can just run the full SLSQP (with w free) from this seed.
        # To save time, we might run a quick SLSQP with w fixed first?
        # No, just run the full one. It's only 5 restarts.
        
        res = run_slsqp(seed_v, maxiter=150)
        if res is not None and res.success:
            score, v = validate_and_score(res.x)
            if score > best_score:
                best_score = score
                best_v = res.x

    # Phase 2: Refine the best solution found
    # If we have a good candidate, run a longer SLSQP to fine-tune
    if best_v is not None:
        if time.time() - start_time < DEADLINE - 10:
            # Run a more intensive optimization from the best point
            # We can increase maxiter and tighten ftol
            res_final = run_slsqp(best_v, maxiter=300, ftol=1e-10)
            if res_final is not None:
                score, v = validate_and_score(res_final.x)
                if score > best_score:
                    best_score = score
                    best_v = res_final.x

    # Fallback if nothing worked
    if best_v is None:
        # Trivial valid packing
        w = 1.0
        h = 1.0
        r = 1e-4
        x = np.linspace(r, w - r, N)
        y = np.full(N, h / 2.0)
        best_v = np.concatenate([r, x, y, [w]])
        best_score = np.sum(r)

    # Extract final result
    r, x, y, w, h = unpack_vars(best_v)
    
    # Final safety clamp
    r = np.maximum(r, 1e-6)
    x = np.clip(x, r, w - r)
    y = np.clip(y, r, h - r)
    
    # Re-check after clamping
    valid = True
    for i in range(N):
        if x[i] < r[i] - 1e-6 or x[i] > w - r[i] + 1e-6:
            valid = False
            break
        if y[i] < r[i] - 1e-6 or y[i] > h - r[i] + 1e-6:
            valid = False
            break
    if valid:
        for i in range(N):
            for j in range(i + 1, N):
                d2 = (x[i] - x[j])**2 + (y[i] - y[j])**2
                if d2 < (r[i] + r[j])**2 - 1e-6:
                    valid = False
                    break
            if not valid:
                break
    
    if not valid:
        # If clamping broke it, shrink radii
        r = r * 0.9
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)
        
    return np.column_stack((x, y, r))
