"""Combined program: Joint optimization of rectangle dimensions and circle packing."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def construct():
    deadline = time.time() + 110.0  # Safety margin before 150s limit
    best_result = None
    best_score = -np.inf

    # Helper to unpack variables: [w, x1, y1, r1, x2, y2, r2, ...]
    def unpack(vars_):
        w = vars_[0]
        h = 2.0 - w
        rest = vars_[1:]
        x = rest[0::3]
        y = rest[1::3]
        r = rest[2::3]
        return w, h, x, y, r

    def penalty_function(vars_, penalty_factor=1000.0):
        w, h, x, y, r = unpack(vars_)
        
        # If dimensions are invalid, return huge penalty
        if w <= 0.1 or h <= 0.1:
            return 1e10
            
        penalty = 0.0
        
        # Wall constraints (circle must be inside)
        # x - r >= 0, w - x - r >= 0, y - r >= 0, h - y - r >= 0
        penalty += penalty_factor * np.sum(np.maximum(0, r - x) ** 2)
        penalty += penalty_factor * np.sum(np.maximum(0, r - (w - x)) ** 2)
        penalty += penalty_factor * np.sum(np.maximum(0, r - y) ** 2)
        penalty += penalty_factor * np.sum(np.maximum(0, r - (h - y)) ** 2)
        
        # Negative radius
        penalty += penalty_factor * np.sum(np.maximum(0, -r) ** 2)
        
        # Overlap constraints
        # Vectorized overlap check is complex, so we loop for clarity in penalty, 
        # but DE calls this many times. Let's try to vectorize or keep it efficient.
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                overlap = np.maximum(0, r[i] + r[j] - dist)
                penalty += penalty_factor * overlap ** 2
        
        return penalty - np.sum(r)

    def objective_constrained(vars_):
        w, h, x, y, r = unpack(vars_)
        return -np.sum(r)

    def make_constraints():
        def cons_dimensions(vars_):
            w = vars_[0]
            h = 2.0 - w
            # Ensure w and h are reasonable
            return np.array([w - 0.1, h - 0.1])

        def cons_inside(vars_):
            w, h, x, y, r = unpack(vars_)
            return np.concatenate([
                x - r,
                w - x - r,
                y - r,
                h - y - r
            ])

        def cons_non_overlap(vars_):
            w, h, x, y, r = unpack(vars_)
            cons = []
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    cons.append(dist - r[i] - r[j])
            return np.array(cons)

        def cons_positive_r(vars_):
            w, h, x, y, r = unpack(vars_)
            return r

        return [
            {'type': 'ineq', 'fun': cons_dimensions},
            {'type': 'ineq', 'fun': cons_inside},
            {'type': 'ineq', 'fun': cons_non_overlap},
            {'type': 'ineq', 'fun': cons_positive_r}
        ]

    def make_bounds():
        bounds = []
        # w is the first variable. 
        # Perimeter 4 => 2(w+h)=4 => h=2-w.
        # If w=1, h=1 (square). 
        # If w=0.5, h=1.5 (aspect 1:3).
        # If w=1.5, h=0.5 (aspect 3:1).
        # Let's allow w in [0.3, 1.7] to cover aspect ratios 1:5 to 5:1.
        bounds.append((0.3, 1.7))
        
        for i in range(N):
            # x, y, r
            # x, y can be anywhere in [0, 2] initially, constraints will pull them in.
            bounds.append((0.0, 2.0))
            bounds.append((0.0, 2.0))
            bounds.append((0.0, 1.0))
        return bounds

    def validate(w, h, x, y, r, tol=1e-7):
        if w < 0.1 or h < 0.1:
            return False
        if np.any(r < -tol):
            return False
        if np.any(x < -tol) or np.any(x > w + tol):
            return False
        if np.any(y < -tol) or np.any(y > h + tol):
            return False
        # Check circle-wall distance
        if np.any(x - r < -tol) or np.any(w - x - r < -tol):
            return False
        if np.any(y - r < -tol) or np.any(h - y - r < -tol):
            return False
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < r[i] + r[j] - tol:
                    return False
        return True

    def update_best(w, h, x, y, r):
        nonlocal best_result, best_score
        if validate(w, h, x, y, r):
            # Clip small numerical errors
            x = np.clip(x, 0, w)
            y = np.clip(y, 0, h)
            r = np.clip(r, 0, None)
            
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_result = np.column_stack([x, y, r])
                return True
        return False

    # Strategy 1: Differential Evolution for Global Search
    # DE is robust and can handle the variable dimension w.
    # We use a penalty function for the objective in DE.
    
    bounds = make_bounds()
    
    # Set random seed for reproducibility if needed, but DE is stochastic.
    # We will run DE with a reasonable number of iterations.
    # 64 variables. Pop size ~ 15*64 = 960? That's too big for 150s if penalty is slow.
    # Let's try popsize=10, maxiter=50.
    
    try:
        if time.time() < deadline - 30:
            de_result = differential_evolution(
                penalty_function,
                bounds,
                seed=42,
                maxiter=50,
                popsize=10,
                tol=1e-6,
                update_interval=10,
                polish=False, # We will polish manually
                disp=False
            )
            
            if de_result.success or de_result.fun < 0: # If it found a negative penalty (meaning sum r > penalty)
                w_de, h_de, x_de, y_de, r_de = unpack(de_result.x)
                update_best(w_de, h_de, x_de, y_de, r_de)
                
                # Use DE result as initial guess for SLSQP
                x0 = de_result.x
            else:
                # Fallback initial guess if DE fails
                w0 = 1.0
                h0 = 1.0
                # Simple grid
                n_cols = 5
                n_rows = 5
                xs = np.zeros(N)
                ys = np.zeros(N)
                rs = np.zeros(N)
                for i in range(N):
                    col = i % n_cols
                    row = i // n_cols
                    xs[i] = (col + 0.5) * w0 / n_cols
                    ys[i] = (row + 0.5) * h0 / n_rows
                    rs[i] = 0.4 * min(w0/n_cols, h0/n_rows)
                
                x0 = np.zeros(3 * N + 1)
                x0[0] = w0
                for i in range(N):
                    x0[1 + 3*i] = xs[i]
                    x0[2 + 3*i] = ys[i]
                    x0[3 + 3*i] = rs[i]
                
                # Initial validation
                w_chk, h_chk, x_chk, y_chk, r_chk = unpack(x0)
                update_best(w_chk, h_chk, x_chk, y_chk, r_chk)
                
    except Exception as e:
        # Fallback if DE crashes
        w0 = 1.0
        h0 = 1.0
        xs = np.random.uniform(0.1, 0.9, N)
        ys = np.random.uniform(0.1, 0.9, N)
        rs = np.full(N, 0.05)
        x0 = np.zeros(3 * N + 1)
        x0[0] = w0
        for i in range(N):
            x0[1 + 3*i] = xs[i]
            x0[2 + 3*i] = ys[i]
            x0[3 + 3*i] = rs[i]
        w_chk, h_chk, x_chk, y_chk, r_chk = unpack(x0)
        update_best(w_chk, h_chk, x_chk, y_chk, r_chk)

    # Strategy 2: SLSQP Refinement
    # SLSQP is precise but local. We use the best result from DE (or fallback) as start.
    # We might need multiple restarts if SLSQP gets stuck.
    
    constraints = make_constraints()
    
    # Try refining the current best
    if best_result is not None:
        # We need to reconstruct x0 from best_result and a guessed w?
        # Actually, we should keep the full vector x0 from the last successful update or DE.
        # Let's re-run SLSQP on the best vector found so far.
        
        # We need to store the full vector (including w) for the best result.
        # The update_best function only stores x,y,r. We need w,h to reconstruct x0.
        # Let's modify update_best to store the full vector or just w.
        
        # Re-doing update_best logic to store full vector
        best_full_vector = None
        
        def update_best_full(w, h, x, y, r):
            nonlocal best_result, best_score, best_full_vector
            if validate(w, h, x, y, r):
                x = np.clip(x, 0, w)
                y = np.clip(y, 0, h)
                r = np.clip(r, 0, None)
                
                score = np.sum(r)
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack([x, y, r])
                    # Reconstruct full vector
                    vec = np.zeros(3 * N + 1)
                    vec[0] = w
                    for i in range(N):
                        vec[1 + 3*i] = x[i]
                        vec[2 + 3*i] = y[i]
                        vec[3 + 3*i] = r[i]
                    best_full_vector = vec
                    return True
            return False

        # Re-evaluate best_full_vector if we didn't capture it in the first pass
        # Since we used the old update_best, we might not have best_full_vector.
        # Let's just use the DE result if it was better, or create a new one.
        
        # To be safe, let's run SLSQP from the DE result specifically, and also from a few random starts.
        
        # Refine DE result
        if 'de_result' in locals() and de_result.x is not None:
            try:
                if time.time() < deadline - 10:
                    slsqp_result = minimize(
                        objective_constrained, 
                        de_result.x, 
                        method='SLSQP',
                        constraints=constraints, 
                        bounds=bounds,
                        options={'maxiter': 500, 'ftol': 1e-10, 'disp': False}
                    )
                    w_sl, h_sl, x_sl, y_sl, r_sl = unpack(slsqp_result.x)
                    update_best_full(w_sl, h_sl, x_sl, y_sl, r_sl)
            except Exception:
                pass

        # If we still don't have a good best_full_vector, try a few random SLSQP starts
        if best_full_vector is None:
            # Generate a few diverse initial guesses
            starts = []
            # Square
            starts.append(1.0)
            # Rectangle 1:2
            starts.append(1.33)
            # Rectangle 2:1
            starts.append(0.66)
            
            for w_init in starts:
                if time.time() > deadline - 5:
                    break
                
                h_init = 2.0 - w_init
                # Random positions inside
                xs = np.random.uniform(0.1, w_init - 0.1, N)
                ys = np.random.uniform(0.1, h_init - 0.1, N)
                # Small radii
                rs = np.full(N, 0.05)
                
                x0 = np.zeros(3 * N + 1)
                x0[0] = w_init
                for i in range(N):
                    x0[1 + 3*i] = xs[i]
                    x0[2 + 3*i] = ys[i]
                    x0[3 + 3*i] = rs[i]
                
                try:
                    if time.time() < deadline - 5:
                        slsqp_result = minimize(
                            objective_constrained, 
                            x0, 
                            method='SLSQP',
                            constraints=constraints, 
                            bounds=bounds,
                            options={'maxiter': 300, 'ftol': 1e-9, 'disp': False}
                        )
                        w_sl, h_sl, x_sl, y_sl, r_sl = unpack(slsqp_result.x)
                        update_best_full(w_sl, h_sl, x_sl, y_sl, r_sl)
                except Exception:
                    continue

    # Strategy 3: Final Polish
    # If we have a best_full_vector, do one more high-precision SLSQP pass
    if best_full_vector is not None and time.time() < deadline - 5:
        try:
            slsqp_result = minimize(
                objective_constrained, 
                best_full_vector, 
                method='SLSQP',
                constraints=constraints, 
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False}
            )
            w_sl, h_sl, x_sl, y_sl, r_sl = unpack(slsqp_result.x)
            update_best_full(w_sl, h_sl, x_sl, y_sl, r_sl)
        except Exception:
            pass

    # Fallback if everything failed
    if best_result is None:
        # Simple grid in square
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3))
        return np.array(cells)

    return best_result
# EVOLVE-BLOCK-END
