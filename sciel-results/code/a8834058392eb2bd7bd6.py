import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_val = -np.inf
    best_result = None  # Stores (x, y, r, W, H)
    
    def evaluate(x, y, r, W, H):
        """
        Check validity and return sum of radii if valid, else -inf.
        x, y, r: arrays of shape (N,)
        W, H: floats
        """
        # Check bounds
        if W <= 0 or H <= 0:
            return -np.inf
        
        for i in range(N):
            if x[i] - r[i] < -1e-7:
                return -np.inf
            if x[i] + r[i] > W + 1e-7:
                return -np.inf
            if y[i] - r[i] < -1e-7:
                return -np.inf
            if y[i] + r[i] > H + 1e-7:
                return -np.inf
            if r[i] <= 1e-6:
                return -np.inf
                
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < r[i] + r[j] - 1e-7:
                    return -np.inf
                    
        return np.sum(r)

    def softplus(x, beta=100.0):
        """Numerically stable softplus."""
        x_beta = beta * x
        # np.where(x_beta > 0, x + log(1+exp(-x_beta))/beta, log(1+exp(x_beta))/beta)
        # To avoid overflow in exp, we use the log1p trick carefully
        # For large positive x_beta: log1p(exp(-x_beta)) is small, result ~ x
        # For large negative x_beta: log1p(exp(x_beta)) is small, result ~ log(1+exp(x_beta))/beta ~ 0
        result = np.zeros_like(x_beta, dtype=float)
        pos_mask = x_beta > 0
        neg_mask = ~pos_mask
        
        # pos: x + log(1 + exp(-x_beta)) / beta
        result[pos_mask] = x_beta[pos_mask] / beta + np.log1p(np.exp(-x_beta[pos_mask])) / beta
        
        # neg: log(1 + exp(x_beta)) / beta
        result[neg_mask] = np.log1p(np.exp(x_beta[neg_mask])) / beta
        
        return result

    def softplus_grad(x, beta=100.0):
        """Gradient of softplus: sigmoid(beta*x)."""
        # sigmoid(x) = 1 / (1 + exp(-x))
        # For large negative x, exp(-x) is large, sigmoid ~ 0
        # For large positive x, exp(-x) is small, sigmoid ~ 1
        # Use stable sigmoid calculation
        x_beta = beta * x
        sig = np.zeros_like(x_beta, dtype=float)
        pos_mask = x_beta > 0
        neg_mask = ~pos_mask
        
        # pos: 1 / (1 + exp(-x_beta))
        sig[pos_mask] = 1.0 / (1.0 + np.exp(-x_beta[pos_mask]))
        
        # neg: exp(x_beta) / (1 + exp(x_beta))
        exp_x = np.exp(x_beta[neg_mask])
        sig[neg_mask] = exp_x / (1.0 + exp_x)
        
        return sig

    def objective_and_grad(params, beta=100.0):
        """
        params: [x0, y0, r0, ..., xN-1, yN-1, rN-1, W]
        Returns (objective_value, gradient_array)
        """
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        W = params[3*N]
        H = 2.0 - W
        
        obj = -np.sum(r)
        grad = np.zeros_like(params)
        
        # Boundary penalties
        # Constraints:
        # x_i >= r_i  -> violation: r_i - x_i
        # x_i <= W - r_i -> violation: x_i + r_i - W
        # y_i >= r_i  -> violation: r_i - y_i
        # y_i <= H - r_i -> violation: y_i + r_i - H
        # r_i > 0 -> violation: -r_i
        
        for i in range(N):
            xi = x[i]
            yi = y[i]
            ri = r[i]
            
            # 1. x_i >= r_i
            v1 = ri - xi
            obj += softplus(v1, beta)
            g1 = softplus_grad(v1, beta)
            # d/dxi (softplus(ri - xi)) = -g1
            # d/dri (softplus(ri - xi)) = +g1
            grad[3*i] += -g1
            grad[3*i + 2] += g1
            
            # 2. x_i <= W - r_i  =>  x_i + r_i - W <= 0
            v2 = xi + ri - W
            obj += softplus(v2, beta)
            g2 = softplus_grad(v2, beta)
            # d/dxi = g2
            # d/dri = g2
            # d/dW = -g2
            grad[3*i] += g2
            grad[3*i + 2] += g2
            grad[3*N] += -g2  # Gradient w.r.t W
            
            # 3. y_i >= r_i
            v3 = ri - yi
            obj += softplus(v3, beta)
            g3 = softplus_grad(v3, beta)
            grad[3*i + 1] += -g3
            grad[3*i + 2] += g3
            
            # 4. y_i <= H - r_i  =>  y_i + r_i - H <= 0
            # Note: H = 2 - W, so dH/dW = -1
            # Constraint: y_i + r_i - (2 - W) <= 0 => y_i + r_i + W - 2 <= 0
            v4 = yi + ri + W - 2.0
            obj += softplus(v4, beta)
            g4 = softplus_grad(v4, beta)
            # d/dyi = g4
            # d/dri = g4
            # d/dW = g4
            grad[3*i + 1] += g4
            grad[3*i + 2] += g4
            grad[3*N] += g4  # Gradient w.r.t W
            
            # 5. r_i > 0
            v5 = -ri
            obj += softplus(v5, beta)
            g5 = softplus_grad(v5, beta)
            # d/dri = -g5
            grad[3*i + 2] += -g5
            
            # Objective term -r_i
            grad[3*i + 2] += -1.0

        # Pairwise overlap penalties
        # Constraint: dist(i,j) >= ri + rj
        # Violation: ri + rj - dist(i,j)
        
        for i in range(N):
            for j in range(i + 1, N):
                xi, yi, ri = x[i], y[i], r[i]
                xj, yj, rj = x[j], y[j], r[j]
                
                dx = xi - xj
                dy = yi - yj
                d = np.sqrt(dx*dx + dy*dy + 1e-12) # small epsilon for stability
                
                v = ri + rj - d
                obj += softplus(v, beta)
                g = softplus_grad(v, beta)
                
                # Gradients of v = ri + rj - d
                # d/dxi = -dx/d
                # d/dyi = -dy/d
                # d/dri = 1
                # d/dxj = +dx/d
                # d/dyj = +dy/d
                # d/drj = 1
                
                # Chain rule: grad_obj += g * grad_v
                
                grad[3*i] += -g * (dx / d)
                grad[3*i + 1] += -g * (dy / d)
                grad[3*i + 2] += g
                
                grad[3*j] += g * (dx / d)
                grad[3*j + 1] += g * (dy / d)
                grad[3*j + 2] += g

        return obj, grad

    def hex_lattice_init(W, H):
        """
        Generate hexagonal lattice initialization.
        Returns (x, y, r) arrays.
        """
        # Try a few aspect ratios implicitly by just using the given W, H
        # We need to fit 21 circles.
        # Hex packing: rows of circles, offset every other row.
        
        # Estimate radius
        # If we have R rows, vertical spacing is r*sqrt(3).
        # Horizontal spacing is 2r.
        
        # Let's try configurations of rows [n1, n2, ...] summing to 21.
        # Common configs for 21:
        # 5 rows: 3,4,4,4,3 (sum 18? no) -> 3+4+4+4+3 = 18. Need 21.
        # 4 rows: 5,5,5,6? 
        # Let's just use a simple grid first to get a valid start, then optimize.
        # Actually, a simple grid is safer for initialization to avoid invalid starts.
        
        # Simple Grid Init
        # Divide W and H into a grid that fits 21 items.
        # 5 cols, 5 rows (25 spots) -> remove 4 corners?
        # Or 4 cols, 6 rows (24 spots) -> remove 3?
        
        # Let's use 5x5 grid, remove 4 corners.
        # This leaves 21 circles.
        # Positions:
        # (1,1) to (3,3) plus edges?
        # Indices 0..4 for cols, 0..4 for rows.
        # Remove (0,0), (0,4), (4,0), (4,4).
        
        rows, cols = 5, 5
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        r0 = np.zeros(N)
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if (i == 0 and j == 0) or (i == 0 and j == cols-1) or (i == rows-1 and j == 0) or (i == rows-1 and j == cols-1):
                    continue
                if idx >= N:
                    break
                x0[idx] = (j + 0.5) * W / cols
                y0[idx] = (i + 0.5) * H / rows
                r0[idx] = min(W / cols, H / rows) * 0.45 # Conservative radius
                idx += 1
        
        return x0, y0, r0

    def run_optimization(x0, y0, r0, W0, beta=100.0, max_iter=300):
        """
        Run L-BFGS-B optimization.
        """
        params0 = np.concatenate([x0, y0, r0, [W0]])
        
        # Bounds
        # x, y: [0, 2] is safe, but tighter is better for optimization?
        # Let's use [0, 2] to be safe, penalties will push them in.
        # r: [1e-6, 1.0]
        # W: [0.5, 1.5]
        
        bounds = []
        for i in range(N):
            bounds.append((0.0, 2.0)) # x
        for i in range(N):
            bounds.append((0.0, 2.0)) # y
        for i in range(N):
            bounds.append((1e-6, 1.0)) # r
        bounds.append((0.5, 1.5)) # W
        
        try:
            res = minimize(
                lambda p: objective_and_grad(p, beta),
                params0,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            return res.x
        except Exception:
            return params0

    # Initial configurations to try
    # Varying aspect ratios W
    W_inits = [0.8, 0.9, 1.0, 1.1, 1.2]
    
    for W_init in W_inits:
        if time.time() > deadline:
            break
            
        H_init = 2.0 - W_init
        
        x0, y0, r0 = hex_lattice_init(W_init, H_init)
        
        # Phase 1: Coarse optimization
        params1 = run_optimization(x0, y0, r0, W_init, beta=50.0, max_iter=200)
        
        if time.time() > deadline - 20:
             # Check if this result is valid and good enough to save
             x1, y1, r1, W1 = params1[:N], params1[N:2*N], params1[2*N:3*N], params1[3*N]
             H1 = 2.0 - W1
             val = evaluate(x1, y1, r1, W1, H1)
             if val > best_val:
                 best_val = val
                 best_result = (x1, y1, r1, W1, H1)
             continue

        # Phase 2: Fine optimization (Polish)
        # Use higher beta for sharper constraints
        params2 = run_optimization(params1[:N], params1[N:2*N], params1[2*N:3*N], params1[3*N], beta=200.0, max_iter=300)
        
        x2, y2, r2, W2 = params2[:N], params2[N:2*N], params2[2*N:3*N], params2[3*N]
        H2 = 2.0 - W2
        
        val = evaluate(x2, y2, r2, W2, H2)
        if val > best_val:
            best_val = val
            best_result = (x2, y2, r2, W2, H2)
            
        # Phase 3: Ultra-fine polish if time permits
        if time.time() < deadline - 5:
             params3 = run_optimization(params2[:N], params2[N:2*N], params2[2*N:3*N], params2[3*N], beta=500.0, max_iter=200)
             x3, y3, r3, W3 = params3[:N], params3[N:2*N], params3[2*N:3*N], params3[3*N]
             H3 = 2.0 - W3
             val3 = evaluate(x3, y3, r3, W3, H3)
             if val3 > best_val:
                 best_val = val3
                 best_result = (x3, y3, r3, W3, H3)

    if best_result is None:
        # Fallback: simple grid
        x, y, r, W, H = hex_lattice_init(1.0, 1.0)
        # Ensure validity for fallback
        # Just return a valid small grid
        side = 5
        x = np.array([(j + 0.5) * 1.0 / side for i in range(side) for j in range(side) if not ((i==0 and j==0) or (i==0 and j==4) or (i==4 and j==0) or (i==4 and j==4))][:N])
        y = np.array([(i + 0.5) * 1.0 / side for i in range(side) for j in range(side) if not ((i==0 and j==0) or (i==0 and j==4) or (i==4 and j==0) or (i==4 and j==4))][:N])
        r = np.full(N, 0.1)
        return np.column_stack((x, y, r))

    x, y, r, W, H = best_result
    return np.column_stack((x, y, r))

