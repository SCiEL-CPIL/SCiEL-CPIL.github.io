"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    def elapsed():
        return time.time() - start_time
    
    def time_left():
        return deadline - elapsed()
    
    # Objective: maximize sum of radii
    # Variables: x[0:N], y[N:2N], r[2N:3N]
    def objective(v):
        return -np.sum(v[2*N:3*N])
    
    def objective_grad(v):
        g = np.zeros(3*N)
        g[2*N:3*N] = -1.0
        return g
    
    # Constraints:
    # 1. Boundary: x_i >= r_i, x_i <= 1-r_i, y_i >= r_i, y_i <= 1-r_i
    # 2. Pairwise: (x_i-x_j)^2 + (y_i-y_j)^2 >= (r_i+r_j)^2
    # 3. r_i >= eps
    
    def make_constraints():
        cons = []
        
        # Boundary constraints (linear)
        # x_i - r_i >= 0
        for i in range(N):
            A = np.zeros(3*N)
            A[i] = 1.0
            A[2*N+i] = -1.0
            cons.append({'type': 'ineq', 'fun': lambda v, A=A: A @ v, 'jac': lambda v, A=A: A})
        
        # x_i + r_i <= 1  =>  1 - x_i - r_i >= 0
        for i in range(N):
            A = np.zeros(3*N)
            A[i] = -1.0
            A[2*N+i] = -1.0
            cons.append({'type': 'ineq', 'fun': lambda v, A=A: 1.0 - A @ v, 'jac': lambda v, A=A: -A})
        
        # y_i - r_i >= 0
        for i in range(N):
            A = np.zeros(3*N)
            A[N+i] = 1.0
            A[2*N+i] = -1.0
            cons.append({'type': 'ineq', 'fun': lambda v, A=A: A @ v, 'jac': lambda v, A=A: A})
        
        # y_i + r_i <= 1  =>  1 - y_i - r_i >= 0
        for i in range(N):
            A = np.zeros(3*N)
            A[N+i] = -1.0
            A[2*N+i] = -1.0
            cons.append({'type': 'ineq', 'fun': lambda v, A=A: 1.0 - A @ v, 'jac': lambda v, A=A: -A})
        
        # Pairwise non-overlap: (x_i-x_j)^2 + (y_i-y_j)^2 - (r_i+r_j)^2 >= 0
        for i in range(N):
            for j in range(i+1, N):
                def pair_fun(v, i=i, j=j):
                    dx = v[i] - v[j]
                    dy = v[N+i] - v[N+j]
                    dr = v[2*N+i] + v[2*N+j]
                    return dx*dx + dy*dy - dr*dr
                
                def pair_jac(v, i=i, j=j):
                    g = np.zeros(3*N)
                    dx = v[i] - v[j]
                    dy = v[N+i] - v[N+j]
                    dr = v[2*N+i] + v[2*N+j]
                    g[i] = 2*dx
                    g[j] = -2*dx
                    g[N+i] = 2*dy
                    g[N+j] = -2*dy
                    g[2*N+i] = -2*dr
                    g[2*N+j] = -2*dr
                    return g
                
                cons.append({'type': 'ineq', 'fun': pair_fun, 'jac': pair_jac})
        
        # r_i >= eps
        eps = 1e-6
        for i in range(N):
            A = np.zeros(3*N)
            A[2*N+i] = 1.0
            cons.append({'type': 'ineq', 'fun': lambda v, A=A: A @ v - eps, 'jac': lambda v, A=A: A})
        
        return cons
    
    def is_valid(v):
        x = v[0:N]
        y = v[N:2*N]
        r = v[2*N:3*N]
        
        # Check boundaries
        if np.any(x < r - 1e-8) or np.any(x > 1 - r + 1e-8):
            return False
        if np.any(y < r - 1e-8) or np.any(y > 1 - r + 1e-8):
            return False
        if np.any(r < 1e-8):
            return False
        
        # Check pairwise
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < r[i] + r[j] - 1e-8:
                    return False
        return True
    
    def grid_init(seed=0):
        rng = np.random.RandomState(seed)
        side = int(np.ceil(np.sqrt(N)))
        v = np.zeros(3*N)
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                v[count] = (i + 0.5) / side + rng.uniform(-0.02, 0.02)
                v[N+count] = (j + 0.5) / side + rng.uniform(-0.02, 0.02)
                v[2*N+count] = 0.5/side - 0.001
                count += 1
        # Clip to valid range
        v[0:N] = np.clip(v[0:N], 0.01, 0.99)
        v[N:2*N] = np.clip(v[N:2*N], 0.01, 0.99)
        v[2*N:3*N] = np.clip(v[2*N:3*N], 0.005, 0.49)
        return v
    
    def random_init(seed=0):
        rng = np.random.RandomState(seed)
        v = np.zeros(3*N)
        v[0:N] = rng.uniform(0.1, 0.9, N)
        v[N:2*N] = rng.uniform(0.1, 0.9, N)
        v[2*N:3*N] = rng.uniform(0.02, 0.15, N)
        return v
    
    def polish(v0, maxiter=200):
        constraints = make_constraints()
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0))  # x
        for i in range(N):
            bounds.append((0.0, 1.0))  # y
        for i in range(N):
            bounds.append((1e-6, 0.5))  # r
        
        try:
            res = minimize(objective, v0, jac=objective_grad, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': maxiter, 'ftol': 1e-12})
            if res.success and is_valid(res.x):
                return res.x
        except Exception:
            pass
        return None
    
    best_v = None
    best_val = -np.inf
    
    # Try multiple restarts
    max_restarts = 15
    for restart in range(max_restarts):
        if time_left() < 30:
            break
        
        if restart == 0:
            v0 = grid_init(seed=0)
        elif restart == 1:
            v0 = grid_init(seed=1)
        elif restart == 2:
            v0 = random_init(seed=42)
        elif restart == 3:
            v0 = random_init(seed=123)
        else:
            # Perturb best solution
            if best_v is not None:
                rng = np.random.RandomState(restart * 100 + 7)
                v0 = best_v + rng.normal(0, 0.02, 3*N)
                v0[0:N] = np.clip(v0[0:N], 0.01, 0.99)
                v0[N:2*N] = np.clip(v0[N:2*N], 0.01, 0.99)
                v0[2*N:3*N] = np.clip(v0[2*N:3*N], 0.005, 0.49)
            else:
                v0 = random_init(seed=restart)
        
        result = polish(v0, maxiter=150)
        if result is not None:
            val = np.sum(result[2*N:3*N])
            if val > best_val:
                best_val = val
                best_v = result.copy()
    
    # Final high-precision polish on best solution
    if best_v is not None and time_left() > 15:
        result = polish(best_v, maxiter=500)
        if result is not None:
            val = np.sum(result[2*N:3*N])
            if val > best_val:
                best_val = val
                best_v = result.copy()
    
    if best_v is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    x = best_v[0:N]
    y = best_v[N:2*N]
    r = best_v[2*N:3*N]
    
    return np.column_stack([x, y, r])

# EVOLVE-BLOCK-END
