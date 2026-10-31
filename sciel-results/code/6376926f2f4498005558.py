import numpy as np
import scipy.optimize as opt
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_value = -np.inf
    best_solution = None
    
    # Helper to unpack solution
    def unpack(z):
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        return x, y, r

    # Objective function
    def objective(z):
        return -np.sum(z[2::3])

    # Objective gradient
    def objective_grad(z):
        grad = np.zeros_like(z)
        grad[2::3] = -1.0
        return grad

    # Constraints
    def constraints(z):
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        
        cons = []
        # Boundary constraints
        # x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda z, i=i: z[3*i] - z[3*i+2], 'jac': None}) # Placeholder, we'll use a vectorized approach or list
        # It's better to define a single constraint function that returns an array for SLSQP if possible, 
        # but SLSQP accepts a list of dicts. For speed, we can define the array of values and jacobians.
        
        # Let's use the list of dicts format for clarity, but it can be slow.
        # Better: Use a single 'ineq' constraint that returns a vector.
        # SciPy SLSQP supports 'ineq' returning a vector if 'jac' is a function returning a matrix.
        
        pass

    # Let's implement a vectorized constraint for SLSQP
    def constraint_funcs(z):
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        
        # Boundary constraints: 4 per circle
        c_bound = np.concatenate([
            x - r,
            1.0 - x - r,
            y - r,
            1.0 - y - r
        ])
        
        # Pairwise constraints
        c_pair = np.zeros((N*(N-1)//2,))
        idx = 0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dr = r[i] + r[j]
                c_pair[idx] = dx*dx + dy*dy - dr*dr
                idx += 1
                
        return np.concatenate([c_bound, c_pair])

    def constraint_jac(z):
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        
        J = np.zeros((4*N + N*(N-1)//2, 3*N))
        
        # Boundary Jacobians
        # Row 0..N-1: x - r. Grad: [1, 0, -1] at i
        for i in range(N):
            J[i, 3*i] = 1.0
            J[i, 3*i+2] = -1.0
            
        # Row N..2N-1: 1 - x - r. Grad: [-1, 0, -1] at i
        for i in range(N):
            J[N+i, 3*i] = -1.0
            J[N+i, 3*i+2] = -1.0
            
        # Row 2N..3N-1: y - r. Grad: [0, 1, -1] at i
        for i in range(N):
            J[2*N+i, 3*i+1] = 1.0
            J[2*N+i, 3*i+2] = -1.0
            
        # Row 3N..4N-1: 1 - y - r. Grad: [0, -1, -1] at i
        for i in range(N):
            J[3*N+i, 3*i+1] = -1.0
            J[3*N+i, 3*i+2] = -1.0
            
        # Pairwise Jacobians
        row = 4*N
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dr = r[i] + r[j]
                
                # d/dx_i
                J[row, 3*i] = 2.0 * dx
                # d/dy_i
                J[row, 3*i+1] = 2.0 * dy
                # d/dr_i
                J[row, 3*i+2] = -2.0 * dr
                
                # d/dx_j
                J[row, 3*j] = -2.0 * dx
                # d/dy_j
                J[row, 3*j+1] = -2.0 * dy
                # d/dr_j
                J[row, 3*j+2] = -2.0 * dr
                
                row += 1
                
        return J

    def run_optimization(init_z):
        try:
            res = opt.minimize(
                objective, 
                init_z, 
                method='SLSQP', 
                jac=objective_grad,
                constraints=[{'type': 'ineq', 'fun': constraint_funcs, 'jac': constraint_jac}],
                bounds=[(0, 1)] * (3*N),
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
            if res.success:
                val = -res.fun
                if val > best_value:
                    return res.x, val
            return None, -np.inf
        except Exception as e:
            return None, -np.inf

    # Initialization 1: Hexagonal Grid
    def hex_init():
        # Approximate hexagonal packing
        # Side length of grid
        side = int(np.ceil(np.sqrt(N * 2 / np.sqrt(3))))
        pts = []
        for row in range(side):
            offset = 0.5 if row % 2 == 1 else 0.0
            for col in range(side):
                x = (col + offset + 0.5) / side
                y = (row + 0.5) / side
                if x < 1.0 and y < 1.0:
                    pts.append([x, y])
        
        # If we have more than N, take first N? Or random N?
        # Better to scale them to fit well.
        if len(pts) >= N:
            # Randomly select N to break symmetry?
            # Or just take the first N in a deterministic order that covers the square.
            # Let's shuffle to avoid bias
            np.random.shuffle(pts)
            pts = pts[:N]
        else:
            # Pad with random
            while len(pts) < N:
                pts.append([np.random.rand(), np.random.rand()])
                
        z = np.zeros(3*N)
        for i, p in enumerate(pts):
            z[3*i] = p[0]
            z[3*i+1] = p[1]
            # Initial radius: estimate based on local density
            # Simply use a small radius to ensure feasibility
            z[3*i+2] = 0.05 
            
        return z

    # Initialization 2: Random with Repulsion
    def random_repel_init():
        x = np.random.rand(N)
        y = np.random.rand(N)
        
        # Simple repulsion steps
        for _ in range(50):
            forces_x = np.zeros(N)
            forces_y = np.zeros(N)
            for i in range(N):
                for j in range(N):
                    if i == j: continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d2 = dx*dx + dy*dy
                    if d2 < 0.01: # Threshold
                        d = np.sqrt(d2) if d2 > 1e-10 else 1e-5
                        f = 1.0 / (d2 + 1e-6)
                        forces_x[i] += f * dx / d
                        forces_y[i] += f * dy / d
            x += 0.01 * forces_x
            y += 0.01 * forces_y
            # Clip to [0,1]
            x = np.clip(x, 0.01, 0.99)
            y = np.clip(y, 0.01, 0.99)
            
        z = np.zeros(3*N)
        z[0::3] = x
        z[1::3] = y
        z[2::3] = 0.05
        return z

    # Run multiple restarts
    restarts = 0
    while time.time() < deadline:
        if restarts % 2 == 0:
            init_z = hex_init()
        else:
            init_z = random_repel_init()
            
        res_z, val = run_optimization(init_z)
        
        if res_z is not None:
            # Polish step: Fix centers, solve LP for max radii
            x = res_z[0::3]
            y = res_z[1::3]
            
            # Build LP
            c = -np.ones(N) # Maximize sum r -> Minimize -sum r
            A_ub = []
            b_ub = []
            
            # Boundary constraints: r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
            # This is 4 inequalities per circle, or just 1 if we take min.
            # LP requires linear inequalities.
            # r_i <= x_i
            # r_i <= 1 - x_i
            # r_i <= y_i
            # r_i <= 1 - y_i
            # r_i + r_j <= d_ij
            
            # A_ub shape (M, N)
            # r_i - x_i <= 0
            for i in range(N):
                row = np.zeros(N)
                row[i] = 1.0
                A_ub.append(row)
                b_ub.append(x[i])
                
                row = np.zeros(N)
                row[i] = 1.0
                A_ub.append(row)
                b_ub.append(1.0 - x[i])
                
                row = np.zeros(N)
                row[i] = 1.0
                A_ub.append(row)
                b_ub.append(y[i])
                
                row = np.zeros(N)
                row[i] = 1.0
                A_ub.append(row)
                b_ub.append(1.0 - y[i])
                
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    row = np.zeros(N)
                    row[i] = 1.0
                    row[j] = 1.0
                    A_ub.append(row)
                    b_ub.append(d)
                    
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            
            try:
                lp_res = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
                if lp_res.success:
                    r_lp = lp_res.x
                    val_lp = np.sum(r_lp)
                    if val_lp > val:
                        val = val_lp
                        # Update res_z radii
                        res_z[2::3] = r_lp
            
            except Exception:
                pass
            
            if val > best_value:
                best_value = val
                best_solution = res_z.copy()
                
        restarts += 1
        # Limit restarts to avoid infinite loop if time is not the only constraint
        if restarts > 50:
            break

    if best_solution is None:
        # Fallback to hex init if optimization failed
        z = hex_init()
        x = z[0::3]
        y = z[1::3]
        r = np.ones(N) * 0.05
        best_solution = np.column_stack((x, y, r))
        best_value = np.sum(r)

    return best_solution

