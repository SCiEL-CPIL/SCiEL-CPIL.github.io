"""Rewritten program: constrained nonlinear optimization for circle packing in perimeter-4 rectangle."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21
DEADLINE = 150.0  # seconds, evaluator kills at 150s; use 145s as safety margin

def construct():
    start_time = time.time()
    
    # We want to pack N=21 circles in a rectangle of perimeter 4.
    # Perimeter = 2*(w+h) = 4 => w+h = 2.
    # We can parameterize by w (width), then h = 2 - w.
    # To maximize symmetry and likely optimal packing, let's start with square w=h=1.
    # But we can also optimize w. For simplicity and robustness, let's fix w=1, h=1 first,
    # or allow w to vary. Let's try w=1, h=1 (square) as it's a natural starting point.
    # Actually, the problem says "variable rectangle of perimeter 4", so w+h=2.
    # Let's parameterize the rectangle by w, with h=2-w. We'll optimize w as well.
    
    # Variables: x_i, y_i, r_i for i in range(N), and w.
    # Objective: maximize sum(r_i) => minimize -sum(r_i)
    # Constraints:
    # 1. 0 <= x_i <= w
    # 2. 0 <= y_i <= h = 2 - w
    # 3. r_i > 0 (we'll enforce r_i >= epsilon)
    # 4. (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2 for all i < j
    # 5. w > 0, h > 0 => 0 < w < 2
    
    # Total variables: 3*N + 1 = 64
    # This is a large NLP. SLSQP might struggle. Let's try.
    # Alternatively, we can fix w=1 and just optimize positions and radii.
    # Let's try fixing w=1, h=1 first for a simpler problem.
    
    w = 1.0
    h = 1.0
    
    # Initial guess: grid layout
    side = int(np.ceil(np.sqrt(N)))
    x0 = []
    for i in range(side):
        for j in range(side):
            if len(x0) >= N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.9  # slightly smaller to be feasible
            x0.extend([x, y, r])
    
    # Pad if needed
    while len(x0) < 3 * N:
        x0.extend([0.5, 0.5, 0.01])
    
    x0 = np.array(x0[:3*N])
    
    def objective(v):
        r = v[2::3]
        return -np.sum(r)
    
    def constraints(v):
        cons = []
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        
        # Wall constraints: 0 <= x_i <= w, 0 <= y_i <= h
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - r[i] * 0 + 0})  # x_i >= 0
            # Actually, let's do it properly:
            # x_i - r_i >= 0? No, x_i >= 0 and x_i <= w.
            # But the circle must be inside: x_i - r_i >= 0 and x_i + r_i <= w.
            # Similarly for y.
        
        # Let's redefine constraints properly.
        cons = []
        for i in range(N):
            # x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]})
            # w - x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: w - v[3*i] - v[3*i+2]})
            # y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]})
            # h - y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: h - v[3*i+1] - v[3*i+2]})
            # r_i >= epsilon
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+2] - 1e-4})
        
        # Circle-circle constraints: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
        for i in range(N):
            for j in range(i+1, N):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: (v[3*i]-v[3*j])**2 + (v[3*i+1]-v[3*j+1])**2 - (v[3*i+2]+v[3*j+2])**2
                })
        
        return cons
    
    # This has 4*N + N*(N-1)/2 = 84 + 210 = 294 constraints.
    # SLSQP may be too slow. Let's try a simpler approach:
    # Use a penalty method or just optimize positions with fixed radii, then expand.
    
    # Alternative strategy:
    # 1. Start with a grid of circles with small radii.
    # 2. Use local search / gradient-based optimization to move circles and increase radii.
    # 3. Use SLSQP on a smaller problem or with fewer constraints.
    
    # Let's try a different approach: optimize positions and radii with a penalty for overlaps.
    # This avoids the large number of hard constraints.
    
    def penalty_objective(v, penalty=100.0):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        
        obj = -np.sum(r)
        
        # Wall violations
        for i in range(N):
            obj += penalty * max(0, r[i] - x[i])**2
            obj += penalty * max(0, r[i] - y[i])**2
            obj += penalty * max(0, x[i] + r[i] - w)**2
            obj += penalty * max(0, y[i] + r[i] - h)**2
        
        # Circle-circle violations
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    obj += penalty * overlap**2
        
        return obj
    
    # Use Nelder-Mead or Powell for the penalty objective (no gradient needed).
    # But N=21 => 63 variables. Nelder-Mead might be slow but let's try.
    
    # Actually, let's use a simpler local search:
    # 1. Initialize with grid.
    # 2. Iteratively try to increase radii and move circles to reduce overlaps.
    
    # Let's try a direct approach with SLSQP but with a smaller penalty or fewer constraints.
    # Or use scipy.optimize.minimize with method='SLSQP' and the hard constraints.
    
    # Given time constraints, let's try the penalty approach with Powell's method.
    
    result = minimize(penalty_objective, x0, method='Powell', 
                      options={'maxiter': 1000, 'xtol': 1e-6, 'ftol': 1e-6})
    
    if result.success or result.fun < 0:  # Check if we got a reasonable result
        v = result.x
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        
        # Ensure feasibility: clamp positions and reduce radii if needed
        for i in range(N):
            r[i] = min(r[i], x[i], w - x[i], y[i], h - y[i])
            r[i] = max(r[i], 1e-4)
        
        # Check circle-circle constraints and reduce radii if overlapping
        for _ in range(10):  # Iterate to resolve overlaps
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    max_r = dist / 2
                    if r[i] + r[j] > dist:
                        # Reduce both radii
                        factor = dist / (r[i] + r[j])
                        r[i] *= factor
                        r[j] *= factor
                        r[i] = max(r[i], 1e-4)
                        r[j] = max(r[j], 1e-4)
        
        # Build result
        circles = np.column_stack([x, y, r])
        
        # Check deadline
        if time.time() - start_time > 140:
            return circles
        
        # Try to improve with local search
        for _ in range(50):  # Iterations
            if time.time() - start_time > 145:
                break
            
            improved = False
            for i in range(N):
                # Try small moves
                for dx in [-0.01, 0, 0.01]:
                    for dy in [-0.01, 0, 0.01]:
                        new_x = x[i] + dx
                        new_y = y[i] + dy
                        if new_x < r[i] or new_x > w - r[i] or new_y < r[i] or new_y > h - r[i]:
                            continue
                        
                        # Check if this move allows increasing r[i]
                        # Try increasing r[i] slightly
                        new_r = r[i] + 0.001
                        feasible = True
                        for j in range(N):
                            if i == j:
                                continue
                            dist = np.sqrt((new_x - x[j])**2 + (new_y - y[j])**2)
                            if dist < new_r + r[j]:
                                feasible = False
                                break
                        
                        if feasible and new_r <= min(new_x, w - new_x, new_y, h - new_y):
                            x[i] = new_x
                            y[i] = new_y
                            r[i] = new_r
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        circles = np.column_stack([x, y, r])
        return circles
    else:
        # Fallback: return initial grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cells.append([((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)])
        return np.array(cells)

# Test
if __name__ == "__main__":
    result = construct()
    print(result)
    print("Sum of radii:", np.sum(result[:, 2]))
