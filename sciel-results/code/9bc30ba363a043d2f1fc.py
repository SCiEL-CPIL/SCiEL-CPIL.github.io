"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import linprog, differential_evolution, minimize

N = 21
PERIMETER = 4.0
START = time.time()
DEADLINE = START + 145.0

def _solve_lp(positions, W, H):
    """
    Solve the Linear Program to maximize sum of radii for fixed positions.
    positions: (N, 2) array of (x, y)
    W, H: dimensions of the rectangle
    Returns: (sum_radii, radii_array)
    """
    n = len(positions)
    
    # Objective: minimize -sum(r_i)
    c = -np.ones(n)
    
    # Constraints:
    # 1. Wall constraints:
    #    r_i <= x_i
    #    r_i <= W - x_i
    #    r_i <= y_i
    #    r_i <= H - y_i
    # 2. Overlap constraints:
    #    r_i + r_j <= dist(i, j)
    
    # We construct A_ub and b_ub
    # Number of constraints: 4*N + N*(N-1)/2
    n_cons = 4 * n + n * (n - 1) // 2
    A_ub = np.zeros((n_cons, n))
    b_ub = np.zeros(n_cons)
    
    idx = 0
    # Wall constraints
    for i in range(n):
        x, y = positions[i]
        # r_i <= x
        A_ub[idx, i] = 1.0
        b_ub[idx] = x
        idx += 1
        # r_i <= W - x
        A_ub[idx, i] = 1.0
        b_ub[idx] = W - x
        idx += 1
        # r_i <= y
        A_ub[idx, i] = 1.0
        b_ub[idx] = y
        idx += 1
        # r_i <= H - y
        A_ub[idx, i] = 1.0
        b_ub[idx] = H - y
        idx += 1
        
    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            d = np.sqrt(dx * dx + dy * dy)
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = d
            idx += 1
            
    # Bounds: r_i >= 0
    bounds = [(0, None)] * n
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return -res.fun, res.x
        else:
            return -float('inf'), np.zeros(n)
    except:
        return -float('inf'), np.zeros(n)

def _objective(pos_flat, W, H):
    """
    Objective function for minimization.
    pos_flat: flattened array of positions (2N elements)
    Returns: -sum_radii
    """
    positions = pos_flat.reshape(-1, 2)
    sum_r, _ = _solve_lp(positions, W, H)
    return -sum_r

def construct():
    global DEADLINE
    t0 = time.time()
    DEADLINE = t0 + 145.0
    
    best_sum = -1.0
    best_positions = None
    best_radii = None
    best_W = 1.0
    best_H = 1.0
    
    # Aspect ratios W where W + H = 2
    # The optimal aspect ratio for packing circles is often close to 1, 
    # but can vary. We search a range.
    W_values = np.linspace(0.6, 1.4, 7)
    
    for W in W_values:
        if time.time() > DEADLINE:
            break
        H = 2.0 - W
        
        # Define bounds for differential evolution
        # x in [0, W], y in [0, H]
        bounds = []
        for i in range(N):
            bounds.append((0, W))
            bounds.append((0, H))
            
        # Initial guess for DE is random within bounds, but we can provide a better start?
        # DE handles initialization.
        
        # Run Differential Evolution
        # maxiter and popsize control the number of function evaluations
        # N=21, so 42 variables.
        # popsize=10 -> 420 individuals.
        # maxiter=50 -> 21000 evaluations? That might be too many if LP is slow.
        # Let's estimate LP time. ~1-5ms. 21000 * 5ms = 105s. Too slow.
        # We need to be more conservative.
        
        # Let's use a smaller popsize and maxiter, or use Nelder-Mead directly.
        # Nelder-Mead is much more efficient per iteration for this problem.
        
        # Strategy:
        # 1. Run DE with small budget to find a good basin.
        # 2. Refine with Nelder-Mead.
        
        de_result = None
        try:
            de_result = differential_evolution(_objective, bounds, args=(W, H), 
                                               maxiter=20, popsize=5, 
                                               tol=0.01, seed=42)
        except:
            pass
            
        if de_result is not None and de_result.success:
            x0 = de_result.x
        else:
            # Fallback initial guess: Grid
            cols = 5
            rows = 5
            xs = np.linspace(W/6, W - W/6, cols)
            ys = np.linspace(H/6, H - H/6, rows)
            init_pos = []
            for i in range(cols):
                for j in range(rows):
                    if len(init_pos) < N:
                        init_pos.append([xs[i], ys[j]])
            x0 = np.array(init_pos).flatten()
            
        # Refine with Nelder-Mead
        try:
            res = minimize(_objective, x0, args=(W, H), method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
            
            sum_r, radii = _solve_lp(res.x.reshape(-1, 2), W, H)
            
            if sum_r > best_sum:
                best_sum = sum_r
                best_positions = res.x.reshape(-1, 2)
                best_radii = radii
                best_W = W
                best_H = H
                
        except Exception as e:
            pass
            
    # If no valid solution found (should not happen), use a simple grid
    if best_positions is None:
        W = 1.0
        H = 1.0
        cols = 5
        rows = 5
        xs = np.linspace(W/6, W - W/6, cols)
        ys = np.linspace(H/6, H - H/6, rows)
        best_positions = []
        for i in range(cols):
            for j in range(rows):
                if len(best_positions) < N:
                    best_positions.append([xs[i], ys[j]])
        best_positions = np.array(best_positions)
        _, best_radii = _solve_lp(best_positions, W, H)
        best_W = W
        best_H = H
        
    # Construct final array
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = [best_positions[i][0], best_positions[i][1], best_radii[i]]
        
    return result

if __name__ == "__main__":
    res = construct()
    print(res.shape)
    print(res[:3])
