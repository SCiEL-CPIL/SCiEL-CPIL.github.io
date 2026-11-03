"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

N = 21
DEADLINE = 115.0  # seconds, safety margin before 120s limit


def _is_valid(x, r, w, h, tol=1e-9):
    """Check validity: circles within bounds and non-overlapping."""
    n = len(x)
    # Bounds check
    for i in range(n):
        if x[i, 0] < r[i] - tol or x[i, 0] > w - r[i] + tol:
            return False
        if x[i, 1] < r[i] - tol or x[i, 1] > h - r[i] + tol:
            return False
    # Non-overlap check
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((x[i, 0] - x[j, 0])**2 + (x[i, 1] - x[j, 1])**2)
            if dist < r[i] + r[j] - tol:
                return False
    return True


def _successive_convex_relaxation(x0, r0, w, h, start_time, deadline):
    """
    SCR (Successive Convex Relaxation) for circle packing.
    
    The problem: maximize sum(r_i) subject to:
      - (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2 for all i < j
      - r_i <= x_i <= w - r_i
      - r_i <= y_i <= h - r_i
      - r_i > 0
    
    The non-convex constraint (x_i-x_j)^2+(y_i-y_j)^2 >= (r_i+r_j)^2
    can be linearized at the current point using the tangent plane.
    
    At current (x, r), the constraint becomes:
      2*(x_i-x_j)*(x_i-x_j_new) + 2*(y_i-y_j)*(y_i-y_j_new) >= (r_i+r_j)^2 + 2*(r_i+r_j)*(r_i_new+r_j_new - r_i - r_j)
    
    Actually, let's use a simpler formulation. The constraint is:
      (x_i-x_j)^2 + (y_i-y_j)^2 >= (r_i+r_j)^2
    
    Linearize at current point (x_k, r_k):
    Let d_i_j = x_i - x_j, e_i_j = y_i - y_j, s_i_j = r_i + r_j
    The constraint is d^2 + e^2 >= s^2
    Gradient w.r.t. (x, y, r):
      d/dx_i: 2*d, d/dx_j: -2*d
      d/dy_i: 2*e, d/dy_j: -2*e
      d/dr_i: -2*s, d/dr_j: -2*s
    
    Linearized constraint:
      2*d*(x_i_new - x_i) - 2*d*(x_j_new - x_j) + 2*e*(y_i_new - y_i) - 2*e*(y_j_new - y_j) - 2*s*(r_i_new - r_i) - 2*s*(r_j_new - r_j) >= 0
      i.e., 2*d*(x_i_new - x_j_new) + 2*e*(y_i_new - y_j_new) - 2*s*(r_i_new + r_j_new) >= 2*d*(x_i - x_j) + 2*e*(y_i - y_j) - 2*s*(r_i + r_j)
    
    Since at the current point d^2+e^2 >= s^2, the RHS = 2*(d^2+e^2-s^2) >= 0.
    
    So the linearized constraint is:
      d*(x_i_new - x_j_new) + e*(y_i_new - y_j_new) - s*(r_i_new + r_j_new) >= d*(x_i - x_j) + e*(y_i - y_j) - s*(r_i + r_j)
    
    Let's simplify: 
      d*x_i_new - d*x_j_new + e*y_i_new - e*y_j_new - s*r_i_new - s*r_j_new >= d*(x_i-x_j) + e*(y_i-y_j) - s*(r_i+r_j)
    
    This is a linear constraint in the variables (x_i_new, y_i_new, r_i_new).
    """
    n = len(x0)
    x = x0.copy()
    r = r0.copy()
    
    # Variable vector: [x_0, y_0, ..., x_{n-1}, y_{n-1}, r_0, ..., r_{n-1}]
    # Total variables: 3*n
    
    def build_linear_constraints(x_cur, r_cur):
        """Build linear constraints for the SCR subproblem."""
        # Variables: v = [x_0, y_0, ..., x_{n-1}, y_{n-1}, r_0, ..., r_{n-1}]
        # Index: circle i -> x at 2*i, y at 2*i+1, r at 2*n+i
        
        m = n * (n - 1) // 2  # number of pair constraints
        A = np.zeros((m, 3 * n))
        b = np.zeros(m)
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                d = x_cur[i, 0] - x_cur[j, 0]
                e = x_cur[i, 1] - x_cur[j, 1]
                s = r_cur[i] + r_cur[j]
                
                # Constraint: d*(x_i - x_j) + e*(y_i - y_j) - s*(r_i + r_j) >= d*(x_i-x_j) + e*(y_i-y_j) - s*(r_i+r_j)
                # LHS: d*x_i - d*x_j + e*y_i - e*y_j - s*r_i - s*r_j
                # RHS: d*(x_i-x_j) + e*(y_i-y_j) - s*(r_i+r_j)
                
                # For scipy LinearConstraint, we need A @ v >= b
                # A row: coefficients of v
                A[idx, 2*i] = d          # x_i
                A[idx, 2*i+1] = e        # y_i
                A[idx, 2*j] = -d         # x_j
                A[idx, 2*j+1] = -e       # y_j
                A[idx, 2*n+i] = -s       # r_i
                A[idx, 2*n+j] = -s       # r_j
                
                b[idx] = d * (x_cur[i, 0] - x_cur[j, 0]) + e * (x_cur[i, 1] - x_cur[j, 1]) - s * (r_cur[i] + r_cur[j])
                
                idx += 1
        
        return A, b
    
    def solve_subproblem(x_cur, r_cur):
        """Solve the LP subproblem: max sum(r) s.t. linear constraints + bounds."""
        # Variables: v = [x_0, y_0, ..., x_{n-1}, y_{n-1}, r_0, ..., r_{n-1}]
        # Objective: maximize sum(r_i) -> minimize -sum(r_i)
        c = np.zeros(3 * n)
        c[2*n:3*n] = -1.0  # minimize -sum(r)
        
        # Bounds:
        # r_i >= 0: lower bound 0
        # x_i: r_i <= x_i <= w - r_i -> this is coupled with r_i, tricky for LP
        # For simplicity in the LP, we use:
        # 0 <= x_i <= w
        # 0 <= y_i <= h
        # 0 <= r_i <= min(w, h)/2
        # And add linear constraints for the bounds involving r:
        #   x_i - r_i >= 0
        #   w - x_i - r_i >= 0
        #   y_i - r_i >= 0
        #   h - y_i - r_i >= 0
        
        # These are 4*n additional linear constraints.
        # Let's add them to A and b.
        
        A_pairs, b_pairs = build_linear_constraints(x_cur, r_cur)
        
        # Add bound constraints:
        # x_i - r_i >= 0
        # -x_i - r_i + w >= 0
        # y_i - r_i >= 0
        # -y_i - r_i + h >= 0
        
        A_bounds = np.zeros((4 * n, 3 * n))
        b_bounds = np.zeros(4 * n)
        
        for i in range(n):
            # x_i - r_i >= 0
            A_bounds[4*i, 2*i] = 1.0
            A_bounds[4*i, 2*n+i] = -1.0
            b_bounds[4*i] = 0.0
            
            # -x_i - r_i + w >= 0  =>  -x_i - r_i >= -w
            A_bounds[4*i+1, 2*i] = -1.0
            A_bounds[4*i+1, 2*n+i] = -1.0
            b_bounds[4*i+1] = -w
            
            # y_i - r_i >= 0
            A_bounds[4*i+2, 2*i+1] = 1.0
            A_bounds[4*i+2, 2*n+i] = -1.0
            b_bounds[4*i+2] = 0.0
            
            # -y_i - r_i + h >= 0  =>  -y_i - r_i >= -h
            A_bounds[4*i+3, 2*i+1] = -1.0
            A_bounds[4*i+3, 2*n+i] = -1.0
            b_bounds[4*i+3] = -h
        
        # Combine
        A_full = np.vstack([A_pairs, A_bounds])
        b_full = np.concatenate([b_pairs, b_bounds])
        
        # Variable bounds
        lb = np.zeros(3 * n)
        ub = np.zeros(3 * n)
        ub[:2*n] = np.concatenate([np.full(n, w), np.full(n, h)])
        ub[2*n:] = min(w, h) / 2.0
        
        # Solve LP using scipy
        # Use linprog
        from scipy.optimize import linprog
        result = linprog(c, A_ub=-A_full, b_ub=-b_full, bounds=list(zip(lb, ub)), method='highs')
        
        if not result.success:
            return None, None
        
        v = result.x
        x_new = v[:2*n].reshape(n, 2)
        r_new = v[2*n:3*n]
        
        return x_new, r_new
    
    # SCR loop
    best_x = x.copy()
    best_r = r.copy()
    best_sum = np.sum(r)
    
    max_iters = 50
    for it in range(max_iters):
        if time.time() - start_time > deadline - 5:
            break
        
        x_new, r_new = solve_subproblem(x, r)
        if x_new is None:
            break
        
        # Check if the new solution is valid
        if _is_valid(x_new, r_new, w, h, tol=1e-7):
            x = x_new.copy()
            r = r_new.copy()
            s = np.sum(r)
            if s > best_sum:
                best_sum = s
                best_x = x.copy()
                best_r = r.copy()
        else:
            # If not valid, try to fix by slightly reducing radii
            r_new = r_new * 0.999
            if _is_valid(x_new, r_new, w, h, tol=1e-7):
                x = x_new.copy()
                r = r_new.copy()
                s = np.sum(r)
                if s > best_sum:
                    best_sum = s
                    best_x = x.copy()
                    best_r = r.copy()
            else:
                break
        
        # Check convergence
        if np.max(np.abs(x - best_x)) < 1e-8 and np.max(np.abs(r - best_r)) < 1e-8:
            break
    
    return best_x, best_r


def _grid_initialization(cols, rows, w, h, n_circles):
    """Create grid-based initial positions and radii."""
    xs = np.linspace(w / (2*cols), w - w / (2*cols), cols) if cols > 1 else np.array([w/2])
    ys = np.linspace(h / (2*rows), h - h / (2*rows), rows) if rows > 1 else np.array([h/2])
    
    positions = []
    for j in range(rows):
        for i in range(cols):
            positions.append([xs[i], ys[j]])
    
    positions = positions[:n_circles]
    positions = np.array(positions, dtype=float)
    
    dx = w / cols if cols > 1 else w
    dy = h / rows if rows > 1 else h
    base_r = min(dx, dy) * 0.45
    
    radii = np.ones(n_circles) * base_r
    
    return positions, radii


def _polish(x, r, w, h, start_time, deadline):
    """Local search polishing: try to increase radii and adjust positions."""
    n = len(x)
    x = x.copy()
    r = r.copy()
    best_sum = np.sum(r)
    best_x = x.copy()
    best_r = r.copy()
    
    np.random.seed(42)
    
    max_iters = 2000
    for it in range(max_iters):
        if time.time() - start_time > deadline - 3:
            break
        
        i = np.random.randint(n)
        
        # Try to increase radius
        old_r = r[i]
        r[i] = old_r * 1.005
        
        if _is_valid(x, r, w, h, tol=1e-9):
            s = np.sum(r)
            if s > best_sum:
                best_sum = s
                best_x = x.copy()
                best_r = r.copy()
        else:
            # Try to move circle i to make room
            moved = False
            old_pos = x[i].copy()
            for attempt in range(10):
                dx = np.random.uniform(-0.01, 0.01)
                dy = np.random.uniform(-0.01, 0.01)
                x[i, 0] = np.clip(old_pos[0] + dx, r[i], w - r[i])
                x[i, 1] = np.clip(old_pos[1] + dy, r[i], h - r[i])
                if _is_valid(x, r, w, h, tol=1e-9):
                    moved = True
                    s = np.sum(r)
                    if s > best_sum:
                        best_sum = s
                        best_x = x.copy()
                        best_r = r.copy()
                    break
            if not moved:
                r[i] = old_r
                x[i] = old_pos
    
    return best_x, best_r


def construct():
    start_time = time.time()
    
    best_result = None
    best_sum = 0.0
    
    # Try multiple aspect ratios and grid layouts
    # Perimeter 4: 2*(w+h) = 4 => w + h = 2
    
    layouts = [(5, 5), (4, 6), (6, 4), (3, 7), (7, 3), (4, 5), (5, 4)]
    aspects = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0]
    
    for cols, rows in layouts:
        if cols * rows < N:
            continue
        
        for aspect in aspects:
            if time.time() - start_time > DEADLINE - 10:
                break
            
            h = 2.0 / (aspect + 1.0)
            w = aspect * h
            
            # Ensure w and h are reasonable
            if w < 0.1 or h < 0.1:
                continue
            
            x0, r0 = _grid_initialization(cols, rows, w, h, N)
            
            # Run SCR
            x_opt, r_opt = _successive_convex_relaxation(x0, r0, w, h, start_time, DEADLINE)
            
            if x_opt is not None and _is_valid(x_opt, r_opt, w, h, tol=1e-7):
                # Polish
                x_pol, r_pol = _polish(x_opt, r_opt, w, h, start_time, DEADLINE)
                s = np.sum(r_pol)
                if s > best_sum:
                    best_sum = s
                    best_result = np.column_stack([x_pol, r_pol])
            
            if time.time() - start_time > DEADLINE - 5:
                break
    
    if best_result is None:
        # Fallback: simple grid in unit square
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3])
        return np.array(cells)
    
    return best_result


if __name__ == "__main__":
    result = construct()
    print(f"Sum of radii: {np.sum(result[:, 2]):.6f}")
    print(f"Number of circles: {len(result)}")
