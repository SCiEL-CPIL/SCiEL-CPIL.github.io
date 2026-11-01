"""Circle packing in unit square n=32 - Joint optimization of positions and radii."""
import time
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint

N = 32
DEADLINE = 145.0  # safety margin before 150s kill


def _objective(x):
    """Objective: minimize negative sum of radii."""
    # x is flattened [x1, y1, r1, x2, y2, r2, ..., xN, yN, rN]
    radii = x[2::3]
    return -np.sum(radii)


def _objective_grad(x):
    """Gradient of objective."""
    n = len(x) // 3
    grad = np.zeros_like(x)
    # d(-sum(r))/dr_i = -1
    grad[2::3] = -1.0
    return grad


def _nonlinear_constraints(x, n):
    """
    Non-overlap constraints:
    For all i < j: ||p_i - p_j||^2 >= (r_i + r_j)^2
    => (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    """
    positions = x[0:2*n].reshape(n, 2)
    radii = x[2::3]
    
    cons = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dr = radii[i] + radii[j]
            cons.append(dx*dx + dy*dy - dr*dr)
    return np.array(cons)


def _nonlinear_grad(x, n):
    """
    Jacobian of nonlinear constraints.
    Constraint k corresponds to pair (i, j).
    f_k = dx^2 + dy^2 - (r_i + r_j)^2
    df_k/dx_i = 2*dx
    df_k/dy_i = 2*dy
    df_k/dr_i = -2*(r_i + r_j)
    df_k/dx_j = -2*dx
    df_k/dy_j = -2*dy
    df_k/dr_j = -2*(r_i + r_j)
    """
    positions = x[0:2*n].reshape(n, 2)
    radii = x[2::3]
    
    # Number of constraints is n*(n-1)/2
    num_cons = n * (n - 1) // 2
    jac = np.zeros((num_cons, 3 * n))
    
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dr = radii[i] + radii[j]
            
            # Indices in x:
            # x_i: 3*i, y_i: 3*i+1, r_i: 3*i+2
            # x_j: 3*j, y_j: 3*j+1, r_j: 3*j+2
            
            jac[k, 3*i] = 2 * dx
            jac[k, 3*i + 1] = 2 * dy
            jac[k, 3*i + 2] = -2 * dr
            
            jac[k, 3*j] = -2 * dx
            jac[k, 3*j + 1] = -2 * dy
            jac[k, 3*j + 2] = -2 * dr
            
            k += 1
            
    return jac


def _linear_constraints(x, n):
    """
    Boundary constraints:
    x_i >= r_i  => x_i - r_i >= 0
    1 - x_i >= r_i => -x_i - r_i >= -1
    y_i >= r_i
    1 - y_i >= r_i
    r_i >= 0
    """
    # We can use LinearConstraint(A, lb, ub)
    # But it's easier to just bound the variables in minimize if possible.
    # However, x_i - r_i >= 0 couples variables.
    
    # Let's define A such that A @ x >= lb
    # Variables: x1, y1, r1, x2, y2, r2, ...
    
    # For each i:
    # 1. x_i - r_i >= 0
    # 2. -x_i - r_i >= -1
    # 3. y_i - r_i >= 0
    # 4. -y_i - r_i >= -1
    # 5. r_i >= 0
    
    num_cons = 5 * n
    A = np.zeros((num_cons, 3 * n))
    lb = np.zeros(num_cons)
    
    for i in range(n):
        xi = 3 * i
        yi = 3 * i + 1
        ri = 3 * i + 2
        
        # 1. x_i - r_i >= 0
        A[5*i, xi] = 1.0
        A[5*i, ri] = -1.0
        lb[5*i] = 0.0
        
        # 2. -x_i - r_i >= -1
        A[5*i + 1, xi] = -1.0
        A[5*i + 1, ri] = -1.0
        lb[5*i + 1] = -1.0
        
        # 3. y_i - r_i >= 0
        A[5*i + 2, yi] = 1.0
        A[5*i + 2, ri] = -1.0
        lb[5*i + 2] = 0.0
        
        # 4. -y_i - r_i >= -1
        A[5*i + 3, yi] = -1.0
        A[5*i + 3, ri] = -1.0
        lb[5*i + 3] = -1.0
        
        # 5. r_i >= 0
        A[5*i + 4, ri] = 1.0
        lb[5*i + 4] = 0.0
        
    return A, lb


def _optimize_joint(positions, radii, time_budget):
    """Optimize joint positions and radii using trust-constr."""
    n = N
    x0 = np.zeros(3 * n)
    x0[0:2*n:3] = positions[:, 0]
    x0[1:2*n:3] = positions[:, 1]
    x0[2::3] = radii
    
    # Bounds for variables to help the solver
    # x, y in [0, 1]
    # r in [0, 0.5] (max radius in unit square)
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((0.0, 0.5))  # r
    
    A, lb = _linear_constraints(None, n)
    
    # Nonlinear constraint: g(x) >= 0
    # trust-constr expects g(x) >= 0 for NonlinearConstraint(g, lb, ub)
    # So lb=0, ub=inf
    
    nl_cons = NonlinearConstraint(
        lambda x: _nonlinear_constraints(x, n),
        0.0,
        np.inf,
        jac=lambda x: _nonlinear_grad(x, n)
    )
    
    lin_cons = LinearConstraint(
        A,
        lb,
        np.inf
    )
    
    start_time = time.time()
    
    try:
        result = minimize(
            _objective,
            x0,
            method='trust-constr',
            jac=_objective_grad,
            bounds=bounds,
            constraints=[nl_cons, lin_cons],
            options={'maxiter': 500, 'gtol': 1e-8, 'xtol': 1e-10, 'verbose': 0}
        )
        
        if result.success or result.fun < _objective(x0):
            best_x = result.x
        else:
            best_x = x0
            
    except Exception:
        best_x = x0
    
    # Extract positions and radii
    best_positions = np.zeros((n, 2))
    best_radii = np.zeros(n)
    best_positions[:, 0] = best_x[0:2*n:3]
    best_positions[:, 1] = best_x[1:2*n:3]
    best_radii = best_x[2::3]
    
    return best_positions, best_radii


def _grid_init(n, seed=0):
    """Grid initialization."""
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(n)))
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < n:
                x = (i + 0.5) / side + rng.randn() * 0.01
                y = (j + 0.5) / side + rng.randn() * 0.01
                positions.append([x, y])
    positions = np.array(positions[:n])
    # Initial radii: small
    radii = np.full(n, 0.05)
    return positions, radii


def _hex_init(n, seed=0):
    """Hexagonal initialization."""
    rng = np.random.RandomState(seed)
    r = 1.0 / 6.5
    dx = 2 * r
    dy = np.sqrt(3) * r
    positions = []
    for row in range(8):
        for col in range(8):
            if len(positions) < n:
                x = dx * (col + 0.5 * (row % 2)) + 0.08 + rng.randn() * 0.005
                y = dy * row + 0.08 + rng.randn() * 0.005
                positions.append([x, y])
    positions = np.array(positions[:n])
    radii = np.full(n, r * 0.9)
    return positions, radii


def _random_init(n, seed=0):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    positions = rng.rand(n, 2) * 0.8 + 0.1
    radii = np.full(n, 0.05)
    return positions, radii


def _fix_validity(positions, radii):
    """Ensure strict validity by shrinking radii if necessary."""
    n = len(positions)
    # Boundary fix
    for i in range(n):
        max_r = min(positions[i, 0], 1 - positions[i, 0],
                    positions[i, 1], 1 - positions[i, 1])
        radii[i] = min(radii[i], max_r)
        radii[i] = max(0, radii[i])
        
    # Overlap fix
    for _ in range(10):
        fixed = True
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(positions[i] - positions[j])
                min_dist = radii[i] + radii[j]
                if d < min_dist - 1e-9:
                    fixed = False
                    # Shrink both radii
                    scale = max(0, (d - 1e-8)) / max(min_dist, 1e-12)
                    radii[i] *= scale
                    radii[j] *= scale
        if fixed:
            break
            
    return radii


def construct():
    """Construct 32 disjoint circles in [0,1]^2 maximizing sum of radii."""
    start_time = time.time()
    
    best_positions = None
    best_radii = None
    best_sum = -np.inf
    
    def update_best(pos, rad):
        nonlocal best_positions, best_radii, best_sum
        rad = _fix_validity(pos, rad)
        s = np.sum(rad)
        if s > best_sum:
            best_sum = s
            best_positions = pos.copy()
            best_radii = rad.copy()
            
    # Initializations
    inits = [
        lambda: _grid_init(N, seed=42),
        lambda: _hex_init(N, seed=42),
        lambda: _random_init(N, seed=1),
        lambda: _random_init(N, seed=2),
        lambda: _grid_init(N, seed=100),
        lambda: _hex_init(N, seed=100),
        lambda: _random_init(N, seed=3),
        lambda: _random_init(N, seed=4),
    ]
    
    for idx, init_fn in enumerate(inits):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        positions, radii = init_fn()
        update_best(positions, radii)
        
        remaining = DEADLINE - (time.time() - start_time)
        if remaining < 15:
            continue
            
        # Joint optimization
        try:
            new_pos, new_rad = _optimize_joint(positions, radii, time_budget=min(10, remaining - 5))
            update_best(new_pos, new_rad)
        except Exception:
            pass
            
    # Final polish on the best result
    if best_positions is not None:
        remaining = DEADLINE - (time.time() - start_time)
        if remaining > 10:
            try:
                new_pos, new_rad = _optimize_joint(best_positions, best_radii, time_budget=min(15, remaining - 5))
                update_best(new_pos, new_rad)
            except Exception:
                pass

    # Fallback
    if best_positions is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3])
        best_positions = np.array([c[:2] for c in cells])
        best_radii = np.array([c[2] for c in cells])
        
    # Final validity check
    best_radii = _fix_validity(best_positions, best_radii)
    
    result = np.zeros((N, 3))
    result[:, 0] = best_positions[:, 0]
    result[:, 1] = best_positions[:, 1]
    result[:, 2] = best_radii
    
    return result
