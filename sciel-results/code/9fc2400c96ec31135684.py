import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * np.sqrt(3) / 2  # area of equilateral triangle with side 1

# Precompute all triangle index triples
TRIPLES = list(itertools.combinations(range(N), 3))
n_triples = len(TRIPLES)  # 165


def triangle_areas(pts):
    """Compute areas of all C(N,3) triangles. pts shape (N,2)."""
    areas = np.empty(n_triples)
    for idx, (i, j, k) in enumerate(TRIPLES):
        ax, ay = pts[i]
        bx, by = pts[j]
        cx, cy = pts[k]
        areas[idx] = 0.5 * abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
    return areas


def soft_min_neg(areas, beta):
    """Negative soft-min: -log(sum(exp(beta*areas)))/beta approximates -min(areas).
    We maximize this, so we minimize its negative."""
    # For stability, subtract max
    m = np.max(areas)
    s = np.sum(np.exp(beta * (areas - m)))
    return -(m + np.log(s) / beta)


def objective_and_grad(x, beta):
    """Objective: minimize -soft_min(areas) = maximize soft_min(areas).
    Returns (value, gradient)."""
    pts = x.reshape(N, 2)
    areas = triangle_areas(pts)
    val = -soft_min_neg(areas, beta)
    
    # Gradient via finite differences or analytic
    # analytic: d/dx of -soft_min = -(1/beta) * d/dx log(sum(exp(beta*areas)))
    # = -(1/beta) * sum(exp(beta*areas_i)/sum(exp(beta*areas))) * beta * d(areas_i)/dx
    # = -sum(w_i * d(areas_i)/dx) where w_i = exp(beta*areas_i)/sum(exp(beta*areas))
    m = np.max(areas)
    exps = np.exp(beta * (areas - m))
    w = exps / np.sum(exps)
    
    grad = np.zeros(N * 2)
    for idx, (i, j, k) in enumerate(TRIPLES):
        wi = w[idx]
        if wi < 1e-12:
            continue
        ax, ay = pts[i]
        bx, by = pts[j]
        cx, cy = pts[k]
        # area = 0.5 * |det|; for gradient we need sign-aware
        # Let's use the signed area for gradient computation
        det = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
        sgn = 1.0 if det >= 0 else -1.0
        # d(area)/d(ax) = 0.5 * sgn * (-(cy-ay) - (by-ay)) = 0.5*sgn*(ay-cy+ay-by) ... 
        # Let me recompute: area = 0.5*|det|, det = (bx-ax)(cy-ay) - (cx-ax)(by-ay)
        # d(det)/d(ax) = -(cy-ay) + (by-ay) = by - cy
        # d(det)/d(ay) = -(bx-ax) + (cx-ax) = cx - bx
        # d(det)/d(bx) = (cy-ay)
        # d(det)/d(by) = -(cx-ax)
        # d(det)/d(cx) = -(by-ay)
        # d(det)/d(cy) = (bx-ax)
        # d(area)/d(var) = 0.5 * sgn * d(det)/d(var)
        h = 0.5 * sgn * wi
        grad[2*i + 0] += h * (by - cy)
        grad[2*i + 1] += h * (cx - bx)
        grad[2*j + 0] += h * (cy - ay)
        grad[2*j + 1] += h * (-(cx - ax))
        grad[2*k + 0] += h * (-(by - ay))
        grad[2*k + 1] += h * (bx - ax)
    
    return val, grad


def constraints_func(x):
    """Return constraint values (all should be >= 0 for feasibility)."""
    pts = x.reshape(N, 2)
    cons = []
    for p in pts:
        # Inside triangle: barycentric coords >= 0
        # Using edge-based: for edge (A,B), point C is on same side as third vertex
        # Edge V0->V1: y >= 0
        cons.append(p[1])
        # Edge V1->V2: line from (1,0) to (0.5, sqrt(3)/2)
        # Direction: (-0.5, sqrt(3)/2), normal pointing inward (toward V0)
        # Line: sqrt(3)/2*(x-1) + 0.5*y = 0 => sqrt(3)/2*x - sqrt(3)/2 + 0.5*y = 0
        # At V0=(0,0): -sqrt(3)/2 < 0, so inward is negative side
        # Constraint: sqrt(3)/2*(x-1) + 0.5*y <= 0 => -(sqrt(3)/2*(x-1) + 0.5*y) >= 0
        cons.append(-(np.sqrt(3)/2 * (p[0] - 1) + 0.5 * p[1]))
        # Edge V2->V0: line from (0.5, sqrt(3)/2) to (0,0)
        # Direction: (-0.5, -sqrt(3)/2), normal inward (toward V1)
        # Line: sqrt(3)/2*x - 0.5*y = 0
        # At V1=(1,0): sqrt(3)/2 > 0, so inward is positive side
        # Constraint: sqrt(3)/2*x - 0.5*y >= 0
        cons.append(np.sqrt(3)/2 * p[0] - 0.5 * p[1])
    return cons


def constraint_jacobian(x):
    """Jacobian of constraints. Shape (3N, 2N)."""
    J = np.zeros((3 * N, 2 * N))
    for i in range(N):
        # Constraint 3i: y_i >= 0
        J[3*i, 2*i+1] = 1.0
        # Constraint 3i+1: -(sqrt(3)/2*(x_i-1) + 0.5*y_i) >= 0
        J[3*i+1, 2*i+0] = -np.sqrt(3)/2
        J[3*i+1, 2*i+1] = -0.5
        # Constraint 3i+2: sqrt(3)/2*x_i - 0.5*y_i >= 0
        J[3*i+2, 2*i+0] = np.sqrt(3)/2
        J[3*i+2, 2*i+1] = -0.5
    return J


def constraint_bounds():
    return [(0, np.inf)] * (3 * N)


def optimize_from_start(x0, beta, deadline):
    """Run SLSQP optimization from x0 with given beta."""
    constraints = [{'type': 'ineq', 'fun': constraints_func, 'jac': constraint_jacobian}]
    
    def obj(x):
        val, grad = objective_and_grad(x, beta)
        return val, grad
    
    res = minimize(
        obj, x0, method='SLSQP',
        constraints=constraints,
        bounds=[(0, 1)] * (2 * N),
        options={'maxiter': 500, 'ftol': 1e-10, 'disp': False}
    )
    return res


def score(points):
    """Compute the raw score: min triangle area / TRI_AREA."""
    areas = triangle_areas(points)
    return np.min(areas) / TRI_AREA


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_points = None
    best_score = -np.inf
    
    np.random.seed(42)
    
    # Multiple restarts with different initial configurations
    n_restarts = 50
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate initial points
        if restart == 0:
            # Symmetric-ish initial guess
            pts = np.zeros((N, 2))
            # Place points roughly evenly
            angles = np.linspace(0, 2*np.pi, N, endpoint=False)
            radii = np.linspace(0.1, 0.4, N)
            cx, cy = 0.5, np.sqrt(3)/6
            for i in range(N):
                pts[i] = [cx + radii[i]*np.cos(angles[i]), cy + radii[i]*np.sin(angles[i])]
            # Clip to triangle
            for i in range(N):
                pts[i][0] = np.clip(pts[i][0], 0.01, 0.99)
                pts[i][1] = np.clip(pts[i][1], 0.01, np.sqrt(3)/2 - 0.01)
                # Ensure inside triangle
                while not (pts[i][1] >= -1e-9 and 
                          -(np.sqrt(3)/2*(pts[i][0]-1) + 0.5*pts[i][1]) >= -1e-9 and
                          np.sqrt(3)/2*pts[i][0] - 0.5*pts[i][1] >= -1e-9):
                    pts[i] = [0.3 + 0.4*np.random.random(), 0.1 + 0.3*np.random.random()]
        else:
            # Random initialization
            pts = np.random.random((N, 2)) * np.array([1.0, np.sqrt(3)/2])
            # Filter to inside triangle
            valid = []
            for _ in range(N):
                for attempt in range(100):
                    p = np.random.random(2) * np.array([1.0, np.sqrt(3)/2])
                    if (p[1] >= -1e-9 and 
                        -(np.sqrt(3)/2*(p[0]-1) + 0.5*p[1]) >= -1e-9 and
                        np.sqrt(3)/2*p[0] - 0.5*p[1] >= -1e-9):
                        valid.append(p)
                        break
                if len(valid) >= N:
                    break
            if len(valid) < N:
                # Fallback: uniform random inside triangle
                valid = []
                for _ in range(N):
                    u, v = np.random.random(2)
                    while u + v > 1:
                        u, v = np.random.random(2)
                    p = [u + 0.5*v, np.sqrt(3)/2*v]
                    valid.append(p)
            pts = np.array(valid[:N])
        
        x0 = pts.flatten()
        
        # Try different beta values (annealing)
        for beta in [5.0, 10.0, 20.0, 50.0]:
            if time.time() > deadline:
                break
            
            try:
                res = optimize_from_start(x0, beta, deadline)
                if res.success or res.fun < np.inf:
                    x_opt = res.x
                    pts_opt = x_opt.reshape(N, 2)
                    s = score(pts_opt)
                    if s > best_score:
                        best_score = s
                        best_points = pts_opt.copy()
                    x0 = x_opt  # Warm start for next beta
            except Exception:
                pass
        
        # Also try a direct min-max approach with higher beta
        if time.time() > deadline:
            break
        
        for beta in [100.0, 200.0]:
            if time.time() > deadline:
                break
            try:
                res = optimize_from_start(x0, beta, deadline)
                if res.success or res.fun < np.inf:
                    x_opt = res.x
                    pts_opt = x_opt.reshape(N, 2)
                    s = score(pts_opt)
                    if s > best_score:
                        best_score = s
                        best_points = pts_opt.copy()
            except Exception:
                pass
    
    if best_points is None:
        # Fallback: return random valid points
        pts = []
        for _ in range(N):
            u, v = np.random.random(2)
            while u + v > 1:
                u, v = np.random.random(2)
            pts.append([u + 0.5*v, np.sqrt(3)/2*v])
        return np.array(pts)
    
    return best_points
