import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11
# Triangle vertices
V1 = np.array([0.0, 0.0])
V2 = np.array([1.0, 0.0])
V3 = np.array([0.5, np.sqrt(3) / 2])

# Precompute triples
triples = list(itertools.combinations(range(N), 3))
num_triples = len(triples)

def area(i, j, k, points):
    pi = points[i]
    pj = points[j]
    pk = points[k]
    # 2 * Area = |det|
    return 0.5 * abs(pi[0] * (pj[1] - pk[1]) + pj[0] * (pk[1] - pi[1]) + pk[0] * (pi[1] - pj[1]))

def compute_min_area(points):
    min_a = float('inf')
    for i, j, k in triples:
        a = area(i, j, k, points)
        if a < min_a:
            min_a = a
    return min_a

def is_inside_triangle(p):
    # Check barycentric coordinates or half-planes
    # Half planes:
    # y >= 0
    # sqrt(3)*x + y <= sqrt(3)
    # sqrt(3)*x - y >= 0
    if p[1] < -1e-9: return False
    if np.sqrt(3)*p[0] + p[1] > np.sqrt(3) + 1e-9: return False
    if np.sqrt(3)*p[0] - p[1] < -1e-9: return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s buffer
    
    best_score = -1.0
    best_points = None
    
    # Initial guess: random points
    rng = np.random.default_rng(42)
    
    # We will try a few different strategies for initial points
    # 1. Random uniform
    # 2. Grid-based
    
    def run_optimization(initial_points):
        # Flatten points and add t
        # x = [x1, y1, ..., xN, yN, t]
        x0 = np.zeros(2 * N + 1)
        for i in range(N):
            x0[2*i] = initial_points[i][0]
            x0[2*i+1] = initial_points[i][1]
        x0[-1] = 0.01 # Initial t guess
        
        def objective(x):
            # Maximize t => Minimize -t
            return -x[-1]
            
        def objective_grad(x):
            g = np.zeros_like(x)
            g[-1] = -1.0
            return g

        # Constraints
        # 1. Area constraints: area(i,j,k) - t >= 0
        # 2. Boundary constraints
        
        cons = []
        
        # Area constraints
        for idx, (i, j, k) in enumerate(triples):
            def constraint(x, i=i, j=j, k=k):
                pi = np.array([x[2*i], x[2*i+1]])
                pj = np.array([x[2*j], x[2*j+1]])
                pk = np.array([x[2*k], x[2*k+1]])
                det = pi[0]*(pj[1]-pk[1]) + pj[0]*(pk[1]-pi[1]) + pk[0]*(pi[1]-pj[1])
                a = 0.5 * abs(det)
                return a - x[-1]
                
            def constraint_grad(x, i=i, j=j, k=k):
                pi = np.array([x[2*i], x[2*i+1]])
                pj = np.array([x[2*j], x[2*j+1]])
                pk = np.array([x[2*k], x[2*k+1]])
                det = pi[0]*(pj[1]-pk[1]) + pj[0]*(pk[1]-pi[1]) + pk[0]*(pi[1]-pj[1])
                sign = 1.0 if det >= 0 else -1.0
                g = np.zeros_like(x)
                # d(det)/dx_i = y_j - y_k
                # d(det)/dy_i = -(x_j - x_k)
                g[2*i] = 0.5 * sign * (pj[1] - pk[1])
                g[2*i+1] = -0.5 * sign * (pj[0] - pk[0])
                
                g[2*j] = 0.5 * sign * (pk[1] - pi[1])
                g[2*j+1] = -0.5 * sign * (pk[0] - pi[0])
                
                g[2*k] = 0.5 * sign * (pi[1] - pj[1])
                g[2*k+1] = -0.5 * sign * (pi[0] - pj[0])
                
                g[-1] = -1.0
                return g

            cons.append({'type': 'ineq', 'fun': constraint, 'jac': constraint_grad})
            
        # Boundary constraints for each point
        # y >= 0
        # sqrt(3)x + y <= sqrt(3) => sqrt(3)x + y - sqrt(3) <= 0 => -sqrt(3)x - y + sqrt(3) >= 0
        # sqrt(3)x - y >= 0
        
        for i in range(N):
            # y_i >= 0
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i: x[2*i+1],
                'jac': lambda x, i=i: np.array([0 if j != 2*i+1 else 1.0 for j in range(2*N+1)])
            })
            # -sqrt(3)x_i - y_i + sqrt(3) >= 0
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i: -np.sqrt(3)*x[2*i] - x[2*i+1] + np.sqrt(3),
                'jac': lambda x, i=i: np.array([-np.sqrt(3) if j == 2*i else (-1.0 if j == 2*i+1 else 0.0) for j in range(2*N+1)])
            })
            # sqrt(3)x_i - y_i >= 0
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i: np.sqrt(3)*x[2*i] - x[2*i+1],
                'jac': lambda x, i=i: np.array([np.sqrt(3) if j == 2*i else (-1.0 if j == 2*i+1 else 0.0) for j in range(2*N+1)])
            })
            
        try:
            res = minimize(objective, x0, method='SLSQP', jac=objective_grad, bounds=[(0, None)] * (2*N+1), constraints=cons, options={'maxiter': 100, 'ftol': 1e-9})
            if res.success:
                pts = res.x[:-1].reshape(N, 2)
                # Verify validity
                valid = True
                for p in pts:
                    if not is_inside_triangle(p):
                        valid = False
                        break
                if valid:
                    score = compute_min_area(pts)
                    return score, pts
        except Exception as e:
            pass
        return -1.0, None

    # Multi-start
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        if r == 0:
            # Grid-like start
            points = []
            # Try to distribute points somewhat evenly
            # Simple random
            while len(points) < N:
                u = rng.random()
                v = rng.random()
                if u + v < 1.0:
                    p = V1 + u * (V2 - V1) + v * (V3 - V1)
                    points.append(p)
        else:
            # Random start
            points = []
            while len(points) < N:
                u = rng.random()
                v = rng.random()
                if u + v < 1.0:
                    p = V1 + u * (V2 - V1) + v * (V3 - V1)
                    # Add small jitter to avoid degeneracy
                    p = p + rng.normal(0, 0.01, 2)
                    # Project back if outside? Or just reject
                    if is_inside_triangle(p):
                        points.append(p)
                        
        score, pts = run_optimization(np.array(points))
        
        if score > best_score:
            best_score = score
            best_points = pts
            
    if best_points is None:
        # Fallback to seed-like random points
        pts = []
        while len(pts) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                pts.append([u + 0.5 * v, (3 ** 0.5) / 2 * v])
        return np.array(pts)
        
    return best_points

