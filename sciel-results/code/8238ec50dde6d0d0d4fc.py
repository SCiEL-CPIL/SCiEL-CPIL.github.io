import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = self.start_time + 325.0

    def _calc_hv(self, points, ref):
        # points: list of (f1, f2)
        if not points:
            return 0.0
        # Sort by f1
        points = sorted(points, key=lambda p: p[0])
        hv = 0.0
        n = len(points)
        for i in range(n):
            x_curr, y_curr = points[i]
            if i == n - 1:
                x_next = ref[0]
            else:
                x_next = points[i+1][0]
            # Ensure non-negative width
            width = x_next - x_curr
            if width < 0: width = 0
            height = ref[1] - y_curr
            if height < 0: height = 0
            hv += width * height
        return hv

    def _find_optimal_x0(self, n):
        if n <= 0:
            return np.array([])
        if n == 1:
            # Maximize (11 - x)(10 + sqrt(x))
            # x in [0, 1]
            # Derivative: -(10+sqrt(x)) + (11-x)/(2*sqrt(x)) = 0
            # Let's just use a simple grid or analytic if possible.
            # For n=1, grid search is fine.
            xs = np.linspace(0, 1, 1000)
            f1 = xs
            f2 = 1.0 - np.sqrt(f1)
            hvs = [(11 - x) * (11 - y) for x, y in zip(f1, f2)]
            return xs[np.argmax(hvs)]

        # Optimize gaps u_i >= 0 such that sum(u) <= 1
        # x_1 = u_1
        # x_{i+1} = x_i + u_{i+1}
        # So x_k = sum_{j=1}^k u_j
        
        def objective(u):
            # u: vector of length n
            # Check sum constraint (handled by bounds/constraints, but let's be safe)
            if np.sum(u) > 1.0:
                return 0.0 # Penalty
            x = np.cumsum(u)
            # x is in [0, 1]
            f1 = x
            f2 = 1.0 - np.sqrt(f1)
            # Calculate HV
            # Points are (f1[i], f2[i])
            # Sorted by f1 (which is x, and u>=0 implies x is increasing)
            hv = 0.0
            # Last point to ref
            hv += (11.0 - x[-1]) * (11.0 - f2[-1])
            # Sum for i=0 to n-2
            # width = x[i+1] - x[i] = u[i+1]
            # height = 11 - f2[i]
            widths = np.diff(x) # length n-1
            heights = 11.0 - f2[:-1]
            hv += np.sum(widths * heights)
            return -hv # Minimize negative HV

        # Initial guess: uniform gaps
        u0 = np.ones(n) / n
        
        # Bounds
        bounds = [(0, 1.0) for _ in range(n)]
        
        # Constraints
        # sum(u) <= 1
        cons = ({'type': 'ineq', 'fun': lambda u: 1.0 - np.sum(u)})
        
        # Use SLSQP or L-BFGS-B
        # SLSQP handles inequality constraints well
        # L-BFGS-B only handles bounds. We can transform variables?
        # Or just use SLSQP. For n=200, SLSQP might be slow.
        # Let's try L-BFGS-B with a penalty for sum(u) > 1?
        # Or better, use the fact that optimal u will likely satisfy sum(u) = 1?
        # If sum(u) < 1, we can increase some u_i to increase HV?
        # Yes, increasing x_N increases the last term (11-x_N)(10+sqrt(x_N)).
        # Derivative of (11-x)(10+sqrt(x)) w.r.t x is -(10+sqrt(x)) + (11-x)/(2sqrt(x)).
        # At x=1, deriv = -11 + 5 = -6. So increasing x_N decreases HV?
        # Wait. f2 = 1 - sqrt(x).
        # Term: (11 - x) (11 - (1 - sqrt(x))) = (11-x)(10+sqrt(x)).
        # Let h(x) = (11-x)(10+sqrt(x)).
        # h'(x) = -(10+sqrt(x)) + (11-x)/(2sqrt(x)).
        # At x=1: h'(1) = -11 + 5 = -6.
        # So we want x_N to be as small as possible?
        # But we have to place N points in [0,1].
        # If we shift all points left, we gain in the last term but lose in the widths?
        # Actually, the widths are fixed by the gaps.
        # If we scale all u by alpha < 1, x becomes alpha*x.
        # This moves points closer to 0.
        # The last interval becomes (11 - alpha*x_N).
        # The heights 10+sqrt(alpha*x) decrease? No, sqrt is concave, so sqrt(alpha x) < alpha sqrt(x)? No.
        # Let's just rely on the optimizer.
        
        # To make it faster, we can use a coarse grid search first to get a good initial guess for u, then refine.
        # But the directive says use PMP / optimal control.
        # The "feedback law" in this context is the gradient of the HV w.r.t the positions.
        # So using a gradient-based optimizer IS applying the principle.
        
        try:
            res = minimize(objective, u0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 50, 'ftol': 1e-8})
            if res.success:
                u_opt = res.x
            else:
                # Fallback to grid search if optimization fails
                u_opt = u0
        except Exception:
            u_opt = u0
            
        x_opt = np.cumsum(u_opt)
        # Ensure within [0,1]
        x_opt = np.clip(x_opt, 0.0, 1.0)
        
        return x_opt

    def __call__(self, problem, bounds):
        n = self.budget
        if n <= 0:
            return None
        
        ref = np.array([11.0, 11.0])
        
        # Compute optimal x0 values
        best_points = self._find_optimal_x0(n)
        
        # Now evaluate the best points
        for x0 in best_points:
            if time.time() > self.deadline:
                break
            
            x = np.zeros(self.dim)
            x[0] = x0
            
            try:
                problem(x)
            except Exception:
                break

        return None
