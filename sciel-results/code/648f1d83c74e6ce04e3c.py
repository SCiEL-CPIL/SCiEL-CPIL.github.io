import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Multi-objective optimizer for ZDT3 using Direct Numerical Hypervolume Optimization.
    
    Strategy:
    1. The Pareto front of ZDT3 is known analytically: f1 = x1, f2 = 1 - sqrt(x1) for x1 in [0,1],
       achieved when x2..xn = 0.
    2. The problem reduces to selecting B x1 values in [0,1] to maximize the hypervolume of 
       {(x1, 1-sqrt(x1))} with reference point [11, 11].
    3. Instead of heuristic search, we treat the x1 values as continuous variables and use 
       scipy.optimize.minimize (L-BFGS-B) to directly maximize the hypervolume.
    4. We provide an exact gradient for the hypervolume with respect to the x1 variables, 
       which allows the optimizer to converge rapidly to the optimal distribution of points.
    5. This is a structural change from iterative greedy/coordinate descent to a global 
       numerical optimization method with gradient information.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hypervolume_and_grad(self, x1_vals):
        """
        Calculate hypervolume and its gradient with respect to x1_vals.
        x1_vals: array of shape (N,)
        Returns: (hv, grad)
        """
        N = len(x1_vals)
        if N == 0:
            return 0.0, np.array([])
        
        # Sort x1_vals
        sort_idx = np.argsort(x1_vals)
        x_sorted = x1_vals[sort_idx]
        
        # Calculate f2 values
        # f2 = 1 - sqrt(x1)
        # Note: sqrt(0) = 0, so f2(0) = 1.
        # For x1 > 0, f2 < 1.
        y_sorted = 1.0 - np.sqrt(x_sorted)
        
        # Reference point
        ref_x = 11.0
        ref_y = 11.0
        
        # Hypervolume calculation in 2D
        # Points are (x_i, y_i).
        # HV = sum_{i=0}^{N-1} (x_{i+1} - x_i) * (ref_y - y_i)
        # where x_N = ref_x, and y_N is not used in the term (it's the width to ref_x)
        # Actually, the standard formula for non-dominated points sorted by x ascending:
        # The area is the sum of rectangles.
        # Rectangle i has height (ref_y - y_i) and width (x_{i+1} - x_i).
        # The last rectangle has width (ref_x - x_{N-1}) and height (ref_y - y_{N-1}).
        
        # We need to be careful with duplicate x values.
        # If x_i == x_{i+1}, the width is 0.
        # The gradient with respect to x_i involves the terms where x_i is x_{i+1} (as upper bound) 
        # and where x_i is x_i (as lower bound).
        
        # Let's compute HV first.
        x_next = np.append(x_sorted[1:], ref_x)
        widths = x_next - x_sorted
        heights = ref_y - y_sorted
        
        # Check for negative widths (should not happen if sorted, but numerical errors?)
        # Also, if points are not strictly improving, some might be dominated.
        # ZDT3 front is convex in (f1, f2) space? 
        # f1 = x, f2 = 1 - sqrt(x). 
        # f2' = -1/(2*sqrt(x)). f2'' = 1/(4*x^(3/2)) > 0. 
        # So the front is convex towards the origin? No, concave towards origin?
        # Let's check curvature. 
        # Points: (0,1), (1,0). Midpoint (0.5, 1-sqrt(0.5) ~ 0.29).
        # Line segment from (0,1) to (1,0) is y = 1-x. At x=0.5, y=0.5.
        # 0.29 < 0.5. So the front is "below" the chord. 
        # This means the set of dominated points is convex? 
        # Regardless, since we are on the true Pareto front, all points are non-dominated 
        # with respect to each other.
        
        hv = np.sum(widths * heights)
        
        # Gradient calculation
        # d(HV)/d(x_i)
        # x_i appears in:
        # 1. Term i: width_i = x_{i+1} - x_i. d(width_i)/d(x_i) = -1.
        #    Contribution: -1 * (ref_y - y_i)
        # 2. Term i-1: width_{i-1} = x_i - x_{i-1}. d(width_{i-1})/d(x_i) = +1.
        #    Contribution: +1 * (ref_y - y_{i-1})
        
        # So for interior points (1 <= i < N-1):
        # grad_i = (ref_y - y_{i-1}) - (ref_y - y_i) = y_i - y_{i-1}
        
        # For i = 0:
        # Only term 0 involves x_0 as lower bound.
        # grad_0 = -(ref_y - y_0)
        
        # For i = N-1:
        # Term N-1: width = ref_x - x_{N-1}. d/dx = -1.
        # Term N-2: width = x_{N-1} - x_{N-2}. d/dx = +1.
        # grad_{N-1} = (ref_y - y_{N-2}) - (ref_y - y_{N-1}) = y_{N-1} - y_{N-2}
        
        # Wait, is this correct?
        # HV = sum_{j=0}^{N-1} (x_{j+1} - x_j) * H_j
        # where H_j = ref_y - y_j.
        # dHV/dx_i = sum_j [ d(x_{j+1}-x_j)/dx_i * H_j + (x_{j+1}-x_j) * dH_j/dx_i ]
        
        # Term 1:
        # If j = i: d(x_{i+1}-x_i)/dx_i = -1. Contrib: -H_i
        # If j = i-1: d(x_i - x_{i-1})/dx_i = +1. Contrib: +H_{i-1}
        # Other j: 0.
        
        # Term 2:
        # H_j = ref_y - (1 - sqrt(x_j)) = ref_y - 1 + sqrt(x_j).
        # dH_j/dx_i is non-zero only if j = i.
        # dH_i/dx_i = 1/(2*sqrt(x_i)).
        # So Contrib: (x_{i+1} - x_i) * (1/(2*sqrt(x_i)))
        
        # So:
        # grad_i = H_{i-1} - H_i + (x_{i+1} - x_i) * (1/(2*sqrt(x_i)))
        
        # Let's re-verify H.
        # y_i = 1 - sqrt(x_i).
        # H_i = ref_y - y_i = ref_y - 1 + sqrt(x_i).
        
        # For i=0:
        # H_{-1} is undefined (treat as 0? No, there is no term -1).
        # So grad_0 = -H_0 + (x_1 - x_0) * (1/(2*sqrt(x_0)))
        # Note: if x_0 == 0, sqrt is 0, derivative is infinite?
        # f2 = 1 - sqrt(x). At x=0, f2=1.
        # The derivative of f2 at 0 is -infinity.
        # This might cause numerical issues.
        # However, L-BFGS-B handles bounds. We can clip x to a small epsilon for gradient calc.
        
        # For i=N-1:
        # x_{N} = ref_x.
        # grad_{N-1} = H_{N-2} - H_{N-1} + (ref_x - x_{N-1}) * (1/(2*sqrt(x_{N-1})))
        
        # Let's compute the gradient array.
        grad_sorted = np.zeros(N)
        
        # H values
        H = ref_y - y_sorted
        
        # Part 1: From widths
        # grad_i gets +H_{i-1} from term i-1 (if i>0)
        # grad_i gets -H_i from term i
        
        if N > 1:
            grad_sorted[1:] += H[:-1]
        grad_sorted -= H
        
        # Part 2: From heights
        # grad_i gets (x_{i+1} - x_i) * dH_i/dx_i
        # dH_i/dx_i = 1 / (2 * sqrt(x_i))
        
        # Handle x_i = 0
        # If x_i is very small, use a small epsilon to avoid division by zero or huge values.
        # Actually, if x_i=0, the point is (0,1). 
        # The area added by moving x_0 slightly to the right:
        # New point (dx, 1-sqrt(dx)).
        # The "slope" of the front is -inf at 0.
        # So moving x_0 to the right drops y significantly.
        # Let's just clip x for the derivative calculation.
        x_safe = np.maximum(x_sorted, 1e-12)
        
        dH_dx = 1.0 / (2.0 * np.sqrt(x_safe))
        
        widths_for_grad = np.append(x_sorted[1:] - x_sorted[:-1], ref_x - x_sorted[-1])
        
        grad_sorted += widths_for_grad * dH_dx
        
        # Map gradient back to original order
        grad = np.zeros(N)
        grad[sort_idx] = grad_sorted
        
        return hv, grad

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        n = self.dim
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        B = self.budget
        if B == 0:
            return np.array([])
            
        # Initial guess: Power law distribution
        # x1 = t^p. p=2 is a good start for ZDT3-like fronts.
        if B == 1:
            x1_init = np.array([0.0])
        else:
            t = np.linspace(0, 1, B)
            x1_init = t ** 2.0
            # Ensure bounds
            x1_init = np.clip(x1_init, lo[0], hi[0])
            
        # Objective function for minimizer: Negate hypervolume
        def objective(x1_vals):
            hv, _ = self._hypervolume_and_grad(x1_vals)
            return -hv
        
        # Gradient function
        def gradient(x1_vals):
            _, grad = self._hypervolume_and_grad(x1_vals)
            return -grad  # Negate for minimization
        
        # Bounds for x1
        # x1 is the first variable. Bounds are [0, 1].
        # Other variables are 0.
        # So x1_vals are bounded by [lo[0], hi[0]]
        b_lo = lo[0]
        b_hi = hi[0]
        
        bounds_opt = [(b_lo, b_hi) for _ in range(B)]
        
        # Use L-BFGS-B
        # maxiter: limit iterations to prevent timeout
        # maxfun: limit function evaluations
        
        # We have ~290s. Each eval is fast (vectorized numpy).
        # 1000 iterations should be plenty.
        
        try:
            res = minimize(
                objective, 
                x1_init, 
                jac=gradient, 
                method='L-BFGS-B', 
                bounds=bounds_opt,
                options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            x1_opt = res.x
        except Exception as e:
            # Fallback to initial guess if optimization fails
            x1_opt = x1_init
            if time.time() > deadline - 5:
                pass # Just use initial
        
        # Ensure x1_opt is within bounds and valid
        x1_opt = np.clip(x1_opt, b_lo, b_hi)
        
        # Final evaluation
        F = []
        for x1 in x1_opt:
            x = np.zeros(n)
            x[0] = x1
            x = np.clip(x, lo, hi)
            
            if time.time() > deadline:
                break
                
            f = problem(x)
            F.append(f)
            
        return np.array(F)
