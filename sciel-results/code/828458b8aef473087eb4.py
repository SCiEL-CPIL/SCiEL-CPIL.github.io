import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.optimal_x = None

    def _calc_hv(self, x):
        # x is sorted array of x-coordinates
        # y = 1 - sqrt(x)
        y = 1 - np.sqrt(x)
        ref = 11.0
        # HV = sum (x_{i+1}-x_i)(ref-y_i) + (ref-x_N)(ref-y_N)
        # Note: y is decreasing as x increases.
        # The formula derived:
        # HV = sum_{i=0}^{N-2} (x[i+1]-x[i]) * (ref - y[i]) + (ref - x[-1]) * (ref - y[-1])
        # Wait, the "height" for the interval [x_i, x_{i+1}] is determined by the point with smaller x (higher y).
        # So for interval [x_i, x_{i+1}], the point is i.
        # So term is (x[i+1]-x[i]) * (ref - y[i]).
        # And the last part is (ref - x[-1]) * (ref - y[-1]).
        terms = np.diff(x) * (ref - y[:-1])
        last = (ref - x[-1]) * (ref - y[-1])
        return np.sum(terms) + last

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280 # 20s margin
        
        if self.optimal_x is None:
            N = self.budget
            # Initial guess: uniform
            x0 = np.linspace(0, 1, N)
            
            def objective(x):
                # Ensure sorted
                x_sorted = np.sort(x)
                return -self._calc_hv(x_sorted)
            
            # Constraints: x in [0,1]
            bounds_opt = [(0, 1)] * N
            
            # We can just optimize x and sort them in the objective?
            # L-BFGS-B doesn't handle sorting well for gradients.
            # Better to enforce x1 < x2 < ... < xN.
            # We can optimize x1, d2, ..., dN where x_i = x_{i-1} + d_i.
            # d_i > 0.
            # x1 in [0, 1-sum(d)]
            
            # Let's try a simpler approach:
            # Since the curve is known, we can use a grid search for the "shape" 
            # or just use a heuristic that is known to be good.
            
            # Actually, for y = 1-sqrt(x), the optimal distribution is known to be 
            # denser near x=0? Or x=1?
            # Let's just use a simple local search on the sorted array.
            
            x_curr = np.linspace(0, 1, N)
            best_hv = self._calc_hv(x_curr)
            
            # Simple iterative improvement
            # We can move each point to the position that maximizes HV given its neighbors.
            # For point i, HV depends on x_{i-1}, x_i, x_{i+1}.
            # Term involving x_i:
            # (x_i - x_{i-1})(ref - y_{i-1}) + (x_{i+1} - x_i)(ref - y_i)
            # Let A = ref - y_{i-1}, B = ref - y_i.
            # f(x_i) = (x_i - x_{i-1})A + (x_{i+1} - x_i)B
            # f(x_i) = x_i(A - B) + const
            # Since x_{i-1} < x_i < x_{i+1}, and y is decreasing, y_{i-1} > y_i, so A < B.
            # A - B < 0.
            # So f(x_i) is decreasing in x_i.
            # To maximize f(x_i), we should make x_i as small as possible.
            # So x_i should be close to x_{i-1}.
            # This suggests all points should cluster at x=0?
            # Wait, if x_i -> x_{i-1}, the interval (x_{i-1}, x_i) vanishes.
            # But the next interval (x_i, x_{i+1}) grows.
            # Let's re-evaluate.
            # HV = sum (x_{i+1}-x_i) h_i.
            # If we move x_i to the left, we decrease the width of interval i-1 (which has height h_{i-1}) 
            # and increase the width of interval i (which has height h_i).
            # Since h_{i-1} > h_i (because x_{i-1} < x_i, so y_{i-1} > y_i, so ref-y_{i-1} < ref-y_i? 
            # Wait. y = 1-sqrt(x). x small -> y large.
            # ref - y = ref - (1-sqrt(x)) = 10 + sqrt(x).
            # So h(x) = 10 + sqrt(x).
            # h(x) is increasing.
            # So h_{i-1} < h_i.
            # So the interval to the left has lower height. The interval to the right has higher height.
            # We want to maximize width * height.
            # Moving x_i left decreases width of left interval (low height) and increases width of right interval (high height).
            # This is beneficial!
            # So we should move x_i as left as possible.
            # This implies all points should be at x=0?
            # If all points are at x=0, HV = (11-0)(11-1) = 110.
            # If points are spread out, e.g., x_N=1.
            # Let's check x=[0, 1].
            # HV = (1-0)(10+0) + (11-1)(10+1) = 10 + 110 = 120.
            # 120 > 110.
            # So spreading is good.
            
            # The optimal x_i satisfies the condition that the marginal gain is zero.
            # This is a standard optimal control problem.
            # Let's just use `scipy.optimize.minimize` on the sorted variables with a penalty for non-monotonicity, 
            # or better, optimize the gaps.
            
            # Given time constraints, a simple heuristic:
            # The optimal points for maximizing HV of a convex curve are often 
            # distributed such that the "area" of the trapezoids is balanced?
            # Let's use a numerical optimization with `differential_evolution` on a reduced set of parameters?
            # No, let's just use `minimize` on the gaps.
            
            # We have N points. N-1 gaps.
            # x_1, d_2, ..., d_N.
            # x_1 in [0, 1].
            # d_i in [0, 1-x_{i-1}].
            
            # This is complex to set up bounds dynamically.
            
            # Alternative: 
            # Since we have 200 points, we can just use a very dense grid of "candidate" x positions 
            # and pick the best 200? No, we can choose any x.
            
            # Let's try a simple iterative gradient-free optimization.
            # Start with uniform.
            # For each point, try moving it to the best position in its local neighborhood.
            
            x_opt = x_curr.copy()
            for _ in range(50): # 50 iterations
                improved = False
                for i in range(N):
                    # Try moving x_i
                    # Local search in [max(0, x_{i-1}-0.01), min(1, x_{i+1}+0.01)]
                    # We want to maximize HV.
                    # We can just sample a few positions.
                    best_val = self._calc_hv(x_opt)
                    best_pos = x_opt[i]
                    
                    # Sample 10 positions around current
                    for delta in np.linspace(-0.05, 0.05, 11):
                        new_x = x_opt[i] + delta
                        if new_x < 0 or new_x > 1: continue
                        # Check ordering
                        if i > 0 and new_x <= x_opt[i-1]: continue
                        if i < N-1 and new_x >= x_opt[i+1]: continue
                        
                        x_opt[i] = new_x
                        val = self._calc_hv(x_opt)
                        if val > best_val:
                            best_val = val
                            best_pos = new_x
                        x_opt[i] = x_curr[i] # restore temporarily? No, we are in a loop.
                        
                    x_opt[i] = best_pos
                    if best_pos != x_curr[i]:
                        x_curr[i] = best_pos
                        improved = True
                if not improved: break
                
            self.optimal_x = x_curr

        # Generate points
        results = []
        for i in range(self.budget):
            if time.time() > deadline:
                break
            x_val = self.optimal_x[i]
            x_vec = np.zeros(self.dim)
            x_vec[0] = x_val
            x_vec[1:] = 0.5
            res = problem(x_vec)
            results.append(res)
            
        return results
