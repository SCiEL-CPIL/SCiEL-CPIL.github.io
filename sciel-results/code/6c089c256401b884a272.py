import numpy as np
import time

class Optimizer:
    """
    Optimizer for BBOB f5 (Linear slope).
    
    Strategy:
    1. Exploit the linearity of the function f(x) = c + sum(c_i * (x_i - shift_i)).
       This is equivalent to f(x) = b + sum(w_i * x_i) where b = c - sum(w_i * shift_i).
    2. We need to find the vector w and scalar b to determine the optimal vertex.
       The optimal vertex x* has x*_i = -5 if w_i > 0, and x*_i = 5 if w_i < 0.
       (If w_i == 0, x*_i can be anything, but 0 is fine).
    3. To find w, we can evaluate the function at specific points that allow us to
       solve for w_i directly or via a linear system.
       
       Since we have a budget of 500 and dim=5, we can afford a robust approach.
       We will use 1 + 2*dim = 11 calls to determine the gradient exactly.
       
       Method:
       Evaluate f(0) = b.
       For each dimension i, evaluate f(e_i * 5) where e_i is the unit vector scaled by 5.
       Let x_i_plus = 5 * e_i.
       f(x_i_plus) = b + w_i * 5.
       So w_i = (f(x_i_plus) - b) / 5.
       
       This gives us the exact gradient components w_i in 11 calls.
       
       Then, we construct the optimal vertex x* based on the signs of w_i.
       x*_i = -5.0 if w_i > 0 else 5.0 (if w_i < 0).
       If w_i is very close to 0, we can set x*_i = 0.0, but for BBOB f5, 
       coefficients are typically non-zero. To be safe, if abs(w_i) < 1e-12, set to 0.
       
       Call f(x*) to confirm and update best.
       
       This takes 12 calls total.
       For the remaining budget, we do nothing (or we could verify with a few random points, 
       but since the model is exact for linear functions, additional calls only add 
       computational overhead without improving the best value found, unless there is 
       noise, which is not mentioned. The task says "BBOB-style", implying deterministic).
       
       However, the AOCC metric rewards early progress. 
       b_i is the best value after call i.
       Calls 1-11: We are gathering info. The best value found so far might be f(0) or 
       one of the f(5e_i). These are likely not the global minimum.
       Call 12: We evaluate the global minimum x*. Now b_12 = f_opt.
       Calls 13-500: b_i = f_opt.
       
       e_i = clip(b_i - f_opt, 1e-8, 100).
       For i >= 12, e_i = 1e-8 (since b_i = f_opt).
       n_i = (log10(1e-8) + 8) / 10 = (-8 + 8) / 10 = 0.
       AOCC contribution for i >= 12 is 1 - 0 = 1.
       
       For i < 12, b_i is some value > f_opt.
       e_i = b_i - f_opt.
       n_i = (log10(b_i - f_opt) + 8) / 10.
       AOCC = 1 - mean(n_1, ..., n_500).
       
       To maximize AOCC, we want n_i to be small (close to 0) for as many i as possible.
       This means we want b_i to be close to f_opt as soon as possible.
       
       Is there a way to find the optimum faster than 12 calls?
       No, we need to know the sign of all 5 coefficients.
       Each coefficient sign requires at least 1 evaluation relative to a baseline.
       So 1 + 5 = 6 calls might be enough if we are clever?
       
       Let's refine the "Fast Gradient" approach:
       1. Evaluate f(0) = b. (Call 1)
       2. For i in 1..5:
          Evaluate f(5*e_i). (Calls 2-6)
          w_i = (f(5*e_i) - b) / 5.
          Determine sign of w_i.
          Construct partial vertex? No, we need all signs to construct the full vertex.
          
       So we must wait until Call 6 to know all signs.
       Call 7: Evaluate the full vertex x*.
       
       This reduces the "blind" period from 12 calls to 7 calls.
       
       Let's trace:
       Call 1: f(0). best = f(0).
       Call 2: f(5e1). best = min(f(0), f(5e1)).
       ...
       Call 6: f(5e5). best = min(f(0), f(5e1), ..., f(5e5)).
       Call 7: f(x*). best = f(x*) = f_opt.
       
       This is significantly better for AOCC.
       
       What if we can infer signs earlier?
       No, we need the value to know the sign.
       
       Can we use a different baseline?
       If we assume the shift is small, maybe we can guess? No, black box.
       
       So the plan is:
       1. f(0)
       2. f(5e1) -> get w1
       3. f(5e2) -> get w2
       4. f(5e3) -> get w3
       5. f(5e4) -> get w4
       6. f(5e5) -> get w5
       7. Construct x* and evaluate f(x*).
       
       Total 7 calls to reach global optimum.
       
       For calls 8-500, we have found the optimum. 
       We should just return the best x found. 
       Do we need to make more calls? 
       "Every call is recorded". 
       If we stop calling, the evaluator repeats the last n_i.
       "repeating the last n_i if fewer calls are made"
       So if we make 7 calls, n_8...n_500 are copies of n_7.
       n_7 = 0.
       So AOCC = 1 - mean(n_1..n_7, 0, ..., 0).
       
       This is optimal.
       
       Implementation details:
       - Track calls.
       - Track best_f and best_x.
       - Handle exceptions.
       - Deadline guard.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        self.calls_made = 0
        self.best_f = float('inf')
        self.best_x = np.zeros(self.dim)

        def wrapped_func(x):
            if self.calls_made >= self.budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            
            self.calls_made += 1
            try:
                x_arr = np.asarray(x, dtype=float)
                val = float(func(x_arr))
                if val < self.best_f:
                    self.best_f = val
                    self.best_x = x_arr.copy()
                return val
            except Exception:
                return float('inf')

        # Phase 1: Exact Gradient Reconstruction
        # f(x) = b + w^T x
        
        # 1. Evaluate at origin
        x0 = np.zeros(self.dim)
        b = wrapped_func(x0)
        
        if self.calls_made >= self.budget or time.time() > deadline:
            return self.best_x, self.best_f
            
        # 2. Evaluate at 5 * e_i for each dimension to get w_i
        w = np.zeros(self.dim)
        for i in range(self.dim):
            if self.calls_made >= self.budget or time.time() > deadline:
                break
            xi = np.zeros(self.dim)
            xi[i] = 5.0
            val_i = wrapped_func(xi)
            w[i] = (val_i - b) / 5.0
            
        # 3. Construct optimal vertex
        # If w_i > 0, min is at -5. If w_i < 0, min is at 5.
        # If w_i is very close to 0, the function is flat in that direction.
        # We can set to 0 or either bound. 0 is safer if there's any non-linearity?
        # But for linear, it doesn't matter. Let's use 0 if abs(w_i) < 1e-12.
        
        x_star = np.zeros(self.dim)
        for i in range(self.dim):
            if w[i] > 1e-12:
                x_star[i] = -5.0
            elif w[i] < -1e-12:
                x_star[i] = 5.0
            else:
                x_star[i] = 0.0
        
        # 4. Evaluate the optimal vertex
        if self.calls_made < self.budget and time.time() < deadline:
            wrapped_func(x_star)
            
        # We have used at most 7 calls.
        # The best value found is now f_opt (assuming exact linearity).
        # Further calls will not improve the best value (for a linear function).
        # We can return now to save time, or we could add a safety check.
        # Given the "BBOB-style" and "Linear slope" description, the function is linear.
        # The "shift" and "rotation" are just affine transformations, which preserve linearity.
        # f(x) = c + v^T (R(x-s) + offset)? No, BBOB f5 is f(x) = c + sum(c_i (x_i - shift_i)).
        # This is linear in x.
        
        # Return the best result found.
        return self.best_x, self.best_f
