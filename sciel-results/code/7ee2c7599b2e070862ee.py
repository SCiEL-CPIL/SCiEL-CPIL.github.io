"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Exploits the fact that f5 (Linear slope) is a linear function.
    
    Strategy:
    1. Evaluate at the origin (clipped to box center, which is 0).
    2. Evaluate at origin + h * e_i for each axis i (6 points total: 1 origin + 5 axes).
       Wait, the box is [-5, 5]^5. The origin 0 is inside.
       We can evaluate at 0, and at 0 + h*e_i and 0 - h*e_i? 
       The directive says "single 6-point simplex evaluation (origin plus 5 unit axis probes)".
       A simplex in 5D has 6 vertices. But a linear function is determined by 6 points 
       if they are affinely independent. 
       
       Actually, for a linear function f(x) = c + g^T x, we need to find g and c.
       f(0) = c.
       f(h * e_i) = c + h * g_i.
       So g_i = (f(h * e_i) - c) / h.
       
       This requires 1 + 5 = 6 evaluations.
       Then we know the gradient g exactly.
       
       The minimum of a linear function on a hypercube [-5, 5]^5 is at the vertex 
       where x_i = -5 * sign(g_i) (if g_i != 0, else any value, but sign(0) is 0, 
       so we can pick 0 or either; let's pick 0 or just use -5*sign, if g_i=0, 
       sign is 0, x_i=0, which is fine).
       
       Wait, if g_i > 0, we want x_i as small as possible, so x_i = -5.
       If g_i < 0, we want x_i as large as possible, so x_i = 5.
       If g_i == 0, x_i doesn't matter, let's say 0.
       
       So x_opt[i] = -5 if g_i > 0, 5 if g_i < 0, 0 if g_i == 0.
       Or more simply: x_opt[i] = -5 * np.sign(g_i) if g_i != 0 else 0.
       Actually, -5 * np.sign(g_i) gives -5 for g_i>0, 5 for g_i<0, 0 for g_i=0. Perfect.
       
       So:
       Call 1: f(0)
       Calls 2-6: f(h * e_i) for i=0..4, with some small h (e.g., 0.1 or 1.0).
       Compute g.
       Call 7: f(x_opt). This should be the global minimum.
       
       But wait, the problem says "func(x) clips a finite length-5 vector to the box".
       So if we evaluate at h*e_i, and h is small, it's within the box.
       We have budget 500. We use 7 calls. The rest of the calls?
       The scoring is AOCC over 500 calls. If we make 7 calls and then stop, 
       the remaining 493 calls are "repeating the last n_i". 
       So if we get the exact minimum at call 7, then b_i for i>=7 is f_opt.
       e_i = clip(f_opt - f_opt, ...) = 1e-8.
       n_i = (log10(1e-8) + 8)/10 = (-8+8)/10 = 0.
       So for i>=7, n_i = 0.
       For i=1..6, n_i might be larger.
       
       Let's check the first few calls:
       Call 1: x=0. b_1 = f(0). e_1 = |f(0)-f_opt|. 
       f(0) = c. f_opt = c + g^T x_opt.
       Since x_opt is chosen to minimize g^T x, g^T x_opt = -5 * sum(|g_i|).
       So f(0) - f_opt = 5 * sum(|g_i|). This could be large.
       
       But after call 7, we are at the optimum. So n_7 = 0, n_8 = 0, ..., n_500 = 0.
       The mean n_i will be (sum of n_1..n_6 + 0) / 500.
       This should be very small, giving AOCC close to 1.
       
       We should make the remaining calls also evaluate the optimum (or just return early? 
       No, __call__ is called once, and inside we can loop. But the evaluator calls __call__ once, 
       and inside we make up to 500 func calls.
       
       Actually, we can just make 7 calls, and then for the remaining 493 calls, 
       we can just call func(x_opt) again? Or we can stop? 
       The problem says "More than 500 calls ... makes the whole result invalid". 
       So we must make at most 500 calls. 
       If we make 7 calls, that's fine. The scoring repeats the last n_i.
       
       But to be safe, and to ensure b_i is as good as possible for all i, 
       we can just evaluate x_opt for all remaining calls. 
       Or we can just break after 7 calls? 
       The loop in the current program goes for budget-1 times. 
       We can structure it as:
       - Do 7 specific calls.
       - Then for the remaining budget-7 calls, call func(x_opt) (which is redundant but ensures 
         that if for some reason the 7th call wasn't exactly f_opt due to floating point, 
         we are still at the best point).
       
       Actually, the 7th call IS x_opt. So b_7 = f(x_opt) = f_opt (approximately).
       So we can just make 7 calls and then stop? But the loop must run for budget times? 
       No, we can just control the number of calls.
       
       Let's implement:
       x0 = np.zeros(dim)
       f0 = func(x0)
       h = 0.1  # small step
       g = np.zeros(dim)
       for i in range(dim):
           xi = np.zeros(dim)
           xi[i] = h
           fi = func(xi)
           g[i] = (fi - f0) / h
       
       # Determine x_opt
       x_opt = -5.0 * np.sign(g)
       # If g[i] is 0, sign is 0, x_opt[i] = 0. That's fine.
       f_opt = func(x_opt)  # Call 7
       
       # For the remaining calls, just evaluate x_opt to be safe (in case of numerical issues)
       # But we've already used 7 calls. Remaining: budget - 7
       for _ in range(self.budget - 7):
           # We can just call func(x_opt) again, or do nothing.
           # Calling func(x_opt) again is safe and ensures b_i is f_opt.
           _ = func(x_opt)
       
       return x_opt, f_opt
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Step 1: Evaluate at origin
        x0 = np.zeros(dim)
        f0 = func(x0)
        
        # Step 2: Evaluate along each axis to compute gradient
        h = 0.1
        g = np.zeros(dim)
        for i in range(dim):
            xi = np.zeros(dim)
            xi[i] = h
            fi = func(xi)
            g[i] = (fi - f0) / h
        
        # Step 3: Compute optimal vertex
        # For linear function, minimum is at the vertex opposing the gradient
        x_opt = -5.0 * np.sign(g)
        # Clip to be safe (though -5*sign is already in [-5,5])
        x_opt = np.clip(x_opt, -5.0, 5.0)
        
        # Step 4: Evaluate at optimal vertex
        f_opt = func(x_opt)
        
        # Step 5: Use remaining budget to evaluate x_opt (ensures best value is recorded)
        # Total calls so far: 1 (origin) + dim (axes) + 1 (opt) = dim + 2 = 7 for dim=5
        remaining = budget - (dim + 2)
        for _ in range(max(0, remaining)):
            _ = func(x_opt)
        
        return x_opt, f_opt
# EVOLVE-BLOCK-END
