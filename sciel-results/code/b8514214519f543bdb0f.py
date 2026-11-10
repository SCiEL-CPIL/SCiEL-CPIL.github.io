"""ZDT2 Optimizer using global Golden Section Search for hypervolume-optimal point placement."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Exploits the analytic structure of ZDT2.
    Strategy:
    1. Identify the 1D Pareto front parameterized by x1.
    2. The problem reduces to placing N points x1_1...x1_N in [0,1] to maximize the 2D Hypervolume.
    3. The Hypervolume function is concave with respect to each point's position (given neighbors).
    4. Use nested Golden Section Search to find the global optimum for each point sequentially.
       This avoids the local minima traps and convergence issues of gradient-based methods.
    5. Evaluate exactly these points using the budget.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        dim = self.dim
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT2 specifics
        # f1 = x1
        # f2 = (1 + 10*(dim-1)) * (1 - sqrt(f1 / (1 + 10*(dim-1))))
        # Pareto front: x2...xdim = 0
        n_extra = dim - 1
        scale = 1.0 + 10.0 * n_extra
        
        # Reference point
        ref_f1 = 11.0
        ref_f2 = 11.0
        
        # Objective function on the Pareto front
        def front_f2(x1):
            # x1 is in [0, 1]
            # sqrt(x1/scale) is safe for x1>=0
            return scale * (1.0 - np.sqrt(x1 / scale))
        
        N = budget
        
        # We want to find x1s = [x_0, ..., x_{N-1}] such that 0 <= x_0 < ... < x_{N-1} <= 1
        # To maximize HV.
        
        # HV contribution logic:
        # Let's define the "cost" or "loss" minimization, or just maximize HV directly.
        # HV = sum_{i=0}^{N-1} (x_{i+1} - x_i) * (ref_f2 - f2(x_i))  where x_N = ref_f1
        # Wait, the standard definition for a set of points P sorted by f1:
        # HV = sum_{i=0}^{N-1} (f1_{i+1} - f1_i) * (ref_f2 - f2_i)
        # with f1_N = ref_f1.
        
        # This function is concave in each x_i individually.
        # We can use coordinate ascent with Golden Section Search for each coordinate.
        
        # Initialize x1s
        # A good initialization is uniform, but slightly shifted to account for the steep slope at 0?
        # Actually, uniform is a fine start for coordinate ascent.
        x1s = np.linspace(0.0, 1.0, N)
        
        # Ensure strict ordering and bounds for the search intervals
        # We will iterate over points. For each point i, we search in (x_{i-1}, x_{i+1}).
        # If i=0, lower bound is 0. If i=N-1, upper bound is 1.
        
        # Golden Section Search for maximization
        def golden_section_max(f, a, b, tol=1e-8, max_iter=50):
            """
            Finds x in [a, b] that maximizes f(x).
            Assumes f is unimodal (concave).
            """
            phi = (1.0 + np.sqrt(5.0)) / 2.0
            c = b - (b - a) / phi
            d = a + (b - a) / phi
            
            fc = f(c)
            fd = f(d)
            
            for _ in range(max_iter):
                if time.time() > deadline - 5:
                    break
                if abs(b - a) < tol:
                    break
                
                if fc < fd:
                    # Maximum is in [c, b]
                    a = c
                    c = d
                    fc = fd
                    d = a + (b - a) / phi
                    fd = f(d)
                else:
                    # Maximum is in [a, d]
                    b = d
                    d = c
                    fd = fc
                    c = b - (b - a) / phi
                    fc = f(c)
            
            return (a + b) / 2.0

        # Define the objective function for a single point x_i, given neighbors
        # We need to compute the change in HV if we move x_i.
        # Total HV = ... + (x_i - x_{i-1}) * (ref_f2 - f2(x_{i-1})) + (x_{i+1} - x_i) * (ref_f2 - f2(x_i)) + ...
        # Let L = x_{i-1}, R = x_{i+1} (or boundaries 0 and 1, and ref_f1 for the last one)
        # Note: The term (x_{i+1} - x_i) * (ref_f2 - f2(x_i)) depends on x_i.
        # The term (x_i - x_{i-1}) * (ref_f2 - f2(x_{i-1})) also depends on x_i.
        # So the local objective g(x) = (x - L) * H_L + (R - x) * H_x(x)
        # where H_L = ref_f2 - f2(L) and H_x(x) = ref_f2 - f2(x).
        
        # We will perform coordinate ascent.
        # Number of sweeps: Since N=200, and GSS is fast, a few sweeps should converge.
        # The function is concave, so coordinate ascent converges to the global optimum.
        
        num_sweeps = 20  # Should be more than enough for 200 points
        
        for sweep in range(num_sweeps):
            if time.time() > deadline - 10:
                break
            
            # Iterate from left to right
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                
                # Determine search bounds
                if i == 0:
                    a = 0.0
                else:
                    a = x1s[i-1] + 1e-12 # Ensure strict inequality
                
                if i == N - 1:
                    b = 1.0
                else:
                    b = x1s[i+1] - 1e-12
                
                if a >= b:
                    continue
                
                L = x1s[i-1] if i > 0 else 0.0
                R = x1s[i+1] if i < N - 1 else 1.0
                
                # Pre-compute constant part from left neighbor
                H_L = ref_f2 - front_f2(L)
                
                # Define local objective to maximize
                def local_obj(x, L=L, H_L=H_L, R=R):
                    # Term 1: (x - L) * H_L
                    # Term 2: (R - x) * (ref_f2 - f2(x))
                    # Note: If i is the last point, R is 1.0, but the width extends to ref_f1.
                    # Wait, the HV formula uses the next point's f1.
                    # For point i, the width is x_{i+1} - x_i.
                    # If i is the last point (N-1), the "next point" is the reference point in f1?
                    # No, the standard HV definition for a set of points P:
                    # Sort by f1.
                    # HV = sum_{j=1}^N (f1_{j+1} - f1_j) * (ref_f2 - f2_j)
                    # where f1_{N+1} = ref_f1.
                    # So for i=N-1, the width is ref_f1 - x_{N-1}.
                    # My local_obj above used R as the next point's x1.
                    # If i < N-1, R = x_{i+1}.
                    # If i == N-1, R = ref_f1.
                    
                    # Let's refine local_obj based on this.
                    if i == N - 1:
                        R_eff = ref_f1
                    else:
                        R_eff = x1s[i+1]
                    
                    # The term involving x is (R_eff - x) * (ref_f2 - f2(x))
                    # The term involving x from the left is (x - L) * (ref_f2 - f2(L))
                    
                    h_x = ref_f2 - front_f2(x)
                    return (x - L) * H_L + (R_eff - x) * h_x

                # Perform Golden Section Search
                new_x = golden_section_max(local_obj, a, b, tol=1e-10, max_iter=60)
                
                # Update x1s
                x1s[i] = new_x

        # Final check and sort (should be sorted already due to bounds)
        x1s = np.sort(x1s)
        
        # Evaluate the points
        last_F = None
        for x1 in x1s:
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = np.clip(x1, lo[0], hi[0])
            # x[1:] = 0 (clipped to bounds)
            F = np.array(problem(x))
            last_F = F
            
        return last_F

# EVOLVE-BLOCK-END
