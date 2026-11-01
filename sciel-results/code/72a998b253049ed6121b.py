import numpy as np
from scipy.special import eval_hermite
from scipy.integrate import quad
from scipy.optimize import differential_evolution, minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 180  # 180s limit, keep margin
    
    # Hermite polynomials H_0, H_4, H_8, H_12
    # H_n(x) from scipy.special.eval_hermite
    # Note: eval_hermite(n, x) returns the physicist's Hermite polynomial H_n(x)
    
    H0_0 = eval_hermite(0, 0)
    H4_0 = eval_hermite(4, 0)
    H8_0 = eval_hermite(8, 0)
    H12_0 = eval_hermite(12, 0)
    
    def get_c12(c0, c4, c8):
        # f(0) = (c0*H0_0 + c4*H4_0 + c8*H8_0 + c12*H12_0) * exp(0) = 0
        # c12 = -(c0*H0_0 + c4*H4_0 + c8*H8_0) / H12_0
        return -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0

    def f_of_x(x, c0, c4, c8):
        c12 = get_c12(c0, c4, c8)
        # P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
        P_x = (c0 * eval_hermite(0, x) + 
               c4 * eval_hermite(4, x) + 
               c8 * eval_hermite(8, x) + 
               c12 * eval_hermite(12, x))
        return P_x * np.exp(-np.pi * x**2)

    def objective(params):
        c0, c4, c8 = params
        # Check time
        if time.time() > deadline - 10:
            return 1e10
            
        # Calculate L2 norm squared
        # Integrate from -inf to inf. Use quad.
        # f(x) is smooth and decays fast.
        try:
            l2_sq, _ = quad(lambda x: f_of_x(x, c0, c4, c8)**2, -np.inf, np.inf, limit=50)
            if l2_sq <= 1e-10:
                return 1e10
            norm_factor = 1.0 / np.sqrt(l2_sq)
            
            # Calculate L1 norm of normalized f
            l1, _ = quad(lambda x: abs(f_of_x(x, c0, c4, c8)) * norm_factor, -np.inf, np.inf, limit=50)
            
            # Score is upper bound on A(f)A(Fhat(f)).
            # Assuming A is L1 norm and Fhat is identity for these functions (up to phase), 
            # the product is L1 * L1 = L1^2.
            return l1**2
        except Exception:
            return 1e10

    # Define bounds for differential evolution
    # Seed: [-1.13, 0.04, 0.0003]
    # Let's search in a reasonable range around the seed and wider.
    bounds = [(-5.0, 5.0), (-1.0, 1.0), (-0.1, 0.1)]
    
    best_result = None
    best_score = 1e10
    
    # Run differential evolution
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=50, 
            tol=1e-6,
            popsize=15,
            init='sobol'
        )
        if res.success or res.fun < best_score:
            best_score = res.fun
            best_result = res.x
    except Exception:
        pass
    
    if time.time() < deadline - 20:
        # Refine with local optimization if we have a candidate
        if best_result is not None:
            try:
                res2 = minimize(
                    objective, 
                    best_result, 
                    method='Nelder-Mead', 
                    options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 200}
                )
                if res2.fun < best_score:
                    best_score = res2.fun
                    best_result = res2.x
            except Exception:
                pass
        else:
            # If DE failed, try a grid or random start
            for _ in range(5):
                if time.time() > deadline - 10:
                    break
                start = np.array([np.random.uniform(-2, 2), np.random.uniform(-0.5, 0.5), np.random.uniform(-0.05, 0.05)])
                try:
                    res3 = minimize(
                        objective, 
                        start, 
                        method='Nelder-Mead', 
                        options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 100}
                    )
                    if res3.fun < best_score:
                        best_score = res3.fun
                        best_result = res3.x
                except Exception:
                    pass

    if best_result is not None:
        return np.array(best_result)
    
    # Fallback to seed if optimization fails
    return np.array([-113 / 100, 1 / 25, 1 / 3240])
