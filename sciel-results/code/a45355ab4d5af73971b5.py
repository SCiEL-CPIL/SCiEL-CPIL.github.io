import numpy as np
from scipy.optimize import minimize, differential_evolution

def construct():
    # Precompute Hermite polynomials (probabilists' He_n)
    # We will evaluate the function h(x) = P(x) * exp(-x^2/2) directly.
    # P(x) = c0*He0 + c4*He4 + c8*He8 + c12*He12
    
    def he0(x):
        return np.ones_like(x)
    
    def he4(x):
        return x**4 - 6*x**2 + 3
    
    def he8(x):
        return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    
    def he12(x):
        return (x**12 - 66*x**10 + 1155*x**8 - 6930*x**6 + 
                13860*x**4 - 7560*x**2 + 10395)
    
    def compute_c12(c0, c4, c8):
        return -(c0 + 3*c4 + 105*c8) / 10395.0

    # Use a fine grid for accurate max estimation
    # The function decays as exp(-x^2/2), so range [-10, 10] is sufficient.
    # 20000 points gives good resolution.
    x_grid = np.linspace(-12, 12, 20000)
    exp_factor = np.exp(-x_grid**2 / 2)
    
    # Precompute Hermite values on the grid to speed up objective function
    He0_grid = he0(x_grid)
    He4_grid = he4(x_grid)
    He8_grid = he8(x_grid)
    He12_grid = he12(x_grid)
    
    def objective(c):
        c0, c4, c8 = c
        c12 = compute_c12(c0, c4, c8)
        
        # Compute P(x) on the grid
        P = c0 * He0_grid + c4 * He4_grid + c8 * He8_grid + c12 * He12_grid
        
        # Compute h(x) = P(x) * exp(-x^2/2)
        h = P * exp_factor
        
        # The metric to minimize is max |h(x)|
        # The final score will be 2*pi*(max|h|)^2, but minimizing max|h| is equivalent.
        return np.max(np.abs(h))
    
    # Optimization Strategy:
    # 1. Use Powell's method (derivative-free, different from Nelder-Mead)
    # 2. Multi-start with random initializations
    # 3. Also try a coarse grid search to find good basins
    
    best_val = np.inf
    best_c = None
    
    # Seed point from previous knowledge or random
    # The task mentions a seed program, but we are starting fresh.
    # Let's use a reasonable starting point.
    # c0=1, c4=0, c8=0 -> c12 = -1/10395
    starts = []
    
    # Add a few heuristic starts
    starts.append([1.0, 0.0, 0.0])
    starts.append([0.0, 1.0, 0.0])
    starts.append([0.0, 0.0, 1.0])
    
    # Random starts
    np.random.seed(123)
    for _ in range(30):
        starts.append(np.random.uniform(-2, 2, size=3))
    
    # Powell optimization
    for start in starts:
        try:
            res = minimize(objective, start, method='Powell', 
                          options={'maxiter': 1000, 'xtol': 1e-8, 'ftol': 1e-10})
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x
        except:
            pass
            
    # If Powell didn't find anything good, try Differential Evolution as a fallback
    # (DE was in the original, but let's use it only if needed or with different params? 
    # The directive says "do not reuse the current designs". 
    # The current design used DE as a secondary global search. 
    # Using Powell as the primary engine is the main change. 
    # I will omit DE to be strictly different in structure, or use a different global method like Basin Hopping? 
    # Basin Hopping is good for this. Let's use Basin Hopping.)
    
    try:
        # Use Basin Hopping from a few starts
        for start in starts[:5]:
            bh_res = minimize(objective, start, method='Powell', options={'maxiter': 500})
            # Use basin_hopping wrapper? scipy.optimize.basin_hopping exists
            from scipy.optimize import basin_hopping
            bh = basin_hopping(objective, start, minimizer_kwargs={'method': 'Powell'}, 
                               niter=50, seed=42)
            if bh.fun < best_val:
                best_val = bh.fun
                best_c = bh.x
    except:
        pass

    if best_c is not None:
        return best_c
    else:
        return np.array([1.0, 0.0, 0.0])
