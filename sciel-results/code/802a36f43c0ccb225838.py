"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Ensure inputs are float64 arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    
    # Record start time for deadline guard
    start_time = time.time()
    # The contract says 120s per evaluation, but the kill is 150s.
    # We need to be safe. Let's aim for 110s to be very safe against overhead.
    deadline = start_time + 110.0
    
    n = len(x)
    
    # Strategy:
    # The hidden function is likely rational (e.g., 1/(x+y+z) or similar) or a polynomial.
    # The current quadratic basis (score 7.39) is too low capacity.
    # A pure RBF basis (score 2.08) is high capacity but might be overfitting or the centers are poorly placed.
    # The "inspiration" mentions a "reciprocal/interaction basis" with score 9.96 (worse than 7.39? No, wait. 
    # The prompt says "Measured: 0 -> 9.964" for a program that improved island 1. 
    # Wait, the parent score was 7.39. A score of 9.96 is WORSE. 
    # Let me re-read: "Switching from a mismatched RBF model to a correctly sized reciprocal/interaction basis aligns with the known superior performance of this feature set (score 9.96 vs 6.38) for the hidden function's likely rational structure."
    # This is confusing. 9.96 > 6.38. Lower is better. So 6.38 is better.
    # If the RBF model was "mismatched" and scored 9.96, and the reciprocal basis scored 6.38, then the reciprocal basis is better.
    # But our current score is 7.39. So 6.38 is an improvement.
    # The "inspiration 3" with score 2.08 used RBF. So RBF *can* be good.
    # Why did the RBF in inspiration 3 get 2.08? It used "internal, time-bounded multi-start BFGS optimization to fit the RBF weights".
    # Our current program just passes `params` to the evaluator's BFGS. It does NOT do internal optimization.
    # The evaluator fits `params` (length 10). 
    # If I use an RBF basis, I need to fit the weights. The evaluator's BFGS will fit them.
    # But RBF centers are fixed. If I fix centers, I only fit weights.
    # Inspiration 3 used internal optimization to fit *weights*? Or did it fit centers?
    # "fit the RBF weights". 
    # If I use a fixed RBF basis, the evaluator's BFGS will fit the 10 weights.
    # The issue might be that 10 RBF centers are not enough, or the initialization (all ones) is bad.
    # Let's try a different basis: A combination of polynomial and reciprocal terms.
    # Specifically, terms like 1/x, 1/y, 1/z, 1/(x+y), etc.
    # Or, a higher order polynomial?
    # Let's try a 3rd order polynomial. That has 10 terms:
    # x, y, z, xy, xz, yz, x^2, y^2, z^2, xyz.
    # This is a natural extension of the quadratic (which had x,y,z,xy,xz,yz,x^2,y^2,z^2 -> 8 terms + bias = 9? No, current has 10 params: bias, x, y, z, xy, xz, yz, x^2, y^2, z^2).
    # The current quadratic is missing xyz and cubic terms.
    # Let's try a basis that includes reciprocal terms, as the "rational structure" hint suggests.
    
    # Basis functions:
    # 0: 1
    # 1: x
    # 2: y
    # 3: z
    # 4: 1/x
    # 5: 1/y
    # 6: 1/z
    # 7: 1/(x+y)
    # 8: 1/(y+z)
    # 9: 1/(x+z)
    
    # This covers linear and simple rational structures.
    # If the function is 1/(x+y+z), this basis might approximate it well?
    # 1/(x+y+z) is not linear in 1/x, 1/y, 1/z.
    # But it's a start.
    
    # Alternatively, let's try the RBF approach from Inspiration 3 but ensure it's robust.
    # Inspiration 3 scored 2.08. That's a big jump from 7.39.
    # Let's replicate the RBF strategy but with fixed centers to let the evaluator's BFGS do the work.
    # If I fix centers, the problem is linear in the weights. BFGS should handle it easily.
    # The risk is that the fixed centers are bad.
    # Let's place centers on a grid within the domain [0.5, 2].
    
    # Define 10 RBF centers.
    # We can use a small grid. 
    # x, y, z in {0.75, 1.25, 1.75}.
    # That's 27 points. We only have 10 params.
    # Let's pick 10 diverse points.
    # Corners of the cube [0.5, 2]^3: 8 points.
    # Plus center (1.25, 1.25, 1.25).
    # Plus one more? (0.5, 0.5, 0.5) is a corner.
    # Let's list 10 centers:
    centers = np.array([
        [0.5, 0.5, 0.5],
        [2.0, 0.5, 0.5],
        [0.5, 2.0, 0.5],
        [0.5, 0.5, 2.0],
        [2.0, 2.0, 0.5],
        [2.0, 0.5, 2.0],
        [0.5, 2.0, 2.0],
        [2.0, 2.0, 2.0],
        [1.25, 1.25, 1.25],
        [1.0, 1.5, 1.0] # Arbitrary extra point
    ])
    
    # Width of RBF. 
    # Domain size is 1.5. 
    # sigma = 0.5 seems reasonable.
    sigma = 0.5
    
    # Precompute RBF values for all points? 
    # We have n points. 10 centers.
    # This is fast.
    
    # Compute distances
    # x, y, z are arrays of length n.
    # centers is (10, 3).
    # We need dist^2 for each point i and center j.
    
    # dx = x[:, None] - centers[None, :, 0]
    # dy = y[:, None] - centers[None, :, 1]
    # dz = z[:, None] - centers[None, :, 2]
    # d2 = dx**2 + dy**2 + dz**2
    # phi = np.exp(-d2 / (2 * sigma**2))
    
    # This creates an (n, 10) matrix.
    # Then pred = phi @ params.
    
    # This is a linear model in params. 
    # The evaluator's BFGS will fit params.
    # Since it's linear, BFGS should converge quickly.
    # The main risk is if the basis is not rich enough.
    # 10 RBFs with sigma=0.5 on a 1.5x1.5x1.5 domain.
    # This should be quite flexible.
    
    # Let's check the time.
    # n=200 (train) or 200 (test).
    # (200, 10) matrix ops are trivial.
    
    # However, there is a catch. 
    # The evaluator fits `params` starting from all ones.
    # If the RBFs are very similar, the Hessian might be ill-conditioned.
    # But BFGS is robust to some extent.
    
    # Let's try this.
    
    # But wait, Inspiration 3 used *internal* optimization. 
    # "use an internal, time-bounded multi-start BFGS optimization to fit the RBF weights".
    # This implies the evaluator's BFGS might not be enough, or they wanted to fit *centers*?
    # "fit the RBF weights".
    # If the weights are the only unknowns, and they are linear, the evaluator's BFGS is fine.
    # Unless... the "params" passed to equation are NOT the RBF weights?
    # No, "params, an array of length 10".
    # So the evaluator fits these 10 params.
    # If I define the basis, the evaluator fits the coefficients.
    
    # Why would Inspiration 3 do internal optimization?
    # Maybe to fit the *centers*? But centers are not in `params`.
    # Or maybe to handle the case where the evaluator's BFGS fails?
    # Or maybe the basis is non-linear in params?
    # If I use `phi = exp(-d2 / (2 * sigma**2))`, and sigma is fixed, it's linear in weights.
    # If I let sigma be a param, it's non-linear. But I only have 10 params.
    # If I use 1 RBF per param, and fix centers and sigma, it's linear.
    
    # Let's stick to the linear RBF basis. It's the most likely to work with the evaluator's BFGS.
    
    # One concern: The domain is [0.5, 2]. 
    # If the function has a singularity or sharp feature, RBFs might need to be closer.
    # But with 10 RBFs, we are limited.
    
    # Let's try a hybrid: 5 RBFs + 5 Polynomial terms?
    # Or just 10 RBFs.
    
    # Let's implement the RBF basis.
    
    # Check time limit
    if time.time() > deadline:
        # Fallback to simple linear if too slow (unlikely)
        return np.ones(n)
        
    # Compute RBF matrix
    # Vectorized for speed
    cx, cy, cz = centers[:, 0], centers[:, 1], centers[:, 2]
    
    # x is (n,), cx is (10,)
    # dx2 = (x[:, None] - cx[None, :])**2
    # This is (n, 10)
    
    dx2 = (x[:, None] - cx[None, :])**2
    dy2 = (y[:, None] - cy[None, :])**2
    dz2 = (z[:, None] - cz[None, :])**2
    
    d2 = dx2 + dy2 + dz2
    
    # sigma = 0.5
    # exp(-d2 / (2 * 0.25)) = exp(-2 * d2)
    # This is fast.
    
    phi = np.exp(-2.0 * d2)
    
    # phi is (n, 10)
    # params is (10,)
    # pred = phi @ params
    
    # Check for non-finite values
    if not np.all(np.isfinite(phi)):
        # This shouldn't happen with exp, but just in case
        phi = np.nan_to_num(phi)
        
    pred = phi @ params
    
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred)
        
    return pred

# EVOLVE-BLOCK-END
