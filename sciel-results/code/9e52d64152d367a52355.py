"""
Strategy: Structured Geometric Initialization + Nelder-Mead Direct Search.

The current program uses a Gaussian Mixture Model (GMM) parameterization and L-BFGS-B.
This strategy abandons the parametric constraint and optimizes the raw 50-dimensional 
height vector directly using Nelder-Mead (a derivative-free method).

Key differences:
1. No parametric model: h is not forced to be a sum of Gaussians. It can take any 
   non-negative shape (e.g., sharp peaks, plateaus, asymmetric bumps).
2. Derivative-free: Nelder-Mead does not rely on gradients, making it robust to the 
   non-smoothness introduced by the L1 and Linf norms in the denominator.
3. Structured Initialization: Instead of random starts, we initialize with specific 
   geometric shapes (cosine, triangle, box, super-Gaussian) that are known to be 
   competitive for this type of norm-ratio problem. This guides the search into 
   promising basins immediately.
4. Scale Invariance: We fix the sum of h to 1.0 during optimization to remove the 
   scale degree of freedom, reducing the effective dimensionality and improving 
   numerical stability.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0

# Pre-compute constants for the value function
# The evaluator uses:
# g = np.convolve(h, h)
# m = len(g) = 2*N - 1 = 99
# y = [0, g[0], ..., g[m-1], 0]
# points = np.linspace(-0.5, 0.5, m + 2) -> 101 points
# w = 1.0 / (m + 1) = 0.01
# l2 = sum (w/3) * (y_i^2 + y_i y_{i+1} + y_{i+1}^2) for i=0..m
# norm_1 = sum(|g|) / (m+1)
# norm_inf = max(|g|)
# value = l2 / (norm_1 * norm_inf)

def compute_value(h):
    """Compute the value metric for given heights."""
    h = np.asarray(h, dtype=float)
    if h.size != N:
        # This should not happen if we control the input, but safety check
        return 0.0
    
    if np.any(h < 0) or not np.all(np.isfinite(h)):
        return 0.0
    
    # Scale invariance: if sum(h) is 0, value is 0.
    # We enforce sum(h)=1 in the optimizer, but compute_value should handle any scale.
    # However, for stability, we can normalize h inside if needed, but the evaluator 
    # uses the raw h. Since value is scale invariant, we can just compute.
    
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    w = 1.0 / (m + 1)
    
    # Vectorized L2 term calculation
    # y has length m+2
    # Terms are for i in 0..m: y[i], y[i+1]
    y_curr = y[:m+1]
    y_next = y[1:m+2]
    
    l2_terms = y_curr**2 + y_curr * y_next + y_next**2
    l2 = (w / 3.0) * np.sum(l2_terms)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    
    value = l2 / (norm_1 * norm_inf)
    return value

def normalize_h(h):
    """Normalize h so that sum(h) = 1.0, ensuring non-negativity."""
    h = np.maximum(0, h)
    s = np.sum(h)
    if s == 0:
        return np.ones(N) / N
    return h / s

def get_initial_guesses():
    """Generate diverse structured initial guesses for the 50 heights."""
    x = np.linspace(-0.25, 0.25, N)
    guesses = []
    
    # 1. Uniform (Seed)
    guesses.append(np.ones(N))
    
    # 2. Cosine bump (Gaussian-like but bounded support)
    guesses.append(np.cos(np.pi * x / 0.5)**2)
    
    # 3. Triangle (Linear decay from center)
    guesses.append(np.maximum(0, 1 - np.abs(x) / 0.25))
    
    # 4. Box (Constant in center, 0 outside)
    guesses.append((np.abs(x) < 0.125).astype(float))
    
    # 5. Super-Gaussian (Sharper peak)
    guesses.append(np.exp(-10 * x**2))
    
    # 6. Wide Super-Gaussian
    guesses.append(np.exp(-2 * x**2))
    
    # 7. Bimodal (Two peaks)
    guesses.append(np.exp(-50 * (x - 0.1)**2) + np.exp(-50 * (x + 0.1)**2))
    
    # 8. Asymmetric Ramp
    guesses.append(np.maximum(0, 1 - (x + 0.25) / 0.5))
    
    # 9. Inverse Cosine (Plateau-like)
    guesses.append(np.cos(np.pi * x / 0.5))
    
    # 10. Random Smooth (spline interpolation of few points)
    pts = np.random.uniform(0.1, 1.0, 5)
    knots = np.linspace(-0.25, 0.25, 5)
    guesses.append(np.interp(x, knots, pts))
    
    # Normalize all guesses
    guesses = [normalize_h(g) for g in guesses]
    
    # Add some random perturbations to the best structured ones
    for g in guesses[:5]:
        perturbed = g + np.random.normal(0, 0.05, N)
        guesses.append(normalize_h(perturbed))
        
    return guesses

def construct():
    start_time = time.time()
    
    def objective(h):
        """Objective function for minimization (maximizing value)."""
        # h is the raw vector. We will normalize it to handle scale invariance
        # and ensure non-negativity is handled by the optimizer bounds or projection.
        # Nelder-Mead doesn't handle bounds natively, so we use a penalty or projection.
        # Given the scale invariance, we can just normalize h inside the objective.
        # However, Nelder-Mead works best in unconstrained space.
        # Let's optimize on the raw h, but normalize inside the value calculation?
        # No, the evaluator uses raw h. But value is scale invariant.
        # So we can fix sum(h)=1.
        # We will use a transformation: h = exp(z) / sum(exp(z)).
        # This ensures h > 0 and sum(h) = 1.
        # This maps R^N to the simplex.
        
        # Wait, Nelder-Mead on 50D is slow.
        # Is there a better way?
        # The current program uses L-BFGS-B on 50D.
        # Let's stick to Nelder-Mead but with a very tight budget per start, 
        # or use COBYLA which handles inequality constraints (h >= 0).
        # Actually, let's use the log-sum-exp trick to enforce positivity and 
        # normalize. We optimize z in R^N.
        
        z = h
        # Avoid overflow
        z_max = np.max(z)
        exp_z = np.exp(z - z_max)
        h_norm = exp_z / np.sum(exp_z)
        
        val = compute_value(h_norm)
        return -val

    best_val = 0.0
    best_h = np.ones(N) / N
    
    # Get initial guesses
    guesses = get_initial_guesses()
    
    # For each guess, run Nelder-Mead on the log-parameters
    # h = softmax(z). We start with z = log(h_guess).
    
    for guess in guesses:
        if time.time() > DEADLINE - 10:
            break
            
        # Initialize z
        # To avoid log(0), add small epsilon
        h_safe = np.maximum(guess, 1e-6)
        z0 = np.log(h_safe)
        
        # Run Nelder-Mead
        # Nelder-Mead is derivative-free.
        # We set maxiter to keep it fast.
        try:
            res = minimize(
                objective,
                z0,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-6,
                    'fatol': 1e-6,
                    'adaptive': True
                }
            )
            
            # Convert back to h
            z_opt = res.x
            z_max = np.max(z_opt)
            exp_z = np.exp(z_opt - z_max)
            h_opt = exp_z / np.sum(exp_z)
            
            val_opt = compute_value(h_opt)
            
            if val_opt > best_val:
                best_val = val_opt
                best_h = h_opt
                
        except Exception as e:
            pass

    # Final polish: Try L-BFGS-B on the best h found, with bounds
    if time.time() < DEADLINE - 5:
        try:
            def obj_lbfgs(h):
                # h is raw, we normalize inside? 
                # If we use bounds [0, inf), we don't need to normalize for the optimizer,
                # but the value is scale invariant.
                # L-BFGS-B might drift in scale.
                # Let's fix the scale by normalizing h in the objective.
                h_norm = normalize_h(h)
                return -compute_value(h_norm)
            
            res = minimize(
                obj_lbfgs,
                best_h,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={
                    'maxiter': 200,
                    'ftol': 1e-8
                }
            )
            
            if res.fun < -best_val:
                h_opt = normalize_h(res.x)
                val_opt = compute_value(h_opt)
                if val_opt > best_val:
                    best_val = val_opt
                    best_h = h_opt
        except Exception:
            pass

    # Ensure the returned h is valid
    best_h = np.maximum(0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    
    # The evaluator computes value on the returned h.
    # Since value is scale invariant, the scale of best_h doesn't matter.
    # But we should return a reasonable scale.
    # The seed returns 50 equal heights. Let's return our optimized shape.
    
    return best_h.tolist()

if __name__ == "__main__":
    h = construct()
    print(f"Score: {compute_value(h)}")
