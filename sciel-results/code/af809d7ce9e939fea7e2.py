"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad
import sys

# Deadline guard
START_TIME = time.time()
DEADLINE = 85  # seconds, leave 5s margin

# Representative cosmologies (w0, wa) for training/validation of the rule
# We use a diverse set to ensure the rule is "universal"
COSMOLOGIES = [
    (-1.0, 0.0),   # LambdaCDM
    (-1.0, 0.5),   # Quintessence-like
    (-0.9, 0.3),   # General CPL
    (-1.1, 0.0),   # Phantom
    (-0.5, 0.0),   # Matter dominated limit
    (-1.0, -0.5),  # Different wa sign
    (-0.8, 0.2),   # Intermediate
    (-1.2, 0.1),   # More phantom
    (-0.95, 0.1),  # Near LCDM
    (-1.0, 0.2),   # Mild quintessence
    (-1.0, 0.8),   # Strong quintessence
    (-0.7, 0.4),   # Another variant
]

# Redshift grid for evaluation
# Use a dense grid to capture variations well
Z_VALS = np.linspace(0.05, 3.0, 50)

def E_z(z, w0, wa):
    """Calculate E(z) = H(z)/H0 for flat universe with CPL w(z) = w0 + wa*(1 - 1/(1+z))"""
    if z < 0:
        return 1.0
    # E(z)^2 = (1+z)^(-3*w0) * (1 - wa * z / (1+z))^(1.5*wa)
    term1 = (1 + z) ** (-3 * w0)
    base = 1 - wa * z / (1 + z)
    if base < 0:
        base = 1e-12
    term2 = base ** (1.5 * wa)
    return np.sqrt(term1 * term2)

def true_integral(z, w0, wa):
    """Calculate the true integral part: int_0^1 dx / E(z*x)"""
    if z == 0:
        return 1.0
    def integrand(x):
        return 1.0 / E_z(z * x, w0, wa)
    val, _ = quad(integrand, 0, 1, limit=100)
    return val

def quadrature_approx(nodes, weights, z, w0, wa):
    """Calculate the quadrature approximation: sum w_i / E(z * x_i)"""
    if z == 0:
        return 1.0
    z_vals = z * nodes
    E_vals = np.array([E_z(zv, w0, wa) for zv in z_vals])
    return np.sum(weights / E_vals)

def smooth_objective(params):
    """
    Objective function to minimize: Mean Squared Error (MSE) of relative error.
    params: [x1, ..., x8, w1, ..., w8]
    """
    if time.time() > DEADLINE - 5:
        return 1e9

    x = params[:8]
    w = params[8:]
    
    # Enforce constraints via penalty or transformation?
    # L-BFGS-B handles bounds, but strict increase and sum(w)=1 are tricky.
    # We will use a transformation for weights: w_i = exp(v_i) / sum(exp(v_j))
    # For nodes, we will use a transformation to ensure strict increase in [0,1].
    # Let y_i be the parameters for nodes. x_1 = y_1, x_{i+1} = x_i + exp(y_{i+1})
    # We need to map 8 node params to 8 nodes.
    # Actually, simpler: just penalize violations heavily in the objective.
    # But L-BFGS-B works best with smooth objectives.
    
    # Let's stick to the raw params but add a smooth penalty for constraints.
    # 1. x_i in [0,1]
    # 2. x_i strictly increasing
    # 3. w_i > 0
    # 4. sum(w_i) = 1
    
    # Smooth penalty for x bounds and increase
    pen_x = 0.0
    for i in range(8):
        if x[i] < 0: pen_x += (x[i])**2
        if x[i] > 1: pen_x += (x[i]-1)**2
    for i in range(7):
        diff = x[i+1] - x[i]
        if diff < 1e-4:
            pen_x += (1e-4 - diff)**2
    
    # Smooth penalty for weights
    pen_w = 0.0
    w_sum = np.sum(w)
    # Penalty for sum not being 1
    pen_w += (w_sum - 1.0)**2
    # Penalty for negative weights
    for i in range(8):
        if w[i] < 1e-6:
            pen_w += (1e-6 - w[i])**2
            
    if pen_x > 1e-3 or pen_w > 1e-3:
        return pen_x + pen_w + 1e6 # Large penalty
    
    # Normalize weights for calculation, but optimize raw w?
    # If we optimize raw w and penalize sum!=1, the gradient might be noisy.
    # Better to optimize 7 weights and derive the 8th? Or use softmax?
    # Let's try optimizing raw weights with the penalty. It's a common trick.
    
    w_norm = w / w_sum if w_sum > 1e-6 else np.full(8, 0.125)
    
    mse = 0.0
    count = 0
    for z in Z_VALS:
        for w0, wa in COSMOLOGIES:
            true_val = true_integral(z, w0, wa)
            if true_val < 1e-10:
                continue
            approx_val = quadrature_approx(x, w_norm, z, w0, wa)
            rel_err = (approx_val - true_val) / true_val
            mse += rel_err**2
            count += 1
            
    if count == 0:
        return 1e6
        
    return mse / count + pen_x + pen_w

def minimax_objective(params):
    """
    Objective function to minimize: Worst-case relative error.
    """
    if time.time() > DEADLINE - 5:
        return 1e9

    x = params[:8]
    w = params[8:]
    
    # Basic constraint check (hard fail if violated)
    if np.any(x < 0) or np.any(x > 1):
        return 1e9
    if np.any(np.diff(x) <= 1e-6):
        return 1e9
    if np.any(w <= 0):
        return 1e9
        
    w_sum = np.sum(w)
    if w_sum < 1e-6:
        return 1e9
    w_norm = w / w_sum
    
    max_rel_err = 0.0
    for z in Z_VALS:
        for w0, wa in COSMOLOGIES:
            true_val = true_integral(z, w0, wa)
            if true_val < 1e-10:
                continue
            approx_val = quadrature_approx(x, w_norm, z, w0, wa)
            rel_err = abs(approx_val - true_val) / true_val
            if rel_err > max_rel_err:
                max_rel_err = rel_err
                
    return max_rel_err

def construct():
    # Initial guess: Gauss-Legendre nodes and weights
    # This is a very strong starting point for smooth integrands
    x_init, w_init = np.polynomial.legendre.leggauss(8)
    # leggauss returns nodes in [-1, 1] and weights summing to 2
    # Map to [0, 1]
    x_init = 0.5 * (x_init + 1)
    w_init = w_init / 2.0
    
    params_init = np.concatenate([x_init, w_init])
    
    bounds = [(0.0, 1.0)] * 8 + [(1e-6, 1.0)] * 8
    
    best_params = params_init
    best_score = 1e9
    
    # Phase 1: Smooth optimization (MSE)
    try:
        if time.time() < DEADLINE - 20:
            result1 = minimize(
                smooth_objective, 
                params_init, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            if result1.fun < best_score:
                best_score = result1.fun
                best_params = result1.x
    except Exception as e:
        pass
        
    # Phase 2: Refinement with Minimax (Worst-case error)
    # Start from the result of Phase 1
    try:
        if time.time() < DEADLINE - 5:
            result2 = minimize(
                minimax_objective, 
                best_params, 
                method='Powell', 
                bounds=bounds,
                options={'maxiter': 50, 'maxfev': 1000}
            )
            if result2.fun < best_score:
                best_score = result2.fun
                best_params = result2.x
    except Exception as e:
        pass

    # Extract nodes and weights
    x = best_params[:8]
    w = best_params[8:]
    
    # Ensure strict increase and bounds
    x = np.sort(x)
    x = np.clip(x, 1e-6, 1 - 1e-6)
    for i in range(1, 8):
        if x[i] <= x[i-1]:
            x[i] = x[i-1] + 1e-6
    x = np.clip(x, 1e-6, 1 - 1e-6)
    
    # Normalize weights
    w = np.abs(w)
    w_sum = np.sum(w)
    if w_sum < 1e-12:
        w = np.full(8, 1.0/8.0)
    else:
        w = w / w_sum
        
    return {"nodes": x, "weights": w}
# EVOLVE-BLOCK-END
