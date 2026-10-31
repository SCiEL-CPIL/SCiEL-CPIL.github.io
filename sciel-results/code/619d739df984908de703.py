"""
Search for optimal Hermite polynomial coefficients (degrees 0, 4, 8)
to minimize the upper bound on A(f)*A(Fourier(f)) for the uncertainty inequality.

The test function is g(x) = (a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)) * exp(-x^2/2)
where a12 is chosen so that g(0) = 0.

Uses Simulated Annealing with normalization to handle scale invariance,
followed by a precise local refinement.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite, factorial


def _compute_hermite_moments():
    """
    Precompute the matrix C_{k,l} = integral x^2 H_k(x) H_l(x) exp(-x^2) dx
    for k, l in {0, 4, 8, 12}.
    """
    indices = [0, 4, 8, 12]
    n = len(indices)
    C = np.zeros((n, n))
    
    # Formula for C_{k,l}:
    # Non-zero if l == k, l == k+2, or l == k-2.
    # C_{k, k+2} = 0.25 * 2^(k+2) * (k+2)! * sqrt(pi)
    # C_{k, k}   = -(k + 0.5) * 2^k * k! * sqrt(pi)
    # C_{k, k-2} = k*(k-1) * 2^(k-2) * (k-2)! * sqrt(pi)
    
    sqrt_pi = np.sqrt(np.pi)
    
    for i, k in enumerate(indices):
        for j, l in enumerate(indices):
            if l == k + 2:
                C[i, j] = 0.25 * (2 ** (k + 2)) * factorial(k + 2) * sqrt_pi
            elif l == k:
                C[i, j] = -(k + 0.5) * (2 ** k) * factorial(k) * sqrt_pi
            elif l == k - 2:
                C[i, j] = k * (k - 1) * (2 ** (k - 2)) * factorial(k - 2) * sqrt_pi
    
    return C, indices


# Precompute constants
C_MAT, INDICES = _compute_hermite_moments()
# Indices map: 0->0, 1->4, 2->8, 3->12
# We only need to optimize a0, a4, a8. a12 is determined by g(0)=0.
# g(0) = a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
# H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=4790016
H0_0 = 1.0
H4_0 = 12.0
H8_0 = 1680.0
H12_0 = 4790016.0

# Weights for N = sum a_k^2 * 2^k * k! * sqrt(pi)
# Let W_k = 2^k * k! * sqrt(pi)
W = np.array([
    (2 ** 0) * factorial(0) * np.sqrt(np.pi),
    (2 ** 4) * factorial(4) * np.sqrt(np.pi),
    (2 ** 8) * factorial(8) * np.sqrt(np.pi),
    (2 ** 12) * factorial(12) * np.sqrt(np.pi)
])


def _compute_bound_analytic(a0, a4, a8):
    """
    Compute the uncertainty product A(g)*A(hat{g}) analytically.
    """
    # Calculate a12
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    a = np.array([a0, a4, a8, a12])
    
    # N = ||g||^2 = sum a_k^2 * W_k
    N = np.sum(a**2 * W)
    if N < 1e-15:
        return np.inf
    
    # M_x = sum a_k a_l C_{k,l}
    M_x = a @ C_MAT @ a
    
    # M_xi = sum a_k a_l i^(k-l) C_{k,l}
    # Since C_{k,l} is 0 unless k=l or |k-l|=2.
    # If k=l, i^0 = 1.
    # If |k-l|=2, i^+/-2 = -1.
    # So M_xi = sum a_k^2 C_{k,k} - sum_{|k-l|=2} a_k a_l C_{k,l}
    
    M_xi = 0.0
    for i in range(4):
        for j in range(4):
            if i == j:
                M_xi += a[i] * a[j] * C_MAT[i, j]
            elif abs(INDICES[i] - INDICES[j]) == 2:
                M_xi -= a[i] * a[j] * C_MAT[i, j]
            # else 0
    
    # Variance in x: A(g) = M_x / N (since E[x]=0 by symmetry)
    # Variance in xi: A(hat{g}) = M_xi / N (since E[xi]=0 by symmetry)
    # Product: (M_x / N) * (M_xi / N) = M_x * M_xi / N^2
    
    if M_x < 0 or M_xi < 0:
        # Should not happen for valid functions, but guard against numerical issues
        return np.inf
        
    return (M_x * M_xi) / (N * N)


def _objective_normalized(coeffs):
    """
    Objective function that normalizes the input to the unit sphere
    to handle the scale invariance of the uncertainty product.
    """
    norm = np.linalg.norm(coeffs)
    if norm < 1e-15:
        return np.inf
    normalized = coeffs / norm
    a0, a4, a8 = normalized
    try:
        val = _compute_bound_analytic(a0, a4, a8)
        if np.isnan(val) or np.isinf(val):
            return np.inf
        return val
    except Exception:
        return np.inf


def construct():
    """
    Find optimal coefficients a0, a4, a8 to minimize the uncertainty bound.
    Returns np.array([a0, a4, a8]).
    """
    # Start from the seed solution
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_x = x0.copy()
    best_val = _objective_normalized(best_x)
    
    # Simulated Annealing
    np.random.seed(42)
    T = 1.0
    T_min = 1e-10
    cooling_rate = 0.9995
    steps = 20000
    
    current_x = x0.copy()
    current_val = _objective_normalized(current_x)
    
    # Initial random restarts to find a good basin
    for _ in range(20):
        rx = np.random.randn(3)
        rx_val = _objective_normalized(rx)
        if rx_val < best_val:
            best_val = rx_val
            best_x = rx.copy()
    
    current_x = best_x.copy()
    current_val = best_val
    
    for i in range(steps):
        # Propose a move
        step_size = 0.1 * (1 + i / steps) # Adaptive step size
        delta = np.random.randn(3) * step_size
        proposed_x = current_x + delta
        
        # Normalize proposed point to stay on sphere
        p_norm = np.linalg.norm(proposed_x)
        if p_norm < 1e-15:
            proposed_x = np.random.randn(3)
            p_norm = 1.0
        proposed_x = proposed_x / p_norm
        
        proposed_val = _objective_normalized(proposed_x)
        
        # Acceptance criterion
        if proposed_val < current_val:
            current_x = proposed_x
            current_val = proposed_val
            if current_val < best_val:
                best_val = current_val
                best_x = current_x.copy()
        else:
            # Metropolis criterion
            if np.random.rand() < np.exp(-(proposed_val - current_val) / T):
                current_x = proposed_x
                current_val = proposed_val
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    # Final local refinement using BFGS on the normalized space
    # We need to map back to unconstrained or use a constrained optimizer.
    # BFGS on the sphere is tricky. Instead, we can run BFGS on the original
    # coordinates starting from the best SA point, but the scale invariance
    # might cause issues with gradient convergence if not careful.
    # However, the SA point is already a good local minimum.
    # Let's try BFGS on the original scale-invariant function, but fix the scale
    # by normalizing the output of the objective? No, BFGS needs a differentiable
    # function in R3.
    
    # Alternative: Use the fact that the function is homogeneous.
    # We can just do a few steps of Newton's method on the sphere.
    # Or simply run BFGS on the 3 variables, but the optimizer might drift in scale.
    # Since the value is invariant, the gradient with respect to scale is 0.
    # BFGS might struggle with flat directions.
    
    # Let's use a small Nelder-Mead on the normalized vector, but Nelder-Mead
    # works in R3. If we constrain it to the sphere, it's not standard.
    
    # Better approach: Run BFGS on the original variables, but initialize
    # with the normalized vector. The scale will be determined by the optimizer,
    # but since the value doesn't change, it might just wander.
    # To prevent wandering, we can add a small penalty for deviation from unit norm
    # in the objective function for the optimizer.
    
    def _obj_with_penalty(coeffs):
        val = _compute_bound_analytic(coeffs[0], coeffs[1], coeffs[2])
        penalty = 1e-6 * (np.linalg.norm(coeffs) - 1.0)**2
        if np.isinf(val):
            return np.inf
        return val + penalty
    
    try:
        result = minimize(_obj_with_penalty, best_x, method='BFGS',
                          options={'maxiter': 200, 'gtol': 1e-8})
        if result.fun < best_val + 1e-6: # Allow for penalty difference
             # Verify the actual bound
             final_val = _compute_bound_analytic(result.x[0], result.x[1], result.x[2])
             if final_val < best_val:
                 best_val = final_val
                 best_x = result.x
    except Exception:
        pass
    
    return np.array(best_x)
# EVOLVE-BLOCK-END
