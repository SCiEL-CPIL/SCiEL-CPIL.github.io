"""
Strategy: Continuous Piecewise-Linear Optimization
Instead of fixed-width steps (discrete combinatorial), we model h(x) as a continuous
piecewise-linear function (linear interpolation between N points).
We use a spectral approach: 
1. The overlap of h and shifted 1-h is related to the autocorrelation of h.
2. We want to minimize the maximum of (h * (1-h_shifted)).
3. Actually, the task says "minimize the maximum overlap of h with shifted 1-h".
   Overlap integral for shift t: I(t) = ∫ h(x) (1 - h(x-t)) dx = ∫ h(x) dx - ∫ h(x)h(x-t) dx = 1 - A(t).
   So minimizing max_t I(t) is equivalent to maximizing min_t A(t).
   Wait, the task says "minimize the maximum overlap".
   If I(t) is the overlap, we want min max_t I(t).
   Since I(t) = 1 - A(t), this is equivalent to max min_t A(t).
   
   Let's verify the seed program's score.
   Seed: h(x) = 0.5 everywhere.
   A(t) = ∫ 0.5 * 0.5 dx = 0.25.
   I(t) = 1 - 0.25 = 0.75.
   Score = 1.0 / 0.75? No, score is seed/value. Seed value is 0.75.
   If we get value < 0.75, score > 1.0.
   
   Let's try a sinusoidal wave.
   h(x) = 0.5 + A * sin(2πkx).
   Constraint: 0 <= h <= 1 => A <= 0.5.
   A(t) = ∫ (0.5 + A sin(2πkx)) (0.5 + A sin(2πk(x-t))) dx
   = 0.25 + 0.5A ∫sin(2πk(x-t))dx + 0.5A ∫sin(2πkx)dx + A^2 ∫sin(2πkx)sin(2πk(x-t))dx
   Integrals of sin over full period are 0.
   Last term: A^2 * (1/2) cos(2πkt).
   So A(t) = 0.25 + 0.5 A^2 cos(2πkt).
   We want to maximize min_t A(t).
   min_t cos(2πkt) = -1.
   min_t A(t) = 0.25 - 0.5 A^2.
   This is maximized when A=0, giving 0.25.
   So sinusoid with k>=1 doesn't help?
   
   Let's check k=0. h is constant. A(t)=0.25.
   
   What if we use a different shape?
   The problem is related to the Erdős overlap problem.
   Known result: For h in [0,1], integral 1, the minimal maximum overlap is 1/2?
   Or is it related to the "overlap" definition?
   
   Let's re-read carefully: "minimize the maximum overlap of h with shifted 1-h".
   Overlap(t) = ∫ h(x) (1-h(x-t)) dx.
   We want to minimize max_t Overlap(t).
   
   If h is a "step" function, e.g., h=1 on [0,0.5], 0 on [0.5,1].
   Integral = 0.5. But task says integral 1.
   So h must have average 0.5.
   
   Let's try h(x) = 1 for x in [0, 0.25] U [0.5, 0.75] and 0 otherwise?
   No, let's look at the "Erdos minimum overlap" literature.
   It often refers to minimizing the L-infinity norm of the convolution or cross-correlation.
   
   If we use a triangular wave (piecewise linear):
   h(x) = 1 - |2x - 1| ? No, that's 0 at 0, 1 at 0.5, 0 at 1. Integral 0.5.
   Scale it: h(x) = 2(1 - |2x - 1|). Integral 1. Max 2 (violates h<=1).
   
   We need h in [0,1].
   Let's try h(x) = 0.5 + 0.5 * sign(sin(2πx)). Square wave.
   This is the seed's "step" idea but oscillating.
   
   Actually, there is a known construction using a "tent" map or similar.
   Let's try to optimize the coefficients of a Fourier series directly, but constrained to [0,1].
   Or simpler: Use a set of "bumps".
   
   Let's try a specific known optimal shape for this type of problem:
   The function that minimizes max overlap is often related to the Rademacher functions or Walsh functions, but those are +/-1.
   
   Let's try: h(x) = 1 if x in [0, 1/3] U [2/3, 1], 0 otherwise?
   Integral = 2/3. Need 1.
   So h(x) = 1.5 on those intervals? No, h<=1.
   
   Let's try h(x) = 1 on [0, 0.5] and 0 on [0.5, 1].
   Integral 0.5. Scale by 2? No, h<=1.
   So we need to "fill" the rest.
   
   Let's try a linear ramp.
   h(x) = 2x for x in [0, 0.5], 2(1-x) for x in [0.5, 1].
   Integral = 0.5.
   We need integral 1.
   So h(x) = 0.5 + ramp?
   
   Let's use a numerical optimizer on the vertices of a piecewise linear function.
   We define N vertices. h is linear between them.
   We want to minimize max_t ∫ h(x)(1-h(x-t)) dx.
   
   Since the time limit is 150s, we can do a simple gradient descent or Nelder-Mead on the vertices.
   N=100 vertices might be too many for Nelder-Mead.
   Let's use a smaller N, say N=20, and refine.
   
   Alternatively, use the fact that the optimal solution for such "overlap" problems often has a specific symmetry.
   
   Let's try a different structural change: **Genetic Algorithm on the bitstring of a high-resolution grid**.
   The seed uses fixed width steps. A genetic algorithm can evolve the "shape" more flexibly.
   But the directive says "different design or modelling family".
   
   Let's go with **Continuous Piecewise Linear + Gradient Descent**.
   We parameterize h by its values at N points.
   We enforce 0<=h<=1 and sum(h)*dx = 1 (approx).
   We compute the overlap for a dense set of shifts.
   We use scipy.optimize.minimize (L-BFGS-B or similar) to minimize the max overlap.
   
   To handle the "max" in the objective, we can use a smooth approximation like log-sum-exp.
   
   Let's implement this.
"""

import numpy as np
import time
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _compute_overlap_values(h, n_shifts=1000):
    """
    Compute the overlap integral I(t) = ∫ h(x)(1-h(x-t)) dx for a set of shifts t.
    h is a piecewise linear function defined by its values at N points.
    We approximate the integral using the trapezoidal rule on a fine grid.
    """
    N = len(h)
    # Create a fine grid for integration
    M = 1000 # Fine grid resolution
    x = np.linspace(0, 1, M, endpoint=False)
    # Interpolate h to the fine grid
    # h is defined at points x_i = i/N.
    # We need to be careful with periodic boundary conditions.
    # The problem is on [0, 2] but periodic? "shifted 1-h".
    # Usually these problems are on a circle [0,1) or [0,2) with periodicity.
    # The seed program uses N=100 steps on [0,2]? 
    # "Choose equal-width step heights h on [0,2]".
    # So the domain is [0,2]. Period 2?
    # If domain is [0,2], and we shift, do we wrap around?
    # "shifted 1-h".
    # Let's assume periodic on [0,2].
    
    # Let's stick to the seed's implied domain.
    # Seed: N=100, h = 0.5.
    # If domain is [0,2], integral is ∫_0^2 0.5 dx = 1. This matches "integral 1".
    # So domain length is 2.
    
    L = 2.0
    x_grid = np.linspace(0, L, M, endpoint=False)
    
    # Interpolate h (defined on [0, L] with N points) to x_grid
    # h points are at x_i = i * L / N
    x_h = np.linspace(0, L, N, endpoint=False)
    # For interpolation, we need to handle the wrap-around.
    # np.interp doesn't handle periodicity well.
    # We can extend h to be periodic.
    h_ext = np.concatenate([h, h[:1]]) # h[0] at the end
    x_ext = np.linspace(0, L + L/N, N + 1, endpoint=True)
    
    h_fine = np.interp(x_grid, x_ext, h_ext, period=L)
    
    # Compute shifts
    # t from 0 to L
    t_vals = np.linspace(0, L, n_shifts, endpoint=False)
    
    overlaps = np.zeros(n_shifts)
    for i, t in enumerate(t_vals):
        # Shift h_fine by t
        # h(x-t)
        x_shifted = (x_grid - t) % L
        h_shifted = np.interp(x_shifted, x_grid, h_fine, period=L)
        
        # Overlap = ∫ h(x) * (1 - h_shifted(x)) dx
        # Using trapezoidal rule (or just sum * dx since uniform)
        integrand = h_fine * (1 - h_shifted)
        overlaps[i] = np.sum(integrand) * (L / M)
        
    return overlaps

def _objective_and_grad(params):
    """
    Objective: max overlap.
    params: flattened array of h values.
    """
    N = int(np.sqrt(len(params))) # Not really, let's just use the length
    # We'll use a fixed N for the piecewise linear representation
    N = 20 # Number of vertices
    h = params
    
    # Enforce bounds 0,1
    h_clipped = np.clip(h, 0, 1)
    
    # Enforce integral constraint?
    # The optimizer might drift away from integral 1.
    # We can add a penalty.
    L = 2.0
    dx = L / N
    integral = np.sum(h_clipped) * dx
    penalty = (integral - 1.0) ** 2
    
    overlaps = _compute_overlap_values(h_clipped, n_shifts=200)
    max_overlap = np.max(overlaps)
    
    return max_overlap + 0.1 * penalty

def construct():
    start_time = time.time()
    deadline = start_time + 120.0 # Safety margin
    
    N = 20 # Number of piecewise linear segments
    L = 2.0
    
    # Initial guess: constant 0.5
    h0 = np.full(N, 0.5)
    
    best_h = h0.copy()
    best_score = _objective_and_grad(h0)
    
    # Try a few random restarts
    np.random.seed(42)
    
    for restart in range(10):
        if time.time() > deadline:
            break
            
        # Random initialization within bounds
        h_init = np.random.uniform(0, 1, N)
        # Normalize to have integral 1
        integral = np.sum(h_init) * (L / N)
        if integral > 0:
            h_init = h_init * (1.0 / integral)
        h_init = np.clip(h_init, 0, 1)
        
        # Optimize
        try:
            res = minimize(_objective_and_grad, h_init, method='L-BFGS-B', 
                           bounds=[(0, 1)] * N, 
                           options={'maxiter': 50, 'ftol': 1e-6})
            
            if res.fun < best_score:
                best_score = res.fun
                best_h = res.x
        except Exception:
            pass
            
    # Refine with a higher resolution if time permits
    if time.time() < deadline - 10:
        # Use the best_h as initial guess for a finer grid?
        # Interpolating N=20 to N=50
        N_fine = 50
        x_coarse = np.linspace(0, L, N, endpoint=False)
        x_fine = np.linspace(0, L, N_fine, endpoint=False)
        h_fine_init = np.interp(x_fine, x_coarse, best_h, period=L)
        
        def _objective_fine(params):
            h = params
            h_clipped = np.clip(h, 0, 1)
            integral = np.sum(h_clipped) * (L / N_fine)
            penalty = (integral - 1.0) ** 2
            overlaps = _compute_overlap_values(h_clipped, n_shifts=500)
            max_overlap = np.max(overlaps)
            return max_overlap + 0.1 * penalty
            
        try:
            res_fine = minimize(_objective_fine, h_fine_init, method='L-BFGS-B', 
                                bounds=[(0, 1)] * N_fine, 
                                options={'maxiter': 100, 'ftol': 1e-7})
            if res_fine.fun < best_score:
                best_score = res_fine.fun
                best_h = res_fine.x
        except Exception:
            pass

    # Return the result in the format expected by the evaluator
    # The evaluator expects a "sequence of step heights".
    # The seed returned np.full(N, 0.5) where N=100.
    # We should probably return a sequence of length 100 to be safe, 
    # or whatever the evaluator expects.
    # The prompt says: "construct() -> sequence of step heights in [0, 1] whose sum is exactly len/2"
    # Wait, "sum is exactly len/2"?
    # If len=100, sum=50.
    # If h is step heights on [0,2], and there are 100 steps, each width is 2/100 = 0.02.
    # Integral = sum(h_i * 0.02) = 0.02 * sum(h_i).
    # We need Integral = 1.
    # So 0.02 * sum(h_i) = 1 => sum(h_i) = 50.
    # This matches "sum is exactly len/2" (100/2 = 50).
    
    # So we need to map our piecewise linear function to 100 step heights.
    # We will sample our optimized function at the centers of the 100 steps.
    
    N_out = 100
    L = 2.0
    step_width = L / N_out
    
    # Sample best_h at the centers of the steps
    # best_h is defined on a grid of size len(best_h).
    # Let's interpolate best_h to N_out points.
    
    # Determine the grid of best_h
    N_opt = len(best_h)
    x_opt = np.linspace(0, L, N_opt, endpoint=False)
    
    # Centers of the output steps
    x_centers = (np.arange(N_out) + 0.5) * step_width
    
    # Interpolate
    # Handle periodicity
    h_ext = np.concatenate([best_h, best_h[:1]])
    x_ext = np.linspace(0, L + L/N_opt, N_opt + 1, endpoint=True)
    
    h_out = np.interp(x_centers, x_ext, h_ext, period=L)
    
    # Ensure sum is exactly N_out / 2
    # The interpolation might drift the sum.
    # We need to adjust h_out so that sum(h_out) == N_out / 2.
    # And 0 <= h_out <= 1.
    
    current_sum = np.sum(h_out)
    target_sum = N_out / 2.0
    
    # Simple scaling
    if current_sum > 0:
        h_out = h_out * (target_sum / current_sum)
    
    # Clip to [0,1]
    h_out = np.clip(h_out, 0, 1)
    
    # After clipping, the sum might be off again.
    # We need a projection to the box [0,1]^N with fixed sum.
    # This is a standard "water filling" or "clipping and redistribution" problem.
    
    # Iterative clipping and scaling
    for _ in range(10):
        h_out = np.clip(h_out, 0, 1)
        s = np.sum(h_out)
        if s == 0:
            h_out = np.full(N_out, target_sum / N_out)
            break
        h_out = h_out * (target_sum / s)
        
    # Final check
    # The evaluator might check sum exactly.
    # Due to floating point, we might be off by 1e-15.
    # Let's adjust the last element to compensate for any tiny error.
    diff = target_sum - np.sum(h_out)
    if abs(diff) > 1e-9:
        # Find an element that can take the difference
        # This is a hack, but for small diffs it works.
        # Better: distribute the difference evenly or to the least constrained elements.
        # For simplicity, add to the last element if it stays in [0,1].
        if 0 <= h_out[-1] + diff <= 1:
            h_out[-1] += diff
        else:
            # Redistribute
            h_out = h_out + diff / N_out
            h_out = np.clip(h_out, 0, 1)
            # Re-normalize again
            s = np.sum(h_out)
            h_out = h_out * (target_sum / s)
            
    return h_out

# EVOLVE-BLOCK-END
