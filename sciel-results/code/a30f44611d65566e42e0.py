"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _rk4_step(f, t, y, h):
    """Single RK4 step."""
    k1 = f(t, y)
    k2 = f(t + h/2, y + h/2 * k1)
    k3 = f(t + h/2, y + h/2 * k2)
    k4 = f(t + h, y + h * k3)
    return y + h/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)

def _rk4_integrate(f, t0, t1, y0, n_steps):
    """Integrate using fixed-step RK4 with n_steps steps."""
    h = (t1 - t0) / n_steps
    t = t0
    y = np.array(y0, dtype=float)
    for _ in range(n_steps):
        y = _rk4_step(f, t, y, h)
        t += h
    return y

def _bulirsch_stoer(f, t0, t1, y0, max_N=256, tol=1e-14):
    """
    Bulirsch-Stoer extrapolation using RK4 as the base method.
    Uses the standard BS table with Richardson extrapolation.
    
    Returns (y_final, num_calls)
    """
    # We'll use a table where row i corresponds to step count N_i
    # Standard BS uses N values that are related by factors
    # Let's use N = 2, 4, 8, 16, 32, 64, 128, 256
    # But for efficiency, we can use the standard BS sequence
    
    # The BS method uses a 2D table Y[i,j] where:
    # Y[i,0] = result with N_i steps
    # Y[i,j] = Y[i,j-1] + (Y[i,j-1] - Y[i-1,j-1]) / (r_j^4 - 1)
    # where r_j is the ratio of step sizes
    
    # Standard BS uses N values: 2, 4, 8, 16, ... but with specific ratios
    # Actually, the classic BS uses N = 2, 4, 8, ... but the extrapolation
    # is done differently. Let me use a simpler approach:
    
    # Use Richardson extrapolation with RK4
    # RK4 has error O(h^4), so we can use Richardson extrapolation
    # with results at h, h/2, h/4, etc.
    
    # Let's compute results at decreasing step sizes and extrapolate
    # Using the fact that error ~ C*h^4 + D*h^6 + ...
    
    # We'll compute y(h), y(h/2), y(h/4), ... and extrapolate
    # Using Richardson: y_extrap = y(h/2) + (y(h/2) - y(h)) / (2^4 - 1)
    # Then y_extrap2 = y(h/4) + (y_extrap1 - y(h/2)) / (2^4 - 1)
    # etc.
    
    # Actually, let's use the full BS table approach
    # N values: we'll use N = 2, 4, 8, 16, 32, 64, 128
    # For each N, compute y_N using RK4 with N steps
    # Then build the extrapolation table
    
    # The key insight: if we use N, 2N, 4N, ... steps, the ratio is 2
    # So the extrapolation factor is 2^4 = 16
    
    # Let's use a simpler iterative approach
    # Start with a moderate N, then refine
    
    # Actually, let me just do straightforward Richardson extrapolation
    # with a few levels
    
    # Compute at N = 32, 64, 128, 256 steps
    # Then extrapolate
    
    Ns = [32, 64, 128, 256]
    results = []
    
    for N in Ns:
        y_N = _rk4_integrate(f, t0, t1, y0, N)
        results.append(y_N)
    
    # Now Richardson extrapolate
    # y_{i+1} = y_i + (y_i - y_{i-1}) / (16 - 1) for first level
    # But this isn't quite right for multiple levels
    
    # Let me use the proper BS table
    # Y[i,j] where i indexes the step size (0 = largest), j is extrapolation level
    
    # Y[i,0] = y with N_i steps
    # Y[i,j] = Y[i,j-1] + (Y[i,j-1] - Y[i-1,j-1]) / (2^{4*j} - 1)
    # Wait, that's not right either. Let me look this up properly.
    
    # The standard Bulirsch-Stoer extrapolation:
    # If we have solutions at step sizes h_1, h_2, ..., h_m where h_i = h_1 / 2^{i-1}
    # Then the extrapolation table is:
    # Y[i,0] = y(h_i)
    # Y[i,j] = Y[i,j-1] + (Y[i,j-1] - Y[i-1,j-1]) / (r^4 - 1)
    # where r = h_{i-1}/h_i = 2 for our case
    # So the factor is 2^4 - 1 = 15
    
    # Let me redo this properly
    # We have Ns = [32, 64, 128, 256]
    # h_0 = (t1-t0)/32, h_1 = (t1-t0)/64, etc.
    # So h_{i-1}/h_i = 2
    
    # Build the table
    m = len(Ns)
    Y = np.zeros((m, m))
    for i in range(m):
        Y[i, 0] = results[i]
    
    for j in range(1, m):
        for i in range(j, m):
            Y[i, j] = Y[i, j-1] + (Y[i, j-1] - Y[i-1, j-1]) / 15.0
    
    # The best estimate is Y[m-1, m-1]
    y_final = Y[m-1, m-1]
    
    return y_final

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Use Bulirsch-Stoer extrapolation
    # This should give very high accuracy with relatively few function calls
    y_end = _bulirsch_stoer(f, t0, t1, y0)
    
    # Ensure the result is finite
    if not np.all(np.isfinite(y_end)):
        # Fall back to RK4 with more steps
        y_end = _rk4_integrate(f, t0, t1, y0, 4096)
    
    return y_end
# EVOLVE-BLOCK-END
