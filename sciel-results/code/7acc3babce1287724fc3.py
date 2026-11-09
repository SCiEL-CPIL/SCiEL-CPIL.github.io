"""Heat2D 10x10 ODE solver with structure-aware optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] starting at y0.
    
    For the 10x10 2D heat equation, detects the 5-point stencil structure
    and uses an explicit method with large steps (since the problem is well-conditioned
    over [0, 0.05]). Falls back to generic RK4 if structure is not detected.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard: leave 10s margin before 330s wall-clock limit
    start_time = time.time()
    DEADLINE = start_time + 320.0
    
    # Try to detect the 2D heat equation 5-point stencil structure
    # For a 10x10 grid, n = 100. The stencil connects each cell to 4 neighbors.
    # f(t, y) for heat equation: dy_i/dt = alpha * (y_left + y_right + y_up + y_down - 4*y_i) / dx^2
    # with boundary conditions (y=0 at boundaries, so boundary cells only have interior neighbors).
    
    # Detect structure by probing with a few test vectors
    def _detect_stencil(f, t, y):
        """
        Try to detect if f is a 2D heat equation with 5-point stencil.
        Returns (alpha_over_dx2, grid_shape) or None.
        """
        n = len(y)
        if n != 100:
            return None
        
        # Use a simple test: set one interior cell to 1, rest to 0
        # For a 10x10 grid, interior cells are those with index not on boundary
        # Cell (i,j) has flat index i*10 + j (row-major)
        # Interior: 1 <= i <= 8, 1 <= j <= 8
        
        test_idx = 5 * 10 + 5  # center cell (5,5), flat index 55
        y_test = np.zeros(n)
        y_test[test_idx] = 1.0
        
        ft = f(t, y_test)
        ft = np.asarray(ft, dtype=float)
        
        if not np.all(np.isfinite(ft)):
            return None
        
        # For heat equation with zero boundary, the derivative at center cell should be
        # -4 * alpha/dx^2, and neighbors should be +1 * alpha/dx^2
        # Check which cells have non-zero derivatives
        nonzero = np.nonzero(np.abs(ft) > 1e-15)[0]
        
        if len(nonzero) != 5:
            return None
        
        # Check if the pattern matches a 5-point stencil centered at test_idx
        # Neighbors of (5,5) are (4,5), (6,5), (5,4), (5,6)
        # Flat indices: 4*10+5=45, 6*10+5=65, 5*10+4=54, 5*10+6=56
        expected_neighbors = [45, 54, 56, 65]
        
        nonzero_set = set(nonzero.tolist())
        if test_idx not in nonzero_set:
            return None
        
        neighbor_set = nonzero_set - {test_idx}
        if neighbor_set != set(expected_neighbors):
            return None
        
        # Extract the coefficients
        center_val = ft[test_idx]
        neighbor_vals = [ft[idx] for idx in expected_neighbors]
        
        # For symmetric stencil: center = -4*c, neighbors = +c
        c_center = -center_val / 4.0
        c_neigh = np.mean(neighbor_vals)
        
        if abs(c_center - c_neigh) > 1e-10:
            return None
        
        c = c_neigh
        if abs(c) < 1e-20:
            return None
        
        return c, (10, 10)
    
    # Detect structure
    detection = _detect_stencil(f, t0, y0)
    
    if detection is not None:
        c, (rows, cols) = detection
        # c = alpha / dx^2
        # The system is: dy/dt = c * L * y where L is the graph Laplacian of the grid
        # We can use an explicit method with step size limited by stability: h < 1/(4*c*max_degree)
        # For 5-point stencil, max eigenvalue of L is about 8 (for interior), so h < 1/(8*c)
        # But we can also use a smarter approach: use scipy to solve or use eigenvalue-based stepping
        
        # Since the problem is fixed and well-conditioned over [0, 0.05],
        # let's use an explicit method with adaptive stepping based on the stencil.
        
        # Actually, for the 10x10 heat equation on [0, 0.05], we can use a moderate number
        # of RK4 steps. The key insight from the knowledge: reducing calls from 100 to ~3
        # by exploiting structure. Let's use a Crank-Nicolson or a large-step explicit method.
        
        # For the 10x10 grid, the Laplacian eigenvalues are known analytically:
        # lambda_{p,q} = 4*c * (sin^2(p*pi/(2*(rows-1))) + sin^2(q*pi/(2*(cols-1)))) for p,q = 1..rows-1
        # The largest eigenvalue determines the stability limit for explicit methods.
        
        # Let's compute the max eigenvalue of the graph Laplacian for the 10x10 grid
        # with zero Dirichlet boundary conditions.
        
        # Actually, a simpler approach: use a moderate number of steps with RK4.
        # The score is -log10(rel_error) - 0.5*log10(calls). So we want high accuracy with few calls.
        
        # Let's use a variable-step approach or a fixed large step.
        # For the heat equation, the solution decays smoothly, so a moderate number of steps
        # with RK4 should give high accuracy.
        
        # Use RK4 with a reasonable number of steps. The time interval is [0, 0.05].
        # Let's try 200 steps first (400 f calls), and if we have time, do more.
        
        y = np.array(y0, dtype=float)
        t = t0
        total_steps = 200
        h = (t1 - t0) / total_steps
        
        for step in range(total_steps):
            if time.time() > DEADLINE - 5.0:
                break
            
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
            
            if not np.all(np.isfinite(y)):
                break
        
        return y
    else:
        # Fallback: generic RK4 with moderate steps
        y = np.array(y0, dtype=float)
        t = t0
        total_steps = 500
        h = (t1 - t0) / total_steps
        
        for step in range(total_steps):
            if time.time() > DEADLINE - 5.0:
                break
            
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        
        return y

# EVOLVE-BLOCK-END
