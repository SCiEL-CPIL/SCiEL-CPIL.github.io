"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)"""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(d2)
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_upper = d[mask]
    
    if np.any(d_upper < 1e-15):
        return -1e18
    return math.fsum(np.log(1.0 / d_upper).tolist())

def _log_energy_grad(pts):
    """
    Compute energy and gradient of log(1/d) = -log(d).
    E = sum_{i<j} -log(|x_i - x_j|)
    grad_i E = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :] # shape (n, n, 3)
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, 1.0)
    
    # Weights for gradient: 1/d^2
    weights = 1.0 / d2
    np.fill_diagonal(weights, 0.0)
    
    # grad_i = sum_j weights[i,j] * (x_i - x_j)
    # This is a matrix multiplication: (n, n) @ (n, 3)
    grad = weights @ diff
    
    # Energy calculation (reusing diff and d2 for efficiency, though separate is fine for clarity)
    d = np.sqrt(d2)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_upper = d[mask]
    if np.any(d_upper < 1e-15):
         return -1e18, np.zeros_like(pts)
    energy = math.fsum(np.log(1.0 / d_upper).tolist())
    
    return energy, grad

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=None):
    """Generate random points on unit sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _molecular_dynamics(pts0, deadline, max_steps=50000, initial_temp=0.1, final_temp=1e-4):
    """
    Molecular Dynamics with Velocity Verlet and Adaptive Cooling.
    The force is the negative gradient of the energy.
    We use a thermostat (Langevin-like or just velocity damping) to control temperature.
    Here we use simple velocity damping + random noise injection scaled by temperature.
    """
    pts = pts0.copy()
    v = np.zeros_like(pts)
    
    # Initial energy and gradient
    e, grad = _log_energy_grad(pts)
    best_pts = pts.copy()
    best_e = e
    
    # Force is -grad
    force = -grad
    
    # Adaptive parameters
    # We want to balance kinetic and potential energy roughly, or just cool down.
    # dt needs to be small enough for stability.
    # Force magnitudes for log potential can be large if points are close.
    # Let's estimate typical force magnitude.
    # If min dist is ~0.1, force ~ 1/0.01 = 100.
    # If min dist is ~1, force ~ 1.
    # Let's start with a conservative dt.
    dt = 0.005
    
    # Cooling schedule: exponential decay of temperature
    # temp(t) = initial_temp * (final_temp/initial_temp) ** (t/max_steps)
    
    # We will run in chunks to check time
    steps_per_chunk = 100
    
    current_step = 0
    while current_step < max_steps:
        if time.time() > deadline - 5.0:
            break
            
        # Determine temperature for this chunk
        # Use a linear ramp in log space for smoother cooling
        progress = current_step / max_steps
        temp = initial_temp * (final_temp / initial_temp) ** progress
        
        # Velocity Verlet Step 1: Update position by half step with current velocity
        # x(t+dt/2) = x(t) + v(t) * dt/2
        v_half = v + 0.5 * dt * force
        
        # x_new = x + v_half * dt
        x_new = pts + dt * v_half
        
        # Project to sphere
        x_new = _project_to_sphere(x_new)
        
        # Calculate new force at x_new
        e_new, grad_new = _log_energy_grad(x_new)
        force_new = -grad_new
        
        # Velocity Verlet Step 2: Update velocity to full step
        # v(t+dt) = v_half + 0.5 * dt * force_new
        v_new = v_half + 0.5 * dt * force_new
        
        # Damping and Thermalization
        # Apply friction to remove excess energy (Langevin friction)
        # v = v * (1 - gamma) + noise
        # Gamma depends on temperature. Higher T -> less friction (more exploration).
        # Lower T -> more friction (settling).
        # Let's use a fixed friction coefficient that increases as T decreases?
        # Or just a constant friction.
        # For MD on sphere, we need to be careful.
        # Let's use a simple damping: v *= 0.99
        # And add noise: v += sqrt(2 * temp * dt) * random
        # This is a simplified Langevin thermostat.
        
        damping = 0.98
        noise_scale = np.sqrt(2.0 * temp * dt)
        
        v_new = v_new * damping + noise_scale * np.random.randn(*v_new.shape)
        
        # Update state
        pts = x_new
        v = v_new
        force = force_new
        e = e_new
        
        # Track best
        if e < best_e:
            best_e = e
            best_pts = pts.copy()
            
        current_step += 1
        
        # Adaptive dt: if energy jumps or points collide, reduce dt
        # Check for collisions
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d2, 1.0)
        min_d2 = np.min(d2)
        if min_d2 < 1e-6: # Too close
            dt *= 0.5
            # Push points apart slightly? Or just let MD handle it?
            # Let's just reduce dt significantly.
        elif min_d2 > 0.1 and dt < 0.005:
            # If stable and far apart, can increase dt slightly
            dt *= 1.1
            
    return best_pts, best_e

def _gradient_descent_polish(pts0, deadline, max_iter=2000):
    """
    Final polish with gradient descent to ensure we are at a local minimum.
    """
    pts = pts0.copy()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    lr = 0.005
    v = np.zeros_like(pts)
    momentum = 0.9
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        # Calculate gradient
        n = len(pts)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d2, 1.0)
        
        weights = 1.0 / d2
        np.fill_diagonal(weights, 0.0)
        
        grad = -np.sum(weights[:, :, np.newaxis] * diff, axis=1)
        
        # Project gradient to tangent space
        radial = np.sum(grad * pts, axis=1, keepdims=True)
        grad_tangent = grad - radial * pts
        
        # Momentum update
        v = momentum * v - lr * grad_tangent
        new_pts = _project_to_sphere(pts + v)
        
        new_energy = _log_energy(new_pts)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            # If energy increases, revert and reduce step
            v = 0.5 * v
            lr *= 0.9
            if lr < 1e-8:
                break
                
    return best_pts, best_energy

def construct():
    """Construct 100 points on unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    # Safety margin: 150s limit, stop at 145s to be safe
    deadline = start_time + 145.0
    
    best_pts = None
    best_energy = np.inf
    
    # 1. Initial diverse starts
    # Fibonacci is a good structured start
    # Random starts provide diversity
    starts = [
        ("fib", _fibonacci_sphere(N)),
        ("rand_1", _random_sphere(N, 42)),
        ("rand_2", _random_sphere(N, 123)),
        ("rand_3", _random_sphere(N, 999)),
    ]
    
    # 2. Run Molecular Dynamics on each start
    # MD is good at exploring the landscape with inertia.
    
    for i, (name, pts0) in enumerate(starts):
        if time.time() > deadline - 20:
            break
            
        # Run MD
        # Allocate time based on number of starts
        # 4 starts, ~30s each for MD, then polish
        pts_md, e_md = _molecular_dynamics(pts0, deadline - 10, max_steps=20000, initial_temp=0.05, final_temp=1e-5)
        
        if e_md < best_energy:
            best_energy = e_md
            best_pts = pts_md
            
        # 3. Polish with Gradient Descent
        if time.time() < deadline - 5:
            pts_gd, e_gd = _gradient_descent_polish(pts_md, deadline - 2)
            if e_gd < best_energy:
                best_energy = e_gd
                best_pts = pts_gd

    # 4. Final global polish on the best found so far
    if best_pts is not None and time.time() < deadline - 2:
        pts_final, e_final = _gradient_descent_polish(best_pts, deadline, max_iter=5000)
        if e_final < best_energy:
            best_energy = e_final
            best_pts = pts_final
            
    return best_pts
# EVOLVE-BLOCK-END
