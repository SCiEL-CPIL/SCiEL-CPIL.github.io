import numpy as np
import time
from scipy.optimize import minimize

N = 32

def _initial_positions(n=N, method='grid'):
    """
    Generate initial positions for n circles in a dense lattice.
    Returns array of shape (n, 2).
    """
    if method == 'grid':
        # Use a square grid layout
        side = int(np.ceil(np.sqrt(n)))
        # If n is not a perfect square, we'll have some empty spots or uneven rows
        # Better to use a layout that fits n exactly or close to it
        # For n=32, 6x6 grid has 36 spots. We use 32 of them.
        # Or 5x7 = 35.
        # Let's just use a simple grid and take the first n positions.
        # To make it dense, we start with a small spacing.
        
        # Let's try to fit them in a rectangle.
        # 32 = 4 * 8? No, too elongated.
        # 32 = 5 * 6 + 2?
        # Let's just use a 6x6 grid and pick 32 points centered.
        xs = np.linspace(0.5/6, 1-0.5/6, 6)
        ys = np.linspace(0.5/6, 1-0.5/6, 6)
        X, Y = np.meshgrid(xs, ys)
        pts = np.column_stack([X.ravel(), Y.ravel()])
        # Center the grid in the unit square
        # The grid is already 0..1 roughly.
        # We need to ensure we pick n points.
        if len(pts) >= n:
            # Pick the n points closest to the center? Or just the first n?
            # First n from meshgrid is row-major, which is fine.
            return pts[:n]
        else:
            # Fallback to random
            return np.random.rand(n, 2)
            
    elif method == 'random':
        return np.random.rand(n, 2)
    
    else:
        # Halton sequence for better coverage
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        points = []
        for i in range(1, n+1):
            x = halton(i, 2)
            y = halton(i, 3)
            points.append([x, y])
        return np.array(points)

def _simulate_expansion(positions, initial_radii, max_time=60.0, dt=0.005, pressure=1.0, damping=0.9, boundary_penalty=100.0, overlap_penalty=1000.0):
    """
    Simulate the expansion of circles under pressure.
    positions: (n, 2) array
    initial_radii: (n,) array
    Returns final positions and radii.
    """
    pos = positions.copy()
    radii = initial_radii.copy()
    velocities = np.zeros_like(pos)
    
    start_time = time.time()
    
    while time.time() - start_time < max_time:
        # Calculate forces
        forces = np.zeros_like(pos)
        
        # 1. Internal pressure (expansion)
        # Force outward from center? Or just increase radius?
        # In this model, "pressure" drives radius growth.
        # We model this as a constant force pushing centers apart?
        # Actually, the prompt says "expand their radii under a constant internal pressure".
        # This is slightly non-standard for rigid body sims of circles where radius is fixed.
        # Here, radius is a variable.
        # Let's interpret: We want to maximize sum(r).
        # So we add a "force" that increases radius.
        # But radius isn't a position.
        # Let's stick to the standard interpretation: 
        # Circles have fixed radius? No, "expand their radii".
        # So radius is dynamic.
        # Let's treat radius growth as a separate dynamic variable.
        # dr/dt = pressure - overlap_force - boundary_force?
        
        # Alternative interpretation: 
        # The circles are soft spheres. 
        # We increase a global pressure parameter P.
        # The equilibrium radius r_i is determined by the balance of forces.
        # For simplicity in a discrete time step:
        # 1. Move positions to resolve overlaps (penalty forces).
        # 2. Increase radii slightly if no overlap.
        
        # Let's use a simpler iterative relaxation:
        # 1. Check overlaps. If overlap, push apart and/or reduce radius growth.
        # 2. Check boundaries. If outside, push in.
        # 3. Increase all radii by a small amount `dr` if the system is "stable" (low potential energy).
        
        # Let's try a different approach: 
        # Optimize positions and radii simultaneously using gradient descent on a potential energy.
        # E = -sum(r_i) + lambda_overlap * sum(overlap_i) + lambda_boundary * sum(boundary_violation_i)
        
        # This is effectively what L-BFGS-B does, but we want a "physics" feel.
        # Let's do explicit Euler integration on a potential field.
        
        # Potential Energy:
        # U = -sum(r) + K_rep * sum_{i<j} max(0, r_i+r_j-dist_ij)^2 + K_bound * sum_i (max(0, r_i-x_i)^2 + max(0, r_i-(1-x_i))^2 + ...)
        
        # Gradient w.r.t positions and radii.
        
        # This is getting complex. Let's stick to the "pressurized granular" idea more simply:
        # 1. Start with small radii.
        # 2. Increase all radii by `dr`.
        # 3. Relax positions to minimize overlap (like a molecular dynamics step).
        # 4. If relaxation fails (too much overlap), rollback radius increase and stop or reduce dr.
        
        # Let's implement Step 3: Relax positions using penalty forces.
        
        # Compute forces on positions due to overlaps and boundaries
        for i in range(N):
            # Boundary forces
            if pos[i, 0] < radii[i]:
                forces[i, 0] += boundary_penalty * (radii[i] - pos[i, 0])
            if pos[i, 0] > 1 - radii[i]:
                forces[i, 0] -= boundary_penalty * (pos[i, 0] - (1 - radii[i]))
            if pos[i, 1] < radii[i]:
                forces[i, 1] += boundary_penalty * (radii[i] - pos[i, 1])
            if pos[i, 1] > 1 - radii[i]:
                forces[i, 1] -= boundary_penalty * (pos[i, 1] - (1 - radii[i]))
                
            # Overlap forces
            for j in range(i+1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist and dist > 1e-8:
                    overlap = min_dist - dist
                    # Force is proportional to overlap, direction along normal
                    fx = (dx / dist) * overlap_penalty * overlap
                    fy = (dy / dist) * overlap_penalty * overlap
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
                elif dist <= 1e-8:
                    # Random push
                    forces[i, 0] += np.random.rand() - 0.5
                    forces[i, 1] += np.random.rand() - 0.5
                    forces[j, 0] -= np.random.rand() - 0.5
                    forces[j, 1] -= np.random.rand() - 0.5
        
        # Update velocities and positions
        velocities = velocities * damping + forces * dt
        pos = pos + velocities * dt
        
        # Clip positions to [0,1] to prevent wild excursions, though penalty should handle it
        # Actually, clipping can cause energy issues. Let's rely on penalty.
        
        # Check for validity and decide on radius growth
        # Calculate current "badness"
        badness = 0.0
        for i in range(N):
            if pos[i, 0] < radii[i] or pos[i, 0] > 1 - radii[i]:
                badness += 100
            if pos[i, 1] < radii[i] or pos[i, 1] > 1 - radii[i]:
                badness += 100
            for j in range(i+1, N):
                dist = np.sqrt((pos[i,0]-pos[j,0])**2 + (pos[i,1]-pos[j,1])**2)
                if dist < radii[i] + radii[j]:
                    badness += 100 * (radii[i] + radii[j] - dist)
        
        if badness < 1e-4:
            # System is stable, grow radii
            radii += 0.001
        else:
            # System is unstable, maybe shrink radii slightly or just wait
            # To allow escape from local minima, we can occasionally shrink
            if np.random.rand() < 0.1:
                radii *= 0.99
            # And keep positions as they are (relaxation continues)
            
        # Safety check: if any radius becomes too small or negative
        if np.any(radii < 1e-5):
            radii = np.maximum(radii, 1e-5)
            
    return pos, radii

def _polish(pos, radii, time_limit=10.0):
    """
    Use L-BFGS-B to polish the result.
    Variables: x_1..x_n, y_1..y_n, r_1..r_n
    Objective: -sum(r)
    Constraints:
    1. x_i >= r_i
    2. x_i <= 1 - r_i
    3. y_i >= r_i
    4. y_i <= 1 - r_i
    5. dist(i,j) >= r_i + r_j
    """
    n = N
    x0 = np.concatenate([pos.flatten(), radii])
    
    def objective(vars):
        p = vars[:2*n].reshape(n, 2)
        r = vars[2*n:]
        # Boundary violations
        pen = 0.0
        for i in range(n):
            if p[i, 0] < r[i]: pen += (r[i] - p[i, 0])**2
            if p[i, 0] > 1 - r[i]: pen += (p[i, 0] - (1 - r[i]))**2
            if p[i, 1] < r[i]: pen += (r[i] - p[i, 1])**2
            if p[i, 1] > 1 - r[i]: pen += (p[i, 1] - (1 - r[i]))**2
        # Overlap violations
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt((p[i,0]-p[j,0])**2 + (p[i,1]-p[j,1])**2)
                min_d = r[i] + r[j]
                if dist < min_d:
                    pen += (min_d - dist)**2
        return -np.sum(r) + 1000.0 * pen

    # Bounds for L-BFGS-B
    # x_i in [0, 1]
    # y_i in [0, 1]
    # r_i in [0, 0.5] (max possible radius in unit square is 0.5)
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
    for i in range(n):
        bounds.append((0.0, 0.5)) # r

    start_time = time.time()
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'maxfun': 20000, 'max_time': time_limit})
        
        if res.success:
            p = res.x[:2*n].reshape(n, 2)
            r = res.x[2*n:]
            return p, r
        else:
            return pos, radii
    except Exception:
        return pos, radii

def _is_valid(xy, r, tol=1e-9):
    n = len(r)
    if np.any(r < -tol):
        return False
    if np.any(xy[:, 0] - r < -tol) or np.any(1 - xy[:, 0] - r < -tol):
        return False
    if np.any(xy[:, 1] - r < -tol) or np.any(1 - xy[:, 1] - r < -tol):
        return False
    dx = xy[:, 0][:, None] - xy[:, 0][None, :]
    dy = xy[:, 1][:, None] - xy[:, 1][None, :]
    dist = np.sqrt(dx**2 + dy**2)
    min_dist = r[:, None] + r[None, :]
    iu = np.triu_indices(n, k=1)
    if np.any(dist[iu] < min_dist[iu] - tol):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_score = -np.inf
    best_result = None
    
    # Try multiple initial configurations
    methods = ['grid', 'halton', 'random']
    n_restarts = 10
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break
            
        # Choose method
        method = methods[restart % len(methods)]
        
        # Initialize
        if method == 'random':
            np.random.seed(restart)
            pos = _initial_positions(N, method='random')
        else:
            pos = _initial_positions(N, method=method)
        
        # Initial radii: small
        # Estimate max possible radius based on grid spacing
        # For a 6x6 grid, spacing is ~1/6. Radius ~ 1/12.
        # Let's start with 0.01
        radii = np.full(N, 0.01)
        
        # Simulate
        sim_time = 30.0
        pos_sim, radii_sim = _simulate_expansion(pos, radii, max_time=sim_time)
        
        # Polish
        pos_pol, radii_pol = _polish(pos_sim, radii_sim, time_limit=10.0)
        
        # Validate and fix
        if not _is_valid(pos_pol, radii_pol):
            # Try to fix by scaling down radii
            for scale in np.linspace(1.0, 0.5, 20):
                r_test = radii_pol * scale
                if _is_valid(pos_pol, r_test):
                    radii_pol = r_test
                    break
            else:
                # Fallback to simulation result if polish failed badly
                if _is_valid(pos_sim, radii_sim):
                    pos_pol, radii_pol = pos_sim, radii_sim
                else:
                    # Last resort: shrink sim result
                    for scale in np.linspace(1.0, 0.5, 20):
                        r_test = radii_sim * scale
                        if _is_valid(pos_sim, r_test):
                            radii_pol = r_test
                            pos_pol = pos_sim
                            break
                    else:
                        continue
        
        score = np.sum(radii_pol)
        
        if score > best_score:
            best_score = score
            best_result = np.column_stack([pos_pol, radii_pol])
            
    if best_result is not None:
        return best_result
    
    # Fallback: simple grid
    side = 6
    xy = np.zeros((N, 2))
    r = np.zeros(N)
    for i in range(N):
        row = i // side
        col = i % side
        xy[i] = [(col + 0.5) / side, (row + 0.5) / side]
        r[i] = 0.5 / side - 0.005
    return np.column_stack([xy, r])

if __name__ == "__main__":
    res = construct()
    print(res)
