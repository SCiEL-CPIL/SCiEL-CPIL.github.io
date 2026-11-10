"""Morse cluster global minima via Basin Hopping over structured seeds."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def morse_energy_and_grad(positions, rho):
    """
    Compute total Morse energy and gradient for a cluster.
    E = sum_{i<j} exp(rho(1-r_ij)) * (exp(rho(1-r_ij)) - 2)
    """
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    # Compute pairwise distances
    # diff[i, j] = positions[i] - positions[j]
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    
    # Avoid division by zero and singularities
    dist = np.maximum(dist, 1e-12)
    np.fill_diagonal(dist, np.inf)
    
    # Calculate pair energies
    x = rho * (1.0 - dist)
    exp_x = np.exp(x)
    pair_energy = exp_x * (exp_x - 2.0)
    
    # Total energy is half the sum of the matrix (since i<j and j<i)
    total_energy = 0.5 * np.sum(pair_energy)
    
    # Calculate gradient
    # dE/dr_ij = -2 * rho * exp_x * (exp_x - 1)
    # dE/dx_i = sum_{j != i} (dE/dr_ij) * (r_ij / dist_ij)
    # Note: dE/dr_ij is the derivative with respect to the distance r_ij.
    # The force on atom i is -dE/dx_i. The gradient is dE/dx_i.
    
    # dE_dr = -2 * rho * exp_x * (exp_x - 1)
    dE_dr = -2.0 * rho * exp_x * (exp_x - 1.0)
    
    # Set diagonal to 0 to avoid self-interaction
    np.fill_diagonal(dE_dr, 0.0)
    
    # Gradient vector: grad[i] = sum_{j != i} dE_dr[i,j] * (positions[i] - positions[j]) / dist[i,j]
    # We can compute this as: sum_j (dE_dr[i,j] / dist[i,j]) * (positions[i] - positions[j])
    # Let W[i,j] = dE_dr[i,j] / dist[i,j]
    W = dE_dr / dist
    
    # grad[i] = sum_j W[i,j] * (positions[i] - positions[j])
    # grad[i] = positions[i] * sum_j W[i,j] - sum_j W[i,j] * positions[j]
    # sum_j W[i,j] is the row sum of W
    row_sum_W = np.sum(W, axis=1)
    
    # sum_j W[i,j] * positions[j] is W @ positions
    weighted_pos = W @ positions
    
    gradient = positions * row_sum_W[:, np.newaxis] - weighted_pos
    
    return total_energy, gradient

def generate_fcc_seed(n, rng):
    """Generate an FCC-like seed."""
    # Approximate radius for n atoms in FCC packing
    # n ~ (4/3) * pi * R^3 / (4 * a^3/2) ... roughly R ~ n^(1/3)
    R = n ** (1.0/3.0) * 1.5
    points = []
    # Generate FCC lattice points within a sphere
    a = 1.0 # lattice constant
    max_idx = int(2 * R / a) + 5
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
                for b in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i*a + b[0]*a, j*a + b[1]*a, k*a + b[2]*a])
                    if np.linalg.norm(p) <= R:
                        points.append(p)
    
    if len(points) < n:
        return None
        
    # Shuffle and take first n
    points = np.array(points)
    idx = rng.choice(len(points), n, replace=False)
    return points[idx]

def generate_bcc_seed(n, rng):
    """Generate a BCC-like seed."""
    R = n ** (1.0/3.0) * 1.5
    points = []
    a = 1.0
    max_idx = int(2 * R / a) + 5
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # BCC basis: (0,0,0), (0.5,0.5,0.5)
                for b in [(0,0,0), (0.5,0.5,0.5)]:
                    p = np.array([i*a + b[0]*a, j*a + b[1]*a, k*a + b[2]*a])
                    if np.linalg.norm(p) <= R:
                        points.append(p)
    
    if len(points) < n:
        return None
        
    points = np.array(points)
    idx = rng.choice(len(points), n, replace=False)
    return points[idx]

def generate_icosahedral_seed(n, rng):
    """Generate an icosahedral seed (for smaller n)."""
    if n > 150:
        return None
        
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts /= np.linalg.norm(verts[0]) # Normalize to radius 1
    
    points = [verts[0].copy() * 0.5] # Center
    # First shell
    points.extend([v * 1.0 for v in verts])
    
    # Second shell (approximate)
    # Just add random points if we need more
    while len(points) < n:
        # Random point on sphere
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        p = u * 1.8
        # Check distance to existing
        if all(np.linalg.norm(p - q) > 0.9 for q in points):
            points.append(p)
            
    if len(points) < n:
        return None
        
    return np.array(points[:n])

def local_relax(positions, rho, maxiter=100):
    """Perform local minimization using L-BFGS-B."""
    def objective(x):
        pos = x.reshape(-1, 3)
        # Penalty for minimum distance constraint
        # We want min dist >= 0.5
        # If dist < 0.5, add a large penalty
        # This is a soft constraint for the optimizer
        # Better to just let the optimizer run and check validity later
        e, g = morse_energy_and_grad(pos, rho)
        return e, g
        
    try:
        res = minimize(objective, positions.flatten(), jac=True, method='L-BFGS-B', 
                       options={'maxiter': maxiter, 'gtol': 1e-5})
        if res.success:
            return res.x.reshape(-1, 3), res.fun
        else:
            return positions, morse_energy_and_grad(positions, rho)[0]
    except Exception:
        return positions, morse_energy_and_grad(positions, rho)[0]

def is_valid(positions):
    """Check if all pairwise distances are >= 0.5."""
    n = len(positions)
    if n < 2:
        return True
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    return min_dist >= 0.5

def construct(instance):
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 290.0 # Safety margin
    
    rng = np.random.default_rng(42)
    
    best_energy = np.inf
    best_positions = None
    
    # Generate initial seeds
    seeds = []
    fcc = generate_fcc_seed(n, rng)
    if fcc is not None:
        seeds.append(fcc)
    bcc = generate_bcc_seed(n, rng)
    if bcc is not None:
        seeds.append(bcc)
    ico = generate_icosahedral_seed(n, rng)
    if ico is not None:
        seeds.append(ico)
        
    # Add some random seeds
    for _ in range(2):
        R = n ** (1.0/3.0) * 1.2
        points = []
        while len(points) < n:
            p = rng.uniform(-R, R, 3)
            if np.linalg.norm(p) <= R:
                # Simple rejection sampling for min dist
                if all(np.linalg.norm(p - q) > 0.6 for q in points):
                    points.append(p)
        if len(points) == n:
            seeds.append(np.array(points))
            
    if not seeds:
        # Fallback: random uniform in sphere
        R = n ** (1.0/3.0) * 1.5
        points = rng.uniform(-R, R, (n, 3))
        # Normalize to fit in sphere
        norms = np.linalg.norm(points, axis=1)
        max_norm = np.max(norms)
        if max_norm > 0:
            points = points * (R / max_norm)
        seeds.append(points)
        
    # Basin Hopping
    current_positions = seeds[0].copy()
    
    # Local relax first
    current_positions, current_energy = local_relax(current_positions, rho)
    if is_valid(current_positions) and current_energy < best_energy:
        best_energy = current_energy
        best_positions = current_positions.copy()
        
    # Basin Hopping loop
    step_size = 0.5
    T = 1.0
    T_min = 0.01
    
    for i in range(1000): # Max iterations
        if time.time() > deadline:
            break
            
        # Perturb
        noise = rng.normal(0, step_size, current_positions.shape)
        new_positions = current_positions + noise
        
        # Local relax
        new_positions, new_energy = local_relax(new_positions, rho)
        
        # Check validity
        if not is_valid(new_positions):
            # Try to repair or reject
            # For simplicity, reject
            T *= 0.9
            continue
            
        # Acceptance criterion (Metropolis)
        delta_e = new_energy - current_energy
        if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
            current_positions = new_positions
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = current_positions.copy()
                
        # Cooling
        T *= 0.9
        if T < T_min:
            T = T_min
            
    # Final polish on best
    if best_positions is not None:
        # Try a few more local relaxes with different maxiter
        for _ in range(3):
            if time.time() > deadline:
                break
            polished, e = local_relax(best_positions, rho, maxiter=200)
            if is_valid(polished) and e < best_energy:
                best_energy = e
                best_positions = polished

    return best_positions
# EVOLVE-BLOCK-END
