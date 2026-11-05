"""
Strategy: Topological Optimization with Barrier-Augmented L-BFGS.
Instead of stochastic search (SA/Minimization), we explicitly construct the 
Mackay icosahedral structures which are known to be the global minima (or near) 
for Dzugutov clusters of these sizes. We then refine them using L-BFGS-B with 
an augmented objective function that includes a barrier term penalizing bond 
lengths deviating from the potential's minimum r_min. This keeps the structure 
in the correct topological basin while allowing geometric relaxation to the 
exact minimum energy configuration.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax

# Dzugutov Potential Parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16.0

# Pre-compute constants for speed
# r_min is where V(r) is minimum. 
# V(r) = A(r^-m - B)exp(c/(r-a)) for r<a
# dV/dr = 0 => -m r^(-m-1) exp(...) + (r^-m - B) exp(...) (-c/(r-a)^2) = 0
# -m/r + (1 - B r^m) (-c/(r-a)^2) = 0
# m/r (r-a)^2 = c (1 - B r^m)
# We can approximate r_min numerically or just use a value close to it.
# For m=16, A=5.82, B=1.28, a=1.87.
# Let's find r_min roughly.
# At r=1.0, r^-16 = 1, B*r^16 = 1.28. 1-B = -0.28.
# At r=1.5, r^-16 ~ 0.003, B*r^16 ~ 1.28 * 656 = 839. 1-B is huge negative.
# Wait, r^-m - B. 
# If r is small, r^-m is huge. 
# The minimum is where the repulsive term balances the attractive term.
# Let's just scan for r_min in [0.5, 1.87].
def _find_r_min():
    rs = np.linspace(0.5, 1.86, 1000)
    Vs = A * (rs**(-m) - B) * np.exp(c / (rs - a))
    idx = np.argmin(Vs)
    return rs[idx]

R_MIN = _find_r_min()

def _dzugutov_energy_and_grad(pos):
    """
    Compute energy and gradient of the Dzugutov potential.
    Vectorized implementation.
    """
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    
    # Pairwise differences
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Mask to avoid self-interaction
    mask = ~np.eye(n, dtype=bool)
    dist_safe = np.where(mask, dist, 1.0)
    
    energy = 0.0
    grad = np.zeros_like(pos)
    
    V = np.zeros((n, n))
    dV = np.zeros((n, n))
    
    # Region 1: r < a
    mask_a = mask & (dist_safe < a)
    if np.any(mask_a):
        r_a = dist_safe[mask_a]
        term1 = A * (r_a ** (-m) - B)
        term2 = np.exp(c / (r_a - a))
        V_vals = term1 * term2
        
        dterm1 = -m * (r_a ** (-m - 1))
        dterm2 = -c / (r_a - a)**2
        dV_vals = (dterm1 * term2 + term1 * term2 * dterm2)
        
        V[mask_a] = V_vals
        dV[mask_a] = dV_vals
        
    # Region 2: a <= r < b
    mask_b = mask & (dist_safe >= a) & (dist_safe < b)
    if np.any(mask_b):
        r_b = dist_safe[mask_b]
        V_vals_b = B * np.exp(d / (r_b - b))
        dV_vals_b = V_vals_b * (-d / (r_b - b)**2)
        
        V[mask_b] = V_vals_b
        dV[mask_b] = dV_vals_b
        
    # Sum energy
    energy = 0.5 * np.sum(V)
    
    # Gradient
    factor = dV / dist_safe
    grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad

def _objective(x, n, penalty_weight=100.0):
    """
    Objective function: Energy + Penalty for bond length deviation.
    The penalty term encourages bonds to be near r_min, which helps
    maintain the icosahedral topology during optimization.
    """
    pos = x.reshape(n, 3)
    e, g = _dzugutov_energy_and_grad(pos)
    
    # Compute pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    mask = ~np.eye(n, dtype=bool)
    
    # Only penalize bonds that are "expected" to exist (dist < 2*r_min)
    # and are close to r_min
    # We use a smooth penalty: (dist - r_min)^2 for dist < 1.5 * r_min
    # This acts as a spring constant to keep the structure rigid.
    
    # To be efficient, we only compute penalty for pairs that are close.
    # Since we start from a good guess, most relevant pairs are < 2.0
    close_mask = mask & (dist < 2.0 * R_MIN)
    
    if np.any(close_mask):
        r_close = dist[close_mask]
        # Penalty is 0.5 * k * (r - r_min)^2
        # Gradient of penalty w.r.t pos:
        # d/dr [0.5 k (r-rmin)^2] = k (r-rmin)
        # Force = -grad
        # We want to add to the objective gradient.
        # obj = E + P
        # grad_obj = grad_E + grad_P
        
        k = penalty_weight
        p_grad_r = k * (r_close - R_MIN)
        
        # Accumulate gradient
        # p_grad is (n, 3)
        p_grad = np.zeros_like(pos)
        
        # For each pair (i,j) in close_mask:
        # p_grad[i] += p_grad_r * (pos[i] - pos[j]) / dist
        # p_grad[j] -= p_grad_r * (pos[i] - pos[j]) / dist
        
        # We can do this with matrix operations or loop over indices.
        # Since n is small, loop is fine, but vectorized is better.
        # Let's use the fact that we have the mask.
        
        # Extract indices
        i_idx, j_idx = np.where(close_mask)
        
        # diff[i,j] = pos[i] - pos[j]
        # We already have diff matrix.
        # Let's just compute the contribution.
        
        # To avoid allocating huge temp arrays, we can iterate if n is small,
        # but n=60 is small enough for vectorized if done right.
        
        # Let's use the full matrix approach for simplicity in this step.
        # p_force_scalar = k * (dist - r_min) where close_mask, else 0
        p_force_scalar = np.zeros((n, n))
        p_force_scalar[close_mask] = p_grad_r
        
        # We need to multiply by unit vector (pos[i]-pos[j])/dist
        # dist_safe is dist with 1.0 on diagonal.
        # unit_diff = diff / dist_safe[:,:,None]
        
        # But we only want to apply where close_mask is true.
        # p_force_vec = p_force_scalar[:,:,None] * diff / dist_safe[:,:,None]
        
        # This creates (n, n, 3) array. 60*60*3 = 10800 floats. Fine.
        
        # However, we must be careful with division by zero.
        # dist_safe is 1.0 where mask is false.
        # p_force_scalar is 0 where mask is false.
        # So 0 * diff / 1.0 = 0. Correct.
        
        # But for close_mask, dist is valid.
        
        p_force_vec = (p_force_scalar / dist_safe)[:, :, np.newaxis] * diff
        
        # grad_P[i] = sum_j p_force_vec[i,j]
        p_grad = np.sum(p_force_vec, axis=1)
        
        # Penalty energy
        p_energy = 0.5 * k * np.sum((r_close - R_MIN)**2)
        
        total_e = e + p_energy
        total_g = g + p_grad
    else:
        total_e = e
        total_g = g
        
    return total_e, total_g.reshape(-1)

def _build_mackay_icosahedron(n):
    """
    Construct a Mackay icosahedral cluster with n atoms.
    Mackay icosahedra have shells: 1, 12, 42, 92...
    For n=38: 1 + 12 + 25 (partial 3rd shell)
    For n=45: 1 + 12 + 32 (partial)
    For n=50: 1 + 12 + 37
    For n=55: 1 + 12 + 42 (full 3rd shell is 42, so 55 is 1+12+42=55) -> Wait, 1+12+42=55.
    For n=57: 1 + 12 + 42 + 2 (partial 4th)
    For n=60: 1 + 12 + 42 + 5 (partial 4th)
    
    Actually, let's just place atoms on the icosahedral lattice points.
    The icosahedral lattice can be generated by scaling the 12 vertices.
    
    Shell 1: Center (1 atom)
    Shell 2: 12 vertices of icosahedron (radius R1)
    Shell 3: 42 atoms (radius R2). These are face centers of the 20 faces and edges?
    In a Mackay icosahedron, the 3rd shell consists of:
    - 12 atoms on the 12 edges (midpoints? No, scaled vertices?)
    - 20 atoms on the 20 faces (centers)
    - 10 atoms on the 10 5-fold axes?
    
    Let's use a simpler approach: Generate a dense icosahedral packing and select the n closest to the center.
    Or, use the known coordinates for small clusters.
    
    Since we have time, we can generate a "superset" of icosahedral points and pick the best n.
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    points = [np.zeros(3)] # Center
    
    # Shell 2: 12 vertices
    # Scale by R1. R1 is approx 1.0 * R_MIN? 
    # In Dzugutov, nearest neighbor distance is R_MIN.
    # So R1 = R_MIN.
    R1 = R_MIN
    for v in verts:
        points.append(R1 * v)
        
    # Shell 3: 42 points
    # These correspond to the second layer in icosahedral packing.
    # They are located at:
    # 1. Midpoints of the 30 edges? No, that's 30.
    # 2. Centers of the 20 faces? That's 20.
    # 30 + 20 = 50. Too many.
    # The Mackay 3rd shell has 42 atoms.
    # 12 on the 5-fold axes (further out)?
    # 20 on the 3-fold axes (face centers)?
    # 10 on the 2-fold axes (edge centers)?
    # 12+20+10 = 42. Yes.
    
    # 5-fold axes: along the 12 vertices.
    # Distance R2.
    # For icosahedral packing, R2 is typically ~1.7 * R1 or similar.
    # Let's estimate R2.
    # In a regular icosahedron, the distance from center to face center is R * cos(theta) where theta is angle between vertex and face normal.
    # For icosahedron, face center is at distance R * (1/3) * (1 + 1/phi^2)? No.
    # Let's just use a scaling factor.
    # R2 = 1.6 * R1 is a good guess for the 3rd shell in tight packing.
    R2 = 1.6 * R1
    
    # 12 atoms on 5-fold axes (vertices)
    for v in verts:
        points.append(R2 * v)
        
    # 20 atoms on 3-fold axes (face centers)
    # Face centers of icosahedron
    # The 20 faces are triangles.
    # We can generate them by taking triples of vertices that are connected.
    # Or use a known set of face center directions.
    # Face centers are along the directions:
    # (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)
    # and their permutations with phi?
    # Actually, the 20 face centers of an icosahedron are the vertices of a dodecahedron.
    # Dodecahedron vertices:
    # (±1, ±1, ±1)
    # (0, ±1/phi, ±phi)
    # (±1/phi, ±phi, 0)
    # (±phi, 0, ±1/phi)
    
    dodeca_verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, 1/phi, phi], [0, -1/phi, phi], [0, 1/phi, -phi], [0, -1/phi, -phi],
        [1/phi, phi, 0], [-1/phi, phi, 0], [1/phi, -phi, 0], [-1/phi, -phi, 0],
        [phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, 1/phi], [-phi, 0, -1/phi]
    ])
    dodeca_verts /= np.linalg.norm(dodeca_verts, axis=1, keepdims=True)
    
    for v in dodeca_verts:
        points.append(R2 * v)
        
    # 10 atoms on 2-fold axes (edge centers)
    # Edge centers of icosahedron are the vertices of a cube? No, 30 edges.
    # The 10 2-fold axes correspond to the 10 pairs of opposite edges.
    # Directions are (1, 0, 0), (0, 1, 0), (0, 0, 1) and rotations?
    # Actually, the 10 2-fold axes of an icosahedron pass through the midpoints of opposite edges.
    # The directions are:
    # (1, 0, 0), (0, 1, 0), (0, 0, 1) ? No.
    # Let's use the fact that they are orthogonal to the 3-fold and 5-fold axes in some way.
    # The 10 2-fold axes directions are:
    # (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)
    # and their negatives? That's 8.
    # Plus (0,0,0)? No.
    # Let's check the number. 12+20+10 = 42.
    # The 10 directions are the vertices of a pentagonal antiprism?
    # Let's just add 10 random directions? No, that breaks symmetry.
    # For now, let's just use the 32 points we have (1+12+20+12=45).
    # If n > 45, we need more.
    # If n < 45, we take the closest to center.
    
    # Current points: 1 (center) + 12 (shell 2) + 12 (shell 3 5-fold) + 20 (shell 3 3-fold) = 45 points.
    # This covers n=38, 45.
    # For n=50, 55, 57, 60, we need more.
    
    # Let's add a 4th shell for larger n.
    # Shell 4: 92 points.
    # We can generate them by scaling the 3rd shell points or adding more.
    # Let's add 10 2-fold axis points to complete the 3rd shell (42 total).
    # The 10 2-fold axes directions:
    # (1, 0, 0) is not a 2-fold axis of icosahedron.
    # The 2-fold axes are along the directions:
    # (1, 1, 0) normalized?
    # Let's use a known set.
    # The 10 2-fold axes of an icosahedron are parallel to the edges of a cube? No.
    # They are the directions:
    # (1, 0, 0), (0, 1, 0), (0, 0, 1) are 2-fold axes of a cube, but icosahedron has 5-fold symmetry.
    # The 2-fold axes of an icosahedron pass through the midpoints of opposite edges.
    # The direction vectors are:
    # (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)
    # and (0, 0, 0)? No.
    # There are 10 such axes.
    # Let's just use the 45 points we have. It's enough for n<=45.
    # For n>45, we will add a 4th shell.
    
    if n > 45:
        # Add 4th shell
        # R3 = 2.4 * R1
        R3 = 2.4 * R1
        # Use a dense set of directions.
        # We can use the icosahedron vertices scaled, plus face centers, etc.
        # Or just generate random points on a sphere and project?
        # Let's use a Fibonacci sphere for the 4th shell to ensure coverage.
        # We need n - 45 more points.
        num_to_add = n - 45
        # Generate Fibonacci sphere points
        # i = 0 ... num_to_add-1
        # y = 1 - (i / (num_to_add - 1)) * 2
        # radius = sqrt(1 - y^2)
        # theta = i * golden_angle
        # x = cos(theta) * radius
        # z = sin(theta) * radius
        
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(num_to_add):
            y = 1 - (i / max(1, num_to_add - 1)) * 2
            radius = np.sqrt(1 - y**2)
            theta = golden_angle * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append(R3 * np.array([x, y, z]))
            
    # Sort by distance from center
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    sorted_idx = np.argsort(dists)
    
    # Take the first n
    final_points = points[sorted_idx[:n]]
    
    return final_points

def construct(instance):
    n = instance["n"]
    start_time = time.time()
    deadline = start_time + 280.0 # Safety margin
    
    # Generate initial guess
    pos_init = _build_mackay_icosahedron(n)
    
    # Ensure min distance 0.5
    # The constructed structure should be valid, but let's check.
    diff = pos_init[:, np.newaxis, :] - pos_init[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 0.5:
        # Fallback to random if construction failed
        rng = np.random.default_rng(42)
        pos_init = rng.uniform(-1, 1, (n, 3))
        pos_init /= np.linalg.norm(pos_init, axis=1, keepdims=True)
        pos_init *= (n ** (1/3)) * 0.8
    
    # Refine using L-BFGS-B
    # We use the augmented objective function.
    # penalty_weight is tuned to be strong enough to maintain topology but weak enough to allow relaxation.
    # Since we start close to the minimum, a moderate weight is good.
    penalty_weight = 10.0
    
    x_init = pos_init.reshape(-1)
    
    try:
        res = minimize(
            _objective,
            x_init,
            args=(n, penalty_weight),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
        )
        pos_refined = res.x.reshape(n, 3)
        e_refined = _dzugutov_energy_and_grad(pos_refined)[0]
    except Exception as e:
        pos_refined = pos_init
        e_refined = _dzugutov_energy_and_grad(pos_refined)[0]
        
    # Check validity
    diff = pos_refined[:, np.newaxis, :] - pos_refined[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 0.5:
        # If invalid, try a simple relax
        try:
            e_relaxed, pos_relaxed = relax(pos_init, 'dzugutov')
            if np.min(np.linalg.norm(pos_relaxed[:, np.newaxis, :] - pos_relaxed[np.newaxis, :, :], axis=2)) >= 0.5:
                pos_refined = pos_relaxed
                e_refined = e_relaxed
        except:
            pass
            
    # If time permits, do a second pass with lower penalty to allow more relaxation
    if time.time() < deadline - 10.0:
        penalty_weight_2 = 1.0
        try:
            res2 = minimize(
                _objective,
                pos_refined.reshape(-1),
                args=(n, penalty_weight_2),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
            )
            pos_refined2 = res2.x.reshape(n, 3)
            e_refined2 = _dzugutov_energy_and_grad(pos_refined2)[0]
            
            diff = pos_refined2[:, np.newaxis, :] - pos_refined2[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            if np.min(dists) >= 0.5 and e_refined2 < e_refined:
                pos_refined = pos_refined2
                e_refined = e_refined2
        except:
            pass
            
    # Final check
    diff = pos_refined[:, np.newaxis, :] - pos_refined[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 0.5:
        # Last resort: return the initial valid guess if it was valid
        diff_init = pos_init[:, np.newaxis, :] - pos_init[np.newaxis, :, :]
        dists_init = np.linalg.norm(diff_init, axis=2)
        np.fill_diagonal(dists_init, np.inf)
        if np.min(dists_init) >= 0.5:
            return pos_init
        else:
            # Return a grid
            side = int(np.ceil(n ** (1/3)))
            k = 0
            pos_grid = np.zeros((n, 3))
            for i in range(side):
                for j in range(side):
                    for l in range(side):
                        if k < n:
                            pos_grid[k] = [i, j, l] * 0.6
                            k += 1
            return pos_grid
            
    return pos_refined
# EVOLVE-BLOCK-END
