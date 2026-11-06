"""Improved program for Dzugutov cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
import time
import threading

# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16

def _dzugutov_energy(positions):
    """Compute total Dzugutov potential energy."""
    n = len(positions)
    if n < 2:
        return 0.0
    e = 0.0
    for i in range(n):
        for j in range(i+1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < 1e-10:
                return 1e10
            if r < a:
                e += A * (r**(-m) - B) * np.exp(c / (r - a))
            elif r < b:
                e += B * np.exp(d / (r - b))
    return e

def _icosahedron_vertices():
    """Return 12 vertices of a regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit radius
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, None]
    return verts

def _fibonacci_sphere(n, radius=1.0):
    """Generate n points approximately uniformly on a sphere of given radius."""
    if n <= 0:
        return np.array([]).reshape(0, 3)
    if n == 1:
        return np.array([[0.0, 0.0, radius]])
    
    golden_angle = np.pi * (3 - np.sqrt(5))
    points = []
    for i in range(n):
        z = 1 - 2 * i / (n - 1) if n > 1 else 0
        r = np.sqrt(1 - z**2)
        theta = golden_angle * i
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append([x, y, z])
    points = np.array(points) * radius
    return points

def _generate_mackay_icosahedron(n_target, spacing=1.1):
    """
    Generate a Mackay icosahedral cluster with approximately n_target atoms.
    Shells: core(1) + shell1(12) + shell2(42) = 55 total
    For n < 55, we take subsets; for n > 55, we add partial shell3.
    """
    ico_verts = _icosahedron_vertices()
    
    # Core
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Shell 1: 12 icosahedron vertices at radius r1
    r1 = spacing * 1.0
    shell1 = ico_verts * r1
    points.extend(list(shell1))
    
    # Shell 2: 42 atoms - face centers and edge midpoints of icosahedron
    # Icosahedron has 20 faces, 30 edges
    # Face centers: 20, Edge midpoints: 30 -> but some coincide with shell1 in Mackay
    # Actually Mackay shell 2 has 42 atoms: these are at the midpoints of edges 
    # and centers of faces, arranged with icosahedral symmetry
    
    # Let's use a different approach: generate points on icosahedral symmetry sites
    # Shell 2 radius r2 = r1 * sqrt(5)/2 approximately for Mackay
    r2 = spacing * 1.618  # golden ratio scaling
    
    # Generate 42 points with icosahedral symmetry
    # These correspond to: 12 at "edge midpoint" type, 20 at "face center" type, 10 at another type
    # Simplified: use Fibonacci sphere for shell 2, then relax
    
    if n_target > 13:
        n_shell2 = min(n_target - 13, 42)
        if n_shell2 > 0:
            # Use icosahedral symmetry sites for shell 2
            # Edge midpoints: 30 positions
            # Face centers: 20 positions
            # Total available: 50, but Mackay uses 42 specific ones
            
            # For simplicity, generate on sphere with icosahedral bias
            shell2_points = _fibonacci_sphere(n_shell2, r2)
            # Add some perturbation to break degeneracy
            rng = np.random.default_rng(42)
            shell2_points = shell2_points + rng.normal(0, 0.05, shell2_points.shape)
            points.extend(list(shell2_points))
    
    # If we need more than 55, add shell 3
    if n_target > 55:
        n_shell3 = n_target - 55
        r3 = spacing * 2.2
        shell3_points = _fibonacci_sphere(n_shell3, r3)
        rng = np.random.default_rng(43)
        shell3_points = shell3_points + rng.normal(0, 0.05, shell3_points.shape)
        points.extend(list(shell3_points))
    
    # Trim or pad to exact n_target
    points = np.array(points)
    if len(points) > n_target:
        # Keep core and closest to core
        dists = np.linalg.norm(points, axis=1)
        order = np.argsort(dists)
        points = points[order[:n_target]]
    elif len(points) < n_target:
        # Add random points near surface
        rng = np.random.default_rng(44)
        while len(points) < n_target:
            p = rng.normal(0, spacing, 3)
            if all(np.linalg.norm(p - q) >= 0.6 * spacing for q in points):
                points = np.vstack([points, p])
    
    return points[:n_target]

def _bcc_initial(n_target, spacing=1.1):
    """Generate BCC-like initial configuration."""
    points = []
    rng = np.random.default_rng(n_target)
    
    # Place atoms in a BCC lattice pattern
    a = spacing * 1.2
    count = 0
    for ix in range(-2, 3):
        for iy in range(-2, 3):
            for iz in range(-2, 3):
                if count >= n_target:
                    break
                # Corner atom
                p = np.array([ix, iy, iz]) * a
                if np.linalg.norm(p) <= a * 2.5:
                    points.append(p)
                    count += 1
                    if count >= n_target:
                        break
                # Body center
                p2 = (np.array([ix, iy, iz]) + 0.5) * a
                if np.linalg.norm(p2) <= a * 2.5 and count < n_target:
                    points.append(p2)
                    count += 1
    
    points = np.array(points)
    if len(points) > n_target:
        # Keep closest to center
        dists = np.linalg.norm(points, axis=1)
        order = np.argsort(dists)
        points = points[order[:n_target]]
    elif len(points) < n_target:
        # Add random
        while len(points) < n_target:
            p = rng.uniform(-a*2, a*2, 3)
            if all(np.linalg.norm(p - q) >= 0.6*spacing for q in points):
                points = np.vstack([points, p])
    
    return points[:n_target]

def _local_minimize(positions, max_iter=200, tol=1e-6):
    """Simple gradient-based local minimization."""
    positions = positions.copy()
    n = len(positions)
    
    for it in range(max_iter):
        # Compute gradient using finite differences or the provided function
        try:
            grad = dzugutov_energy_gradient(positions)
        except:
            # Fallback to finite differences
            grad = np.zeros_like(positions)
            h = 1e-5
            e0 = _dzugutov_energy(positions)
            for i in range(n):
                for j in range(3):
                    orig = positions[i, j]
                    positions[i, j] = orig + h
                    e_plus = _dzugutov_energy(positions)
                    positions[i, j] = orig - h
                    e_minus = _dzugutov_energy(positions)
                    positions[i, j] = orig
                    grad[i, j] = (e_plus - e_minus) / (2*h)
        
        gnorm = np.linalg.norm(grad)
        if gnorm < tol:
            break
        
        # Adaptive step size
        step = 0.01 * spacing / (gnorm + 1e-10)
        positions = positions - step * grad
        
        # Enforce minimum distance constraint
        for i in range(n):
            for j in range(i+1, n):
                d = positions[i] - positions[j]
                dist = np.linalg.norm(d)
                if dist < 0.5:
                    if dist < 1e-10:
                        positions[i] += np.random.normal(0, 0.1, 3)
                    else:
                        corr = (0.5 - dist) / dist * d
                        positions[i] += 0.5 * corr
                        positions[j] -= 0.5 * corr
    
    return positions

def construct(instance):
    """Find low-energy Dzugutov cluster configuration."""
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    n = instance["n"]
    potential = instance["potential"]
    
    # Generate diverse initial configurations
    rng = np.random.default_rng(n)
    
    candidates = []
    
    # 1. Mackay icosahedron based
    try:
        mackay = _generate_mackay_icosahedron(n, spacing=1.1)
        candidates.append(mackay)
    except:
        pass
    
    # 2. BCC based
    try:
        bcc = _bcc_initial(n, spacing=1.1)
        candidates.append(bcc)
    except:
        pass
    
    # 3. Random compact
    for trial in range(3):
        radius = 1.1 * n**(1/3) * 0.8
        points = []
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.7 for q in points):
                points.append(p)
        if len(points) == n:
            candidates.append(np.array(points))
    
    # 4. Icosahedron with random perturbations
    try:
        ico_base = _generate_mackay_icosahedron(n, spacing=1.1)
        for trial in range(2):
            perturbed = ico_base + rng.normal(0, 0.15, ico_base.shape)
            candidates.append(perturbed)
    except:
        pass
    
    # Relax each candidate and keep the best
    best_energy = 1e10
    best_positions = None
    
    for idx, cand in enumerate(candidates):
        if time.time() > deadline - 30:
            break
        
        # Try the built-in relax first
        try:
            energy, positions = relax(cand, potential)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except:
            pass
        
        # Also try our local minimizer
        try:
            relaxed = _local_minimize(cand, max_iter=100)
            e = _dzugutov_energy(relaxed)
            if e < best_energy:
                best_energy = e
                best_positions = relaxed.copy()
        except:
            pass
    
    # If we have time, do basin hopping from the best solution
    if best_positions is not None and time.time() < deadline - 60:
        current_pos = best_positions.copy()
        current_e = best_energy
        T = 0.5
        T_min = 0.01
        max_bh = 30
        
        for bh_iter in range(max_bh):
            if time.time() > deadline - 30:
                break
            
            # Perturb
            noise_scale = T * 0.3
            perturbed = current_pos + rng.normal(0, noise_scale, current_pos.shape)
            
            # Enforce minimum distance
            for i in range(n):
                for j in range(i+1, n):
                    d = perturbed[i] - perturbed[j]
                    dist = np.linalg.norm(d)
                    if dist < 0.5:
                        if dist < 1e-10:
                            perturbed[i] += rng.normal(0, 0.1, 3)
                        else:
                            corr = (0.5 - dist) / dist * d
                            perturbed[i] += 0.5 * corr
                            perturbed[j] -= 0.5 * corr
            
            # Local minimize
            try:
                relaxed = _local_minimize(perturbed, max_iter=50)
                e_new = _dzugutov_energy(relaxed)
                
                # Metropolis acceptance
                if e_new < current_e:
                    current_pos = relaxed
                    current_e = e_new
                    if e_new < best_energy:
                        best_energy = e_new
                        best_positions = relaxed.copy()
                else:
                    delta = e_new - current_e
                    if np.log(rng.uniform()) > -delta / T:
                        current_pos = relaxed
                        current_e = e_new
            except:
                pass
            
            T = max(T_min, T * 0.9)
    
    # Final check: ensure minimum distance constraint
    if best_positions is not None:
        for i in range(n):
            for j in range(i+1, n):
                d = best_positions[i] - best_positions[j]
                dist = np.linalg.norm(d)
                if dist < 0.5:
                    if dist < 1e-10:
                        best_positions[i] += rng.normal(0, 0.1, 3)
                    else:
                        corr = (0.5 - dist) / dist * d
                        best_positions[i] += 0.5 * corr
                        best_positions[j] -= 0.5 * corr
    
    if best_positions is None:
        # Fallback: simple random
        points = rng.uniform(-3, 3, (n, 3))
        best_positions = points
    
    return best_positions
# EVOLVE-BLOCK-END
