"""Morse cluster global minima search with multi-start basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global time management
_GLOBAL_START = [None]
_DEADLINE = 315.0  # Safety margin before 330s kill

def _get_remaining():
    if _GLOBAL_START[0] is None:
        _GLOBAL_START[0] = time.time()
    return _DEADLINE - (time.time() - _GLOBAL_START[0])

def _morse_energy(pos, rho):
    """Compute Morse potential energy."""
    n = len(pos)
    E = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(pos[i] - pos[j])
            x = rho * (1.0 - r)
            ex = math.exp(x)
            E += ex * (ex - 2.0)
    return E

def _morse_grad(pos, rho):
    """Compute gradient of Morse potential energy."""
    n = len(pos)
    g = np.zeros_like(pos)
    for i in range(n):
        for j in range(i + 1, n):
            d = pos[i] - pos[j]
            r = np.linalg.norm(d)
            if r < 1e-10:
                continue
            x = rho * (1.0 - r)
            ex = math.exp(x)
            # dE/dr = 2*exp(x)*(exp(x)-1)*(-rho) for each pair
            # Actually: E_pair = exp(x)*(exp(x)-2), x = rho*(1-r)
            # dE_pair/dx = exp(x)*(exp(x)-2) + exp(x)*exp(x) = exp(x)*(2*exp(x)-2) = 2*exp(x)*(exp(x)-1)
            # dx/dr = -rho
            # dE_pair/dr = 2*exp(x)*(exp(x)-1)*(-rho) = -2*rho*exp(x)*(exp(x)-1)
            # dE_pair/dpos_i = dE_pair/dr * d(r)/dpos_i = dE_pair/dr * (pos_i - pos_j)/r
            coeff = -2.0 * rho * ex * (ex - 1.0) / r
            g[i] += coeff * d
            g[j] -= coeff * d
    return g

def _generate_fcc(n, spacing=1.0):
    """Generate FCC-like lattice points."""
    points = []
    # FCC lattice vectors
    a = spacing
    # Generate a cube of FCC points
    max_idx = int(math.ceil(n ** (1.0/3.0))) + 2
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # FCC: corners and face centers
                # Corner positions
                p1 = np.array([i, j, k]) * a
                # Face center positions (shifted by half in two directions)
                p2 = np.array([i + 0.5, j + 0.5, k]) * a
                p3 = np.array([i + 0.5, j, k + 0.5]) * a
                p4 = np.array([i, j + 0.5, k + 0.5]) * a
                for p in [p1, p2, p3, p4]:
                    if len(points) < n:
                        # Check minimum distance
                        if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                            points.append(p)
    # If we don't have enough, add random points
    rng = np.random.default_rng(42)
    while len(points) < n:
        p = rng.uniform(-3*a, 3*a, 3)
        if all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)
    return np.array(points[:n])

def _generate_bcc(n, spacing=1.0):
    """Generate BCC-like lattice points."""
    points = []
    a = spacing
    max_idx = int(math.ceil(n ** (1.0/3.0))) + 2
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # BCC: corners and body center
                p1 = np.array([i, j, k]) * a
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5]) * a
                for p in [p1, p2]:
                    if len(points) < n:
                        if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                            points.append(p)
    rng = np.random.default_rng(43)
    while len(points) < n:
        p = rng.uniform(-3*a, 3*a, 3)
        if all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)
    return np.array(points[:n])

def _generate_icosahedral(n, spacing=1.0):
    """Generate icosahedral shell structure."""
    points = [np.zeros(3)]  # Center
    rng = np.random.default_rng(44)
    # First shell: 12 vertices of icosahedron
    t = (1.0 + math.sqrt(5.0)) / 2.0
    ico_verts = [
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ]
    ico_verts = np.array(ico_verts, dtype=float)
    ico_verts /= np.linalg.norm(ico_verts[0])  # Normalize to unit sphere
    for v in ico_verts:
        if len(points) < n:
            p = v * spacing
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
    # Add more shells
    shell_radius = spacing
    while len(points) < n:
        for _ in range(30):
            p = rng.normal(0, shell_radius, 3)
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
                if len(points) >= n:
                    break
        shell_radius *= 1.3
    return np.array(points[:n])

def _generate_random_compact(n, spacing=1.0, seed=None):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 100:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
        attempts += 1
    # Fill remaining with random points
    while len(points) < n:
        p = rng.uniform(-radius*1.5, radius*1.5, 3)
        points.append(p)
    return np.array(points[:n])

def _generate_deformed(n, base_positions, noise_scale=0.1, seed=None):
    """Add noise to base positions."""
    rng = np.random.default_rng(seed)
    return base_positions + rng.normal(0, noise_scale, base_positions.shape)

def construct(instance):
    """Find low-energy Morse cluster configurations using multi-start basin hopping."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    potential = instance["potential"]
    
    _GLOBAL_START[0] = time.time()
    
    best_energy = float('inf')
    best_positions = None
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # FCC-based
    fcc_pos = _generate_fcc(n, spacing=1.0)
    initial_configs.append(("fcc", fcc_pos))
    initial_configs.append(("fcc_noise", _generate_deformed(n, fcc_pos, 0.15, seed=100)))
    initial_configs.append(("fcc_noise2", _generate_deformed(n, fcc_pos, 0.3, seed=101)))
    
    # BCC-based
    bcc_pos = _generate_bcc(n, spacing=1.0)
    initial_configs.append(("bcc", bcc_pos))
    initial_configs.append(("bcc_noise", _generate_deformed(n, bcc_pos, 0.15, seed=200)))
    
    # Icosahedral
    ico_pos = _generate_icosahedral(n, spacing=1.0)
    initial_configs.append(("ico", ico_pos))
    initial_configs.append(("ico_noise", _generate_deformed(n, ico_pos, 0.2, seed=300)))
    
    # Random compact
    for seed in [50, 51, 52, 53]:
        initial_configs.append((f"rand_{seed}", _generate_random_compact(n, 1.0, seed=seed)))
    
    # Try each initial configuration with basin hopping
    for idx, (name, pos) in enumerate(initial_configs):
        if _get_remaining() < 5.0:
            break
        
        # First, do local relaxation
        try:
            energy, relaxed_pos = relax(pos.copy(), potential, rho)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos.copy()
        except:
            continue
        
        # Basin hopping from the relaxed position
        current_pos = relaxed_pos.copy()
        current_energy = energy
        bh_iterations = 0
        max_bh_iter = 30 if n <= 50 else 15
        
        for _ in range(max_bh_iter):
            if _get_remaining() < 3.0:
                break
            
            # Random perturbation
            noise_scale = 0.3 + 0.1 * math.sin(bh_iterations * 0.5)
            perturbed = current_pos + np.random.default_rng(idx*1000 + _).normal(0, noise_scale, current_pos.shape)
            
            # Relax the perturbed configuration
            try:
                new_energy, new_pos = relax(perturbed, potential, rho)
            except:
                continue
            
            # Accept if lower energy or with some probability
            if new_energy < current_energy:
                current_pos = new_pos
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_positions = new_pos.copy()
            else:
                # Metropolis acceptance
                delta = new_energy - current_energy
                if delta < 0.5 and np.random.random() < math.exp(-delta / 0.1):
                    current_pos = new_pos
                    current_energy = new_energy
            
            bh_iterations += 1
    
    # If no valid result, generate a fallback
    if best_positions is None:
        fallback = _generate_random_compact(n, 1.0, seed=999)
        try:
            _, best_positions = relax(fallback, potential, rho)
        except:
            best_positions = fallback
    
    # Final check: ensure minimum distance constraint
    # If any pair is too close, nudge them apart
    for i in range(len(best_positions)):
        for j in range(i+1, len(best_positions)):
            d = best_positions[i] - best_positions[j]
            r = np.linalg.norm(d)
            if r < 0.5 and r > 1e-10:
                # Push apart
                push = (0.5 - r + 0.01) * d / r
                best_positions[i] += push / 2
                best_positions[j] -= push / 2
    
    return best_positions
# EVOLVE-BLOCK-END
