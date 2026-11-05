"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global time tracking
_START_TIME = [None]
_DEADLINE = 320.0  # Safety margin before 330s kill

def _remaining_time():
    if _START_TIME[0] is None:
        _START_TIME[0] = time.time()
    return _DEADLINE - (time.time() - _START_TIME[0])

def _morse_energy(positions, rho):
    """Compute Morse energy for validation or if relax fails."""
    n = len(positions)
    E = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            x = rho * (1.0 - r)
            ex = math.exp(x)
            E += ex * (ex - 2.0)
    return E

def _generate_fcc(n):
    """Generate an FCC cluster of n atoms."""
    max_coord = int(math.ceil(n ** (1/3))) + 2
    candidates = []
    a = math.sqrt(2)
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                p1 = np.array([i*a, j*a, k*a])
                p2 = np.array([i*a, (j+0.5)*a, (k+0.5)*a])
                p3 = np.array([(i+0.5)*a, j*a, (k+0.5)*a])
                p4 = np.array([(i+0.5)*a, (j+0.5)*a, k*a])
                
                for p in [p1, p2, p3, p4]:
                    if np.linalg.norm(p) < max_coord * a * 1.5:
                        candidates.append(p)
                        
    unique_points = set()
    for p in candidates:
        unique_points.add(tuple(np.round(p, 5)))
        
    unique_list = [np.array(p) for p in unique_points]
    unique_list.sort(key=lambda p: np.linalg.norm(p))
    
    if len(unique_list) < n:
        return np.array(unique_list[:n])
        
    return np.array(unique_list[:n])

def _generate_bcc(n):
    """Generate a BCC cluster of n atoms."""
    max_coord = int(math.ceil(n ** (1/3))) + 2
    candidates = []
    a = 2.0 / math.sqrt(3.0)
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                p1 = np.array([i*a, j*a, k*a])
                p2 = np.array([(i+0.5)*a, (j+0.5)*a, (k+0.5)*a])
                
                for p in [p1, p2]:
                    if np.linalg.norm(p) < max_coord * a * 1.5:
                        candidates.append(p)
                        
    unique_points = set()
    for p in candidates:
        unique_points.add(tuple(np.round(p, 5)))
        
    unique_list = [np.array(p) for p in unique_points]
    unique_list.sort(key=lambda p: np.linalg.norm(p))
    
    if len(unique_list) < n:
        return np.array(unique_list[:n])
        
    return np.array(unique_list[:n])

def _generate_icosahedral(n):
    """Generate an icosahedral cluster with concentric shells."""
    # Icosahedron vertices (12 vertices)
    phi = (1 + math.sqrt(5)) / 2  # golden ratio
    icosahedron = [
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1]
    ]
    # Normalize to unit sphere
    icosahedron = [np.array(v) / np.linalg.norm(v) for v in icosahedron]
    
    # Build concentric shells: 1 center, 12 first shell, 30 second shell, etc.
    # For simplicity, use 1 center + 12 first shell + additional random points
    points = [np.zeros(3)]  # center
    
    # First shell: 12 icosahedral vertices at distance 1
    for v in icosahedron:
        points.append(v * 1.0)
    
    # If we need more, add second shell at distance ~1.7 (approx)
    if n > 13:
        # Generate points on a sphere of radius 1.7 using Fibonacci sphere
        remaining = n - len(points)
        for i in range(remaining):
            t = (i + 0.5) / remaining
            z = 1 - 2 * t
            theta = math.acos(z) * (1 + phi)
            r = 1.7
            x = r * math.sin(theta) * math.cos(theta)
            y = r * math.sin(theta) * math.sin(theta)
            z_val = r * z
            points.append(np.array([x, y, z_val]))
    
    return np.array(points[:n])

def _generate_decahedral(n):
    """Generate a decahedral (Mackay) cluster."""
    # Approximate with a distorted FCC-like structure
    # For simplicity, use a cubic arrangement with some distortion
    side = int(math.ceil(n ** (1/3)))
    points = []
    for i in range(side):
        for j in range(side):
            for k in range(side):
                if len(points) < n:
                    # Add small random perturbation
                    p = np.array([i - side//2, j - side//2, k - side//2]) * 1.0
                    p += np.random.default_rng(i*100+j*10+k).normal(0, 0.05, 3)
                    points.append(p)
    
    if len(points) < n:
        # Fill remaining with random points
        rng = np.random.default_rng(123)
        while len(points) < n:
            p = rng.uniform(-side//2, side//2, 3)
            points.append(p)
    
    return np.array(points[:n])

def _generate_random_compact(n, seed=0):
    """Generate a random compact configuration."""
    rng = np.random.default_rng(seed)
    radius = 1.0 * n ** (1.0 / 3.0) * 0.9
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            # Check minimum distance
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
        attempts += 1
    
    if len(points) < n:
        # Fallback: just take what we have
        return np.array(points)
    
    return np.array(points[:n])

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    
    if _START_TIME[0] is None:
        _START_TIME[0] = time.time()
        
    # 1. Generate diverse initial configurations
    inits = []
    
    # FCC
    try:
        fcc_pos = _generate_fcc(size)
        if len(fcc_pos) == size:
            inits.append(fcc_pos)
    except:
        pass
        
    # BCC
    try:
        bcc_pos = _generate_bcc(size)
        if len(bcc_pos) == size:
            inits.append(bcc_pos)
    except:
        pass
        
    # Icosahedral
    try:
        ico_pos = _generate_icosahedral(size)
        if len(ico_pos) == size:
            inits.append(ico_pos)
    except:
        pass
        
    # Decahedral
    try:
        deca_pos = _generate_decahedral(size)
        if len(deca_pos) == size:
            inits.append(deca_pos)
    except:
        pass
        
    # Random compact (multiple seeds)
    for seed in [size, size + 1, size + 2]:
        try:
            rand_pos = _generate_random_compact(size, seed)
            if len(rand_pos) == size:
                inits.append(rand_pos)
        except:
            pass

    # Relax all initial configurations and keep the best few
    relaxed_configs = []
    for init_pos in inits:
        if _remaining_time() < 10.0:
            break
        try:
            energy, pos = relax(init_pos, potential, rho)
            relaxed_configs.append((energy, pos))
        except:
            continue
            
    if not relaxed_configs:
        # Fallback if all relaxations failed
        return inits[0] if inits else np.zeros((size, 3))

    # Sort by energy and keep top 3
    relaxed_configs.sort(key=lambda x: x[0])
    top_configs = relaxed_configs[:min(3, len(relaxed_configs))]
    
    # 2. Adaptive Basin Hopping from each top configuration
    best_energy = top_configs[0][0]
    best_pos = top_configs[0][1].copy()
    
    for config_energy, config_pos in top_configs:
        if _remaining_time() < 15.0:
            break
            
        current_pos = config_pos.copy()
        current_energy = config_energy
        
        # Adaptive parameters
        T = 0.5
        T_min = 0.05
        T_factor = 0.97
        step_size = 0.15
        max_steps = 5000
        stuck_count = 0
        improvement_threshold = 1e-6
        
        rng_bh = np.random.default_rng(42 + size)
        
        for step in range(max_steps):
            if _remaining_time() < 10.0:
                break
            
            # Adaptive step size: expand if stuck, contract if improving
            if stuck_count > 50:
                step_size = min(step_size * 1.2, 1.0)
                stuck_count = 0
            elif stuck_count > 20:
                step_size = min(step_size * 1.1, 1.0)
            
            # Propose move
            perturbation = rng_bh.normal(0, step_size, size=(size, 3))
            new_pos = current_pos + perturbation
            
            # Relax
            try:
                new_energy, new_pos_relaxed = relax(new_pos, potential, rho)
            except:
                continue
            
            # Accept/Reject
            delta_E = new_energy - current_energy
            if delta_E < 0 or rng_bh.random() < math.exp(-delta_E / T):
                current_pos = new_pos_relaxed
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pos = current_pos.copy()
                    stuck_count = 0
                elif current_energy - best_energy > improvement_threshold:
                    stuck_count += 1
            else:
                stuck_count += 1
                
            # Cool down
            T = max(T * T_factor, T_min)
            
            # If we've found a very good solution, reduce step size
            if best_energy < -size * 0.9:
                step_size = max(step_size * 0.95, 0.05)
        
        # Reheat occasionally
        T = max(T * 1.5, 0.3)

    return best_pos
# EVOLVE-BLOCK-END
