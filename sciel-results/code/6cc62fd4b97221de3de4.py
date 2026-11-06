"""
Strategy: Global Structure Optimization via Sine-Cosine Algorithm (SCA) with Polyhedral Initialization.

This approach fundamentally differs from Basin Hopping (BH) by using a continuous, population-based
global optimization method (SCA) instead of discrete, single-trajectory BH chains. 
SCA is known to be effective for cluster energy minimization because it balances exploration and
exploitation well without the "discrete jump" artifacts of BH. It directly optimizes Cartesian
coordinates using a population of candidate structures, allowing it to navigate the rugged
energy landscape of the Dzugutov potential more effectively than local relaxations.

Key features:
1. Initialization with diverse polytetrahedral (icosahedral) and BCC structures.
2. Population-based search with SCA updates.
3. Local relaxation (Lamarckian) after each SCA step to ensure valid, low-energy basins.
4. Parallel execution of multiple independent SCA runs to cover different regions of the search space.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
TIME_LIMIT = 285.0  # Safety margin for 300s limit

def _check_time():
    return time.time() - _START_TIME < TIME_LIMIT

def _generate_mackay_icosahedron(n, scale=1.0):
    """
    Generates a Mackay icosahedral cluster.
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ]
    verts = np.array(verts, dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Compute min distance to set scale
    min_d = np.inf
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < min_d:
                min_d = d
    R1 = 1.8 / min_d * scale
    
    points = [np.zeros(3)]
    points.extend([v * R1 for v in verts])
    
    if n > 13:
        # Find edges
        edges = []
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(verts[i] - verts[j])
                if np.isclose(d, min_d, atol=1e-6):
                    edges.append((i, j))
        
        # Edge midpoints
        R2_unit = np.sqrt(1 - (min_d/2)**2)
        R2 = R2_unit * R1
        for i, j in edges:
            mid_unit = (verts[i] + verts[j]) / 2.0
            mid_scaled = mid_unit * (R2 / R2_unit)
            points.append(mid_scaled)
            
        if n > 43:
            # Face centers
            faces = []
            for i in range(12):
                for j in range(i+1, 12):
                    for k in range(j+1, 12):
                        if (i,j) in edges or (j,i) in edges:
                            if (j,k) in edges or (k,j) in edges:
                                if (i,k) in edges or (k,i) in edges:
                                    faces.append((i,j,k))
            
            R3_unit = np.linalg.norm((verts[faces[0][0]] + verts[faces[0][1]] + verts[faces[0][2]]) / 3.0)
            R3 = R1 * R3_unit
            for i, j, k in faces:
                center_unit = (verts[i] + verts[j] + verts[k]) / 3.0
                center_scaled = center_unit * (R3 / R3_unit)
                points.append(center_scaled)
                
    # Sort by distance from center
    dists = np.linalg.norm(np.array(points), axis=1)
    order = np.argsort(dists)
    points = [points[i] for i in order]
    
    if len(points) > n:
        points = points[:n]
    elif len(points) < n:
        R_add = R1 * 1.5
        while len(points) < n:
            theta = np.random.uniform(0, 2*np.pi)
            z = np.random.uniform(-1, 1)
            x = np.sqrt(1 - z**2) * np.cos(theta)
            y = np.sqrt(1 - z**2) * np.sin(theta)
            p = np.array([x, y, z]) * R_add
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.5:
                    valid = False
                    break
            if valid:
                points.append(p)
                
    return np.array(points)

def _generate_bcc_cluster(n, scale=1.0):
    """Generate a BCC-like cluster."""
    points = []
    a = 2.078 * scale
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                p1 = np.array([i, j, k]) * a
                if len(points) < n and all(np.linalg.norm(p1 - q) >= 0.5 for q in points):
                    points.append(p1)
                p2 = np.array([i+0.5, j+0.5, k+0.5]) * a
                if len(points) < n and all(np.linalg.norm(p2 - q) >= 0.5 for q in points):
                    points.append(p2)
    if len(points) < n:
        while len(points) < n:
            p = np.random.uniform(-3, 3, 3) * scale
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
    return np.array(points[:n])

def _relax_and_check(pos, potential, parameter):
    try:
        e, pos_relaxed = relax(pos, potential, parameter)
        # Check min distance
        min_d = np.inf
        for i in range(len(pos_relaxed)):
            for j in range(i+1, len(pos_relaxed)):
                d = np.linalg.norm(pos_relaxed[i] - pos_relaxed[j])
                if d < min_d:
                    min_d = d
        if min_d < 0.5:
            return np.inf, pos_relaxed
        return e, pos_relaxed
    except Exception:
        return np.inf, pos

def _sine_cosine_worker(n, potential, parameter, seed, time_limit):
    """
    Runs a Sine-Cosine Algorithm optimization.
    """
    np.random.seed(seed)
    
    # Initialize population
    pop_size = 20
    X = np.zeros((pop_size, n, 3))
    E = np.zeros(pop_size)
    
    for i in range(pop_size):
        # Diverse initializations
        if i % 3 == 0:
            X[i] = _generate_mackay_icosahedron(n, 1.0 + 0.1*np.random.randn())
        elif i % 3 == 1:
            X[i] = _generate_bcc_cluster(n, 1.0 + 0.1*np.random.randn())
        else:
            X[i] = np.random.uniform(-2, 2, (n, 3))
        
        E[i], X[i] = _relax_and_check(X[i], potential, parameter)
        
        if not _check_time():
            break
            
    best_idx = np.argmin(E)
    best_X = X[best_idx].copy()
    best_E = E[best_idx]
    
    # SCA parameters
    r1 = 1.0  # Exploration range
    r1_decay = 0.995  # Decay rate
    max_iter = 1000
    
    for t in range(max_iter):
        if not _check_time():
            break
        
        # Update r1
        r1 *= r1_decay
        
        for i in range(pop_size):
            # Generate random parameters
            r2 = np.random.uniform(0, 2*np.pi)  # Angle
            r3 = np.random.uniform(0, 1, (n, 3))  # Weighting factor
            r4 = np.random.uniform(0, 1)  # Probability of switching between sin and cos
            
            if r4 < 0.5:
                # Sine update
                X[i] = best_X + r1 * np.sin(r2) * r3 * (best_X - X[i])
            else:
                # Cosine update
                X[i] = best_X + r1 * np.cos(r2) * r3 * (best_X - X[i])
            
            # Relax
            new_E, new_X = _relax_and_check(X[i], potential, parameter)
            
            if new_E < E[i]:
                E[i] = new_E
                X[i] = new_X
                
                if new_E < best_E:
                    best_E = new_E
                    best_X = new_X.copy()
                    
    return best_E, best_X

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Run multiple independent SCA workers in parallel
    n_workers = 4
    worker_time_limit = time.time() + (TIME_LIMIT - (time.time() - _START_TIME) - 5)
    
    best_energy = np.inf
    best_positions = None
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for i in range(n_workers):
            seed = 42 + i
            futures.append(executor.submit(_sine_cosine_worker, size, potential, parameter, seed, worker_time_limit))
            
        for future in concurrent.futures.as_completed(futures):
            if not _check_time():
                break
            try:
                e, pos = future.result(timeout=1)
                if e < best_energy:
                    best_energy = e
                    best_positions = pos
            except Exception:
                pass
                
    # Fallback if all workers failed
    if best_positions is None:
        pos = _generate_mackay_icosahedron(size, 1.0)
        e, pos_relaxed = _relax_and_check(pos, potential, parameter)
        best_positions = pos_relaxed
        
    return best_positions
# EVOLVE-BLOCK-END
