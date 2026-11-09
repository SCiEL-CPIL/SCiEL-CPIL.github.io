"""Morse cluster global minima via structured initial geometries + basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy(positions, rho):
    """Compute exact Morse energy for a cluster."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(r, np.inf)
    iu = np.triu_indices(n, k=1)
    r_pair = r[iu]
    x = rho * (1.0 - r_pair)
    exp_x = np.exp(x)
    e_pair = exp_x * (exp_x - 2.0)
    return float(np.sum(e_pair))


def _generate_fcc(n):
    """Generate n atoms in FCC-like arrangement."""
    atoms = []
    # FCC lattice points in a cube
    a = 1.0  # lattice parameter (equilibrium spacing)
    # Generate FCC points: corners + face centers
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                # Corner
                atoms.append([i*a, j*a, k*a])
                # Face centers
                atoms.append([(i+0.5)*a, (j+0.5)*a, k*a])
                atoms.append([(i+0.5)*a, j*a, (k+0.5)*a])
                atoms.append([i*a, (j+0.5)*a, (k+0.5)*a])
    atoms = np.array(atoms)
    # Center
    atoms -= atoms.mean(axis=0)
    # Sort by distance from center, take first n
    dists = np.linalg.norm(atoms, axis=1)
    order = np.argsort(dists)
    atoms = atoms[order[:n]]
    return atoms


def _generate_hcp(n):
    """Generate n atoms in HCP-like arrangement."""
    atoms = []
    a = 1.0
    c = a * np.sqrt(8.0/3.0)
    # HCP: ABAB stacking
    for layer in range(-4, 5):
        z = layer * c
        # A layer
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i * a + (0.5 * a if j % 2 else 0)
                y = j * a * np.sqrt(3.0/4.0)
                atoms.append([x, y, z])
        # B layer (offset)
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i * a + (0.5 * a if (j+1) % 2 else 0)
                y = j * a * np.sqrt(3.0/4.0) + a * np.sqrt(3.0)/6.0
                atoms.append([x, y, z + c/2.0])
    atoms = np.array(atoms)
    atoms -= atoms.mean(axis=0)
    dists = np.linalg.norm(atoms, axis=1)
    order = np.argsort(dists)
    atoms = atoms[order[:n]]
    return atoms


def _generate_icosahedral(n):
    """Generate n atoms in icosahedral shells."""
    atoms = np.array([[0.0, 0.0, 0.0]])
    # Icosahedron vertices
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0]
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    # Second shell: 30 vertices (cuboctahedron-like)
    cubo_verts = np.array([
        [1,1,0],[-1,1,0],[1,-1,0],[-1,-1,0],
        [1,0,1],[-1,0,1],[1,0,-1],[-1,0,-1],
        [0,1,1],[0,-1,1],[0,1,-1],[0,-1,-1],
        [1,1,0],[-1,-1,0],[1,-1,0],[-1,1,0],
        [0,1,1],[0,-1,-1],[0,1,-1],[0,-1,1],
        [1,0,1],[-1,0,-1],[1,0,-1],[-1,0,1],
        [1,1,0],[-1,-1,0],[-1,1,0],[1,-1,0],
        [0,1,1],[0,-1,-1]
    ])
    # Use proper 30-point second shell: edge midpoints of icosahedron
    edges = [(0,4),(0,5),(0,11),(0,10),(1,5),(1,11),(1,10),(1,4),
             (2,4),(2,5),(2,11),(2,10),(3,4),(3,5),(3,11),(3,10),
             (4,8),(4,9),(5,8),(5,9),(6,8),(6,9),(7,8),(7,9),
             (10,12),(10,13),(11,12),(11,13),(12,14),(12,15),(13,14),(13,15),
             (8,14),(8,15),(9,14),(9,15)]
    shell2 = []
    for i, j in edges:
        mid = (ico_verts[i] + ico_verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid) * 1.7013  # ~sqrt(3) for icosahedral second shell
        shell2.append(mid)
    shell2 = np.array(shell2)
    # Remove duplicates (each edge counted once, but some may overlap)
    unique_shell2 = [shell2[0]]
    for v in shell2[1:]:
        if all(np.linalg.norm(v - u) > 0.01 for u in unique_shell2):
            unique_shell2.append(v)
    shell2 = np.array(unique_shell2)
    # Third shell: 56 vertices (truncated icosahedron / soccer ball)
    # Approximate with points on a sphere
    if n > 1 + 12 + len(shell2):
        # Use Fibonacci sphere for third shell
        remaining = n - 1 - 12 - len(shell2)
        if remaining > 0:
            m = remaining
            golden = (1.0 + np.sqrt(5.0)) / 2.0
            shell3 = []
            for k in range(m):
                z = 1.0 - 2.0 * (k + 0.5) / m
                r = np.sqrt(1.0 - z*z)
                theta = golden * 2.0 * np.pi * k
                shell3.append([r*np.cos(theta), r*np.sin(theta), z] * 2.7015)
            shell3 = np.array(shell3)
        else:
            shell3 = np.array([]).reshape(0,3)
    else:
        shell3 = np.array([]).reshape(0,3)
    
    all_atoms = np.vstack([atoms, ico_verts, shell2, shell3])
    all_atoms -= all_atoms.mean(axis=0)
    dists = np.linalg.norm(all_atoms, axis=1)
    order = np.argsort(dists)
    all_atoms = all_atoms[order[:n]]
    return all_atoms


def _generate_compact_random(n, rng):
    """Generate a compact random cluster."""
    radius = n ** (1.0/3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 200:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
        attempts += 1
    # If we couldn't get enough, relax spacing
    if len(points) < n:
        points = []
        attempts = 0
        while len(points) < n and attempts < n * 500:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= 0.45 for q in points):
                    points.append(p)
            attempts += 1
    if len(points) < n:
        # Fallback: grid
        side = int(np.ceil(n ** (1.0/3.0)))
        grid = []
        for i in range(side):
            for j in range(side):
                for k in range(side):
                    grid.append([i, j, k])
        grid = np.array(grid[:n], dtype=float)
        grid = (grid - side/2.0) * 1.1
        return grid
    return np.array(points)


def _basin_hopping(positions, rho, rng, time_budget, max_steps=200):
    """Basin hopping with time control."""
    start_time = time.time()
    current = positions.copy()
    energy, current = relax(current, 'morse', rho)
    best = current.copy()
    best_energy = energy
    
    T = 0.5
    T_min = 0.01
    step_size = 0.3
    
    for step in range(max_steps):
        if time.time() - start_time > time_budget:
            break
        
        # Perturbation
        noise = rng.normal(0, step_size, current.shape)
        perturbed = current + noise
        
        # Relax perturbed
        try:
            new_energy, new_pos = relax(perturbed, 'morse', rho)
        except:
            new_energy = energy + 100
            new_pos = current
        
        # Metropolis acceptance
        delta = new_energy - energy
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current = new_pos
            energy = new_energy
        
        # Update best
        if energy < best_energy:
            best_energy = energy
            best = current.copy()
        
        # Anneal
        T = max(T_min, T * 0.97)
        step_size = max(0.1, step_size * 0.99)
    
    return best, best_energy


def construct(instance):
    """Find low-energy Morse cluster via structured starts + basin hopping."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 6))
    
    start_time = time.time()
    total_budget = 290.0  # Safety margin from 330s kill
    
    # Generate diverse initial structures
    rng = np.random.default_rng(n * 1000 + int(rho))
    
    initial_structures = []
    
    # 1. FCC-like
    try:
        fcc = _generate_fcc(n)
        if len(fcc) == n:
            initial_structures.append(fcc)
    except:
        pass
    
    # 2. HCP-like
    try:
        hcp = _generate_hcp(n)
        if len(hcp) == n:
            initial_structures.append(hcp)
    except:
        pass
    
    # 3. Icosahedral
    try:
        ico = _generate_icosahedral(n)
        if len(ico) == n:
            initial_structures.append(ico)
    except:
        pass
    
    # 4. Random compact (multiple)
    for _ in range(3):
        try:
            rand = _generate_compact_random(n, rng)
            if len(rand) == n:
                initial_structures.append(rand)
        except:
            pass
    
    # 5. Spherical shell
    try:
        # Place atoms on nested shells
        shells = []
        remaining = n
        shell_radii = [0.0]
        # First shell: 12 icosahedron
        if remaining > 1:
            phi = (1.0 + np.sqrt(5.0)) / 2.0
            ico = np.array([
                [0,1,phi],[0,-1,phi],[0,1,-phi],[0,-1,-phi],
                [1,phi,0],[-1,phi,0],[1,-phi,0],[-1,-phi,0],
                [phi,0,1],[-phi,0,1],[phi,0,-1],[-phi,0,-1]
            ])
            ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
            shells.append(ico)
            remaining -= 12
        # Second shell
        if remaining > 0:
            m = min(remaining, 30)
            # Fibonacci sphere
            golden = (1.0 + np.sqrt(5.0)) / 2.0
            pts = []
            for k in range(m):
                z = 1.0 - 2.0 * (k + 0.5) / m
                r = np.sqrt(1.0 - z*z)
                theta = golden * 2.0 * np.pi * k
                pts.append([r*np.cos(theta), r*np.sin(theta), z])
            pts = np.array(pts) * 1.7013
            shells.append(pts)
            remaining -= m
        # Third shell
        if remaining > 0:
            m = remaining
            golden = (1.0 + np.sqrt(5.0)) / 2.0
            pts = []
            for k in range(m):
                z = 1.0 - 2.0 * (k + 0.5) / m
                r = np.sqrt(1.0 - z*z)
                theta = golden * 2.0 * np.pi * k
                pts.append([r*np.cos(theta), r*np.sin(theta), z])
            pts = np.array(pts) * 2.7015
            shells.append(pts)
        
        sphere_atoms = np.vstack([np.array([[0,0,0]])] + shells)
        sphere_atoms -= sphere_atoms.mean(axis=0)
        if len(sphere_atoms) == n:
            initial_structures.append(sphere_atoms)
    except:
        pass
    
    # Relax each initial structure and keep the best
    best_energy = np.inf
    best_positions = None
    
    for i, init in enumerate(initial_structures):
        if time.time() - start_time > total_budget - 30:
            break
        try:
            e, pos = relax(init.copy(), 'morse', rho)
            if e < best_energy:
                best_energy = e
                best_positions = pos.copy()
        except:
            pass
    
    # If we have time, do basin hopping from the best structure
    if best_positions is not None and time.time() - start_time < total_budget - 20:
        remaining_time = total_budget - (time.time() - start_time) - 10
        if remaining_time > 5:
            try:
                bh_pos, bh_energy = _basin_hopping(
                    best_positions, rho, rng, 
                    time_budget=min(remaining_time, 60.0),
                    max_steps=100
                )
                if bh_energy < best_energy:
                    best_energy = bh_energy
                    best_positions = bh_pos
            except:
                pass
    
    # Final safety: if no valid positions, return a simple grid
    if best_positions is None:
        side = int(np.ceil(n ** (1.0/3.0)))
        grid = []
        for i in range(side):
            for j in range(side):
                for k in range(side):
                    grid.append([i, j, k])
        grid = np.array(grid[:n], dtype=float)
        grid = (grid - side/2.0) * 1.1
        best_positions = grid
        try:
            best_energy, best_positions = relax(best_positions, 'morse', rho)
        except:
            pass
    
    return best_positions
# EVOLVE-BLOCK-END
