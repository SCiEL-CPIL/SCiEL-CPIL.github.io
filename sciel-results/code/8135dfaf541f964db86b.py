"""Sutton-Chen transition-metal cluster global minima - multi-template + basin hopping."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _energy_grad(pos, kind):
    """Compute energy and gradient."""
    return sutton_chen_energy_gradient(pos, kind)


def _min_dist(pos):
    """Compute minimum pairwise distance."""
    if len(pos) < 2:
        return np.inf
    return pdist(pos, 'euclidean').min()


def _valid(pos):
    """Check if positions are valid (finite, min distance > 0.35)."""
    if not np.all(np.isfinite(pos)):
        return False
    if len(pos) < 2:
        return True
    return _min_dist(pos) > 0.35


def _fcc_cluster(n, spacing=1.0):
    """Generate n positions on an FCC lattice."""
    a = spacing * np.sqrt(2.0)  # lattice constant for nn distance = spacing
    points = []
    max_coord = int(np.ceil((n * 4) ** (1.0/3.0))) + 3
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * a / 2.0
                    points.append(p)
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    return points[order[:n]]


def _hcp_cluster(n, spacing=1.0):
    """Generate n positions on an HCP lattice."""
    a = spacing
    c = a * np.sqrt(8.0 / 3.0)
    points = []
    max_layer = int(np.ceil((n / 3) ** 0.5)) + 3
    for layer in range(-max_layer, max_layer + 1):
        z = layer * c
        for i in range(-max_layer, max_layer + 1):
            for j in range(-max_layer, max_layer + 1):
                if layer % 2 == 0:
                    x = (i + 0.5 * (j % 2)) * a
                    y = j * a * np.sqrt(3.0) / 2.0
                else:
                    x = i * a
                    y = (j + 0.5 * (i % 2)) * a * np.sqrt(3.0) / 2.0
                p = np.array([x, y, z])
                points.append(p)
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    return points[order[:n]]


def _mackay_cluster(n, spacing=1.0):
    """Generate Mackay icosahedral cluster positions."""
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    verts = verts / np.linalg.norm(verts[0]) * spacing
    
    # Shell sizes for Mackay: 1, 12, 42, 92, 140, ...
    shell_sizes = [1, 12, 42, 92, 140, 232, 312, 380]
    
    points = [np.zeros(3)]  # center
    
    # Shell 1: 12 vertices
    if n > 1:
        points.extend(verts.tolist())
    
    # Shell 2: 42 positions (octahedron + icosahedron vertices at r2)
    if n > 13:
        r2 = spacing * np.sqrt(3.0)
        # Octahedron vertices
        oct_verts = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=float) * r2
        # Additional icosahedron-like positions
        # For simplicity, use scaled icosahedron
        shell2 = oct_verts.tolist()
        # Add more points to reach 42
        # Use a second icosahedron rotated
        for angle in [0, np.pi/5, 2*np.pi/5]:
            R = np.array([[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]])
            rotated = verts @ R.T * (r2 / spacing)
            shell2.extend(rotated.tolist())
        
        # Take first 42 - 6 = 36 from the extra
        existing = len(points)
        needed = min(42, n - existing)
        for p in shell2:
            if len(points) - existing >= needed:
                break
            points.append(np.array(p))
    
    # Shell 3: 92 positions
    if n > 55:
        r3 = spacing * np.sqrt(5.0)
        # Generate points on sphere
        golden = (1 + np.sqrt(5)) / 2
        for i in range(92):
            theta = 2 * np.pi * i / golden
            phi = np.arccos(1 - 2 * (i + 0.5) / 92)
            p = r3 * np.array([np.sin(phi) * np.cos(theta),
                               np.sin(phi) * np.sin(theta),
                               np.cos(phi)])
            points.append(p)
    
    # Fill remaining with random points on outer shell
    while len(points) < n:
        r = spacing * (len(points) + 1) ** (1.0/3.0) * 0.9
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.arccos(np.random.uniform(-1, 1))
        p = r * np.array([np.sin(phi) * np.cos(theta),
                          np.sin(phi) * np.sin(theta),
                          np.cos(phi)])
        points.append(p)
    
    return np.array(points[:n])


def _random_compact(n, spacing=1.0, seed=None):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.9
    points = []
    attempts = 0
    min_d = 0.7 * spacing
    while len(points) < n and attempts < n * 200:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if len(points) == 0 or np.min(np.linalg.norm(np.array(points) - p, axis=1)) >= min_d:
                points.append(p)
        attempts += 1
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _growth_cluster(n, kind, spacing=1.0, seed=None):
    """Build cluster by growing from small relaxed cores (fast version)."""
    rng = np.random.default_rng(seed)
    
    if n <= 3:
        if n == 1:
            return np.zeros((1, 3))
        elif n == 2:
            return np.array([[0, 0, 0], [spacing, 0, 0]])
        else:
            return np.array([[0, 0, 0], [spacing, 0, 0], [spacing/2, spacing*np.sqrt(3)/2, 0]])
    
    # Start with 3-atom triangle
    current = np.array([[0, 0, 0], [spacing, 0, 0], [spacing/2, spacing*np.sqrt(3)/2, 0]])
    try:
        current, _ = relax(current, 'sutton_chen', kind)
    except:
        pass
    
    for target in range(4, n + 1):
        if len(current) >= target:
            break
        
        best_pos = None
        best_e = np.inf
        
        # Try positions around existing atoms (fewer candidates for speed)
        for atom in current:
            for r in [spacing * 0.95, spacing, spacing * 1.05]:
                for theta in np.linspace(0, 2*np.pi, 6, endpoint=False):
                    for phi in [0, np.pi/3, np.pi/2]:
                        new_p = atom + r * np.array([
                            np.sin(phi) * np.cos(theta),
                            np.sin(phi) * np.sin(theta),
                            np.cos(phi)
                        ])
                        trial = np.vstack([current, new_p])
                        if _valid(trial):
                            e, _ = _energy_grad(trial, kind)
                            if e < best_e:
                                best_e = e
                                best_pos = new_p
        
        # Random positions
        radius = spacing * target ** (1.0/3.0) * 1.0
        for _ in range(10):
            new_p = rng.uniform(-radius, radius, 3)
            trial = np.vstack([current, new_p])
            if _valid(trial):
                e, _ = _energy_grad(trial, kind)
                if e < best_e:
                    best_e = e
                    best_pos = new_p
        
        if best_pos is not None:
            current = np.vstack([current, best_pos])
            try:
                current, _ = relax(current, 'sutton_chen', kind)
            except:
                pass
        
        if time.time() > _DEADLINE - 30:
            break
    
    while len(current) < n:
        radius = spacing * n ** (1.0/3.0) * 1.0
        p = rng.uniform(-radius, radius, 3)
        current = np.vstack([current, p])
    
    return current[:n]


def _basin_hopping(pos, kind, n_iter=20, step_size=0.5, rng=None):
    """Basin hopping with adaptive step size."""
    if rng is None:
        rng = np.random.default_rng()
    
    best_pos = pos.copy()
    best_e, _ = _energy_grad(pos, kind)
    current_pos = pos.copy()
    current_e = best_e
    current_step = step_size
    
    for _ in range(n_iter):
        if time.time() > _DEADLINE - 10:
            break
        
        # Perturb
        trial = current_pos + rng.normal(0, current_step, current_pos.shape)
        
        # Relax
        try:
            e, relaxed = relax(trial, 'sutton_chen', kind)
        except:
            continue
        
        if not np.isfinite(e) or not _valid(relaxed):
            continue
        
        if e < current_e:
            current_e = e
            current_pos = relaxed
            current_step = max(0.1, current_step * 0.9)
            if e < best_e:
                best_e = e
                best_pos = relaxed
        else:
            # Metropolis acceptance with temperature decay
            delta = e - current_e
            T = max(0.01, 0.5 * np.exp(-_ / max(1, n_iter/3)))
            if rng.random() < np.exp(-delta / max(T, 1e-10)):
                current_pos = relaxed
                current_e = e
                current_step = max(0.1, current_step * 0.95)
            else:
                current_step = min(1.5, current_step * 1.05)
    
    return best_pos, best_e


def _de_step(positions, kind, bounds, pop_size, F, CR, iters, rng, deadline):
    """One round of differential evolution (vectorized mutation)."""
    n_atoms = len(positions)
    dim = n_atoms * 3
    x = positions.flatten()
    
    pop = np.zeros((pop_size, dim))
    pop[0] = x
    for i in range(1, pop_size):
        scale = 0.1 + 0.5 * (i / pop_size)
        pop[i] = x + rng.normal(0, scale, dim)
        pop[i] = np.clip(pop[i], bounds[0], bounds[1])
    
    energies = np.full(pop_size, 1e10)
    for i in range(pop_size):
        pos = pop[i].reshape(-1, 3)
        if _valid(pos):
            try:
                e, _ = _energy_grad(pos, kind)
                if np.isfinite(e):
                    energies[i] = e
            except:
                pass
    
    best_idx = np.argmin(energies)
    
    for it in range(iters):
        if time.time() > deadline - 15:
            break
        
        new_pop = pop.copy()
        new_energies = energies.copy()
        
        for i in range(pop_size):
            a, b, c = rng.choice(pop_size, 3, replace=False)
            mutant = pop[a] + F * (pop[b] - pop[c])
            
            # Vectorized crossover
            mask = rng.random(dim) < CR
            mask[rng.integers(dim)] = True
            trial = np.where(mask, mutant, pop[i])
            trial = np.clip(trial, bounds[0], bounds[1])
            
            pos = trial.reshape(-1, 3)
            if _valid(pos):
                try:
                    e, _ = _energy_grad(pos, kind)
                    if np.isfinite(e) and e <= new_energies[i]:
                        new_pop[i] = trial
                        new_energies[i] = e
                except:
                    pass
        
        pop = new_pop
        energies = new_energies
        best_idx = np.argmin(energies)
    
    return pop[best_idx].reshape(-1, 3), energies[best_idx]


_DEADLINE = time.time() + 330.0  # Set once at module load; 330s total budget


def construct(instance):
    """Multi-template + basin hopping for Sutton-Chen clusters."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho", "12-6"))
    
    spacing = 0.72
    rng = np.random.default_rng(42)
    
    best_energy = np.inf
    best_positions = None
    
    def _try_relax(pos, label=""):
        nonlocal best_energy, best_positions
        if time.time() > _DEADLINE - 20:
            return
        if not _valid(pos):
            return
        try:
            e, relaxed = relax(pos, potential, kind)
            if np.isfinite(e) and _valid(relaxed):
                if e < best_energy:
                    best_energy = e
                    best_positions = relaxed
        except:
            pass
    
    # 1. FCC template
    try:
        fcc = _fcc_cluster(size, spacing)
        if _valid(fcc):
            _try_relax(fcc, "fcc")
    except:
        pass
    
    # 2. HCP template
    try:
        hcp = _hcp_cluster(size, spacing)
        if _valid(hcp):
            _try_relax(hcp, "hcp")
    except:
        pass
    
    # 3. Mackay template
    try:
        mackay = _mackay_cluster(size, spacing)
        if _valid(mackay):
            _try_relax(mackay, "mackay")
    except:
        pass
    
    # 4. Growth-based template
    try:
        growth = _growth_cluster(size, kind, spacing, seed=42)
        if _valid(growth):
            _try_relax(growth, "growth")
    except:
        pass
    
    # 5. Random compact templates
    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        if time.time() > _DEADLINE - 30:
            break
        try:
            rand = _random_compact(size, spacing, seed)
            if _valid(rand):
                _try_relax(rand, f"rand{seed}")
        except:
            pass
    
    # 6. Basin hopping from best
    if best_positions is not None and time.time() < _DEADLINE - 50:
        try:
            bh_pos, bh_e = _basin_hopping(best_positions, kind, n_iter=25, step_size=0.4, rng=rng)
            if np.isfinite(bh_e) and _valid(bh_pos):
                if bh_e < best_energy:
                    best_energy = bh_e
                    best_positions = bh_pos
        except:
            pass
    
    # 7. Basin hopping from second-best (if available, use FCC as alternate start)
    if best_positions is not None and time.time() < _DEADLINE - 40:
        try:
            alt = _fcc_cluster(size, spacing)
            if _valid(alt):
                bh_pos2, bh_e2 = _basin_hopping(alt, kind, n_iter=15, step_size=0.5, rng=rng)
                if np.isfinite(bh_e2) and _valid(bh_pos2):
                    if bh_e2 < best_energy:
                        best_energy = bh_e2
                        best_positions = bh_pos2
        except:
            pass
    
    # 8. DE from best
    if best_positions is not None and time.time() < _DEADLINE - 30:
        radius = spacing * size ** (1.0/3.0) * 1.5
        bounds = (-radius, radius)
        pop_size = min(15, max(8, size // 3))
        try:
            de_pos, de_e = _de_step(best_positions, kind, bounds, pop_size, 0.8, 0.7, 6, rng, _DEADLINE)
            if np.isfinite(de_e) and _valid(de_pos):
                if de_e < best_energy:
                    best_energy = de_e
                    best_positions = de_pos
        except:
            pass
    
    # 9. Final polish
    if best_positions is not None and time.time() < _DEADLINE - 5:
        try:
            e, relaxed = relax(best_positions, potential, kind)
            if np.isfinite(e) and _valid(relaxed):
                if e < best_energy:
                    best_energy = e
                    best_positions = relaxed
        except:
            pass
    
    # Fallback
    if best_positions is None:
        best_positions = _random_compact(size, spacing, 99)
        try:
            best_energy, best_positions = relax(best_positions, potential, kind)
        except:
            pass
    
    # Ensure validity
    if not _valid(best_positions):
        for i in range(len(best_positions)):
            for j in range(i + 1, len(best_positions)):
                d = best_positions[i] - best_positions[j]
                dist = np.linalg.norm(d)
                if dist < 0.35:
                    if dist < 1e-10:
                        d = rng.normal(0, 0.1, 3)
                        dist = np.linalg.norm(d)
                    d = d / dist
                    best_positions[i] += 0.18 * d
                    best_positions[j] -= 0.18 * d
    
    return best_positions
