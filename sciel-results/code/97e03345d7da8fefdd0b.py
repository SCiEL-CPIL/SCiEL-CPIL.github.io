"""Program for Sutton-Chen TM cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 285.0  # Safety margin before 300s kill


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _is_valid(positions, min_dist=0.35):
    """Check if positions are valid (finite, no atoms closer than min_dist)."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    for i in range(n - 1):
        d = np.linalg.norm(positions[i + 1:] - positions[i], axis=1)
        if np.any(d < min_dist):
            return False
    return True


def _generate_fcc(n):
    """Generate a cluster of n atoms in an FCC-like structure."""
    max_r = int(np.ceil(n ** (1 / 3))) + 3
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * (1.0 / np.sqrt(2.0))
                    candidates.append((np.linalg.norm(p), p))
    
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _generate_bcc(n):
    """Generate a cluster of n atoms in a BCC-like structure."""
    max_r = int(np.ceil(n ** (1 / 3))) + 3
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k])
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5])
                candidates.append((np.linalg.norm(p1), p1))
                candidates.append((np.linalg.norm(p2), p2))
    
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _generate_hcp(n):
    """Generate a cluster of n atoms in an HCP-like structure."""
    a = 1.0 / np.sqrt(2.0)
    c = a * np.sqrt(8.0 / 3.0)
    
    candidates = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                x = a * (i + j / 2.0)
                y = a * np.sqrt(3.0) / 2.0 * j
                z = c * k
                candidates.append((np.sqrt(x*x + y*y + z*z), np.array([x, y, z])))
                x2 = a * (i + j / 2.0 + 0.5)
                y2 = a * np.sqrt(3.0) / 2.0 * (j + 1.0 / np.sqrt(3.0))
                z2 = c * (k + 1.0 / 3.0)
                candidates.append((np.sqrt(x2*x2 + y2*y2 + z2*z2), np.array([x2, y2, z2])))
    
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _generate_mackay_icosahedron(n):
    """Generate a Mackay icosahedral cluster with shells."""
    phi = (1 + np.sqrt(5)) / 2
    a = 1.0 / np.sqrt(2.0)
    
    # Icosahedral vertices
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Generate shell points
    candidates = [(0.0, np.zeros(3))]  # center
    
    # Shell 1: 12 vertices
    for v in verts:
        candidates.append((np.linalg.norm(v), v.copy()))
    
    # Shell 2: 30 edge midpoints
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < 1.2:  # adjacent vertices
                mid = (verts[i] + verts[j]) / 2
                mid = mid / np.linalg.norm(mid)
                candidates.append((np.linalg.norm(mid), mid))
    
    # Shell 3: 20 face centers
    # Find faces (triangles)
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[k] - verts[i])
                if d1 < 1.2 and d2 < 1.2 and d3 < 1.2:
                    center = (verts[i] + verts[j] + verts[k]) / 3
                    center = center / np.linalg.norm(center)
                    faces.append(center)
    
    for center in faces:
        candidates.append((np.linalg.norm(center), center))
    
    # Shell 4: more points from scaling
    for scale in [1.5, 2.0]:
        for v in verts:
            p = v * scale
            candidates.append((np.linalg.norm(p), p))
    
    # Add FCC-like points for larger shells
    max_r = int(np.ceil(n ** (1/3))) + 2
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * (1.0 / np.sqrt(2.0))
                    candidates.append((np.linalg.norm(p), p))
    
    # Sort by distance and take first n
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _basin_hopping(positions, kind, max_iter=50, step_size=0.3, rng=None, max_restarts=5):
    """
    Basin hopping algorithm for finding global minima.
    Each iteration: perturb positions, relax, accept if energy decreases.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n_atoms = len(positions)
    current_pos = positions.copy()
    current_energy = np.inf
    
    # Initial relaxation
    try:
        current_energy, current_pos = relax(current_pos, 'sutton_chen', kind)
        if not np.isfinite(current_energy) or not _is_valid(current_pos):
            current_energy = np.inf
    except Exception:
        current_energy = np.inf
    
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    for iteration in range(max_iter):
        if _time_left() < 15.0:
            break
        
        # Perturb: add random displacement
        step = step_size * (1.0 + 0.5 * np.exp(-iteration / 10.0))
        noise = rng.normal(0, step, (n_atoms, 3))
        perturbed = current_pos + noise
        
        # Relax the perturbed structure
        try:
            new_energy, new_pos = relax(perturbed, 'sutton_chen', kind)
            if not np.isfinite(new_energy) or not _is_valid(new_pos):
                continue
        except Exception:
            continue
        
        # Accept if energy decreased (or with Metropolis criterion for early iterations)
        if new_energy < current_energy:
            current_energy = new_energy
            current_pos = new_pos
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
        elif iteration < 10 and rng.random() < np.exp(-(new_energy - current_energy) / (0.1 + 0.05 * iteration)):
            # Metropolis acceptance for early iterations
            current_energy = new_energy
            current_pos = new_pos
    
    return best_energy, best_pos


def _polish(positions, kind, max_iter=500, tol=1e-12):
    """High-precision polish pass using gradient-based optimization."""
    def energy_fn(x):
        pos = x.reshape(-1, 3)
        e, _ = sutton_chen_energy_gradient(pos, kind)
        return e
    
    def gradient_fn(x):
        pos = x.reshape(-1, 3)
        _, g = sutton_chen_energy_gradient(pos, kind)
        return g.ravel()
    
    try:
        result = minimize(
            energy_fn,
            positions.ravel(),
            jac=gradient_fn,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'gtol': tol, 'ftol': tol, 'maxls': 50}
        )
        if result.success and np.isfinite(result.fun):
            polished = result.x.reshape(-1, 3)
            if _is_valid(polished):
                return result.fun, polished
    except Exception:
        pass
    
    return np.inf, positions


def _growth_construct(n, kind, rng):
    """
    Deterministic growth algorithm: build cluster by adding atoms one at a time
    to the lowest-energy interstitial site of the (N-1)-atom cluster.
    """
    if n <= 1:
        return np.zeros((n, 3))
    
    # Start with 2 atoms at optimal distance
    best_energy = np.inf
    best_pos = None
    
    for d0 in [0.9, 1.0, 1.1, 1.2]:
        pos = np.array([[0, 0, 0], [d0, 0, 0]])
        try:
            energy, relaxed = relax(pos, 'sutton_chen', kind)
            if np.isfinite(energy) and _is_valid(relaxed):
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed
        except Exception:
            continue
    
    if best_pos is None:
        best_pos = np.array([[0, 0, 0], [1.0, 0, 0]])
    
    current = best_pos.copy()
    
    for target_n in range(3, n + 1):
        if _time_left() < 40.0:
            break
        
        candidates = []
        
        # For each existing atom, try positions at various offsets
        for i in range(len(current)):
            for trial in range(8):
                direction = rng.normal(0, 1, 3)
                direction /= np.linalg.norm(direction)
                for dist in [0.8, 0.9, 1.0, 1.1, 1.2]:
                    new_pos = current[i] + direction * dist
                    candidates.append(new_pos)
        
        # Also try positions in gaps
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                midpoint = (current[i] + current[j]) / 2
                bond_vec = current[j] - current[i]
                bond_len = np.linalg.norm(bond_vec)
                if bond_len > 0.1:
                    perp = np.cross(bond_vec, [0, 0, 1])
                    if np.linalg.norm(perp) < 1e-6:
                        perp = np.cross(bond_vec, [1, 0, 0])
                    perp /= np.linalg.norm(perp)
                    for offset in [0.3, 0.5, 0.7]:
                        candidates.append(midpoint + perp * offset)
                        candidates.append(midpoint - perp * offset)
        
        best_e = np.inf
        best_p = None
        
        rng.shuffle(candidates)
        for candidate in candidates[:15]:
            if _time_left() < 35.0:
                break
            
            test_pos = np.vstack([current, candidate.reshape(1, 3)])
            try:
                energy, relaxed = relax(test_pos, 'sutton_chen', kind)
                if np.isfinite(energy) and _is_valid(relaxed):
                    if energy < best_e:
                        best_e = energy
                        best_p = relaxed
            except Exception:
                continue
        
        if best_p is not None:
            current = best_p
        else:
            new_atom = current[0] + rng.normal(0, 0.5, 3)
            current = np.vstack([current, new_atom.reshape(1, 3)])
    
    return current


def construct(instance):
    n = instance["n"]
    kind = instance["kind"]
    
    best_energy = np.inf
    best_pos = None
    
    rng = np.random.default_rng(42 + n)
    
    # Generate diverse initial structures
    initial_structures = []
    
    # 1. Growth-based construction
    try:
        growth_pos = _growth_construct(n, kind, rng)
        if _is_valid(growth_pos):
            initial_structures.append(growth_pos)
    except Exception:
        pass
    
    # 2. FCC template
    try:
        fcc_pos = _generate_fcc(n)
        if _is_valid(fcc_pos):
            initial_structures.append(fcc_pos)
    except Exception:
        pass
    
    # 3. BCC template
    try:
        bcc_pos = _generate_bcc(n)
        if _is_valid(bcc_pos):
            initial_structures.append(bcc_pos)
    except Exception:
        pass
    
    # 4. HCP template
    try:
        hcp_pos = _generate_hcp(n)
        if _is_valid(hcp_pos):
            initial_structures.append(hcp_pos)
    except Exception:
        pass
    
    # 5. Mackay icosahedron
    try:
        ico_pos = _generate_mackay_icosahedron(n)
        if _is_valid(ico_pos):
            initial_structures.append(ico_pos)
    except Exception:
        pass
    
    # 6. Random compact structures
    for i in range(3):
        radius = 1.2 + 0.3 * i
        pos = rng.uniform(-radius, radius, (n, 3))
        if _is_valid(pos):
            initial_structures.append(pos)
    
    # For each initial structure, run basin hopping
    for init_pos in initial_structures:
        if _time_left() < 20.0:
            break
        
        # First, relax the initial structure
        try:
            energy, relaxed = relax(init_pos, 'sutton_chen', kind)
            if np.isfinite(energy) and _is_valid(relaxed):
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed.copy()
        except Exception:
            continue
        
        # Then run basin hopping from this relaxed structure
        if _time_left() > 30.0:
            try:
                bh_energy, bh_pos = _basin_hopping(
                    relaxed, kind,
                    max_iter=40,
                    step_size=0.25,
                    rng=rng
                )
                if np.isfinite(bh_energy) and _is_valid(bh_pos):
                    if bh_energy < best_energy:
                        best_energy = bh_energy
                        best_pos = bh_pos.copy()
            except Exception:
                continue
    
    # Final high-precision polish
    if best_pos is not None and _time_left() > 15.0:
        try:
            polished_energy, polished_pos = _polish(best_pos, kind, max_iter=300, tol=1e-12)
            if polished_energy < best_energy:
                best_energy = polished_energy
                best_pos = polished_pos
        except Exception:
            pass
    
    # Fallback
    if best_pos is None:
        best_pos = _generate_fcc(n)
        try:
            _, best_pos = relax(best_pos, 'sutton_chen', kind)
        except Exception:
            pass
    
    return best_pos

# EVOLVE-BLOCK-END
